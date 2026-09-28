"""Export Contrastive-LM/CLM-v0.1-8B (Qwen3-8B encoder + state/action heads) to a weightless ONNX graph.

    CLM_SRC=/path/to/CLM/src uv run --with requests python -m ollaya_convert.families.clm.export \
        --out out/clm-8b --base BASE --head HEAD.pt

Graph (layout `clm-v1`, see ref.py and docs/families/clm.md):
    inputs   input_ids  int64   [rows, seq]  one text per row, right-padded with any id
             last_pos   int64   [rows]       position of the row's last token
             action     int64   [rows]       1: an option text (action head), 0: a state text (state head)
    outputs  z          float32 [rows, 512]  the row's L2-normalised projection

The runtime scores an option by `scale * z_option . z_state` and takes the softmax over a question's
options; `scale` is in decision.json. Weights reference the author's files byte for byte: the five
BF16 shards of Qwen/Qwen3-8B (widened by Cast, per forward pass with weights_in_memory bf16) and the
heads, F32 tensors stored uncompressed in the torch zip `CLM_v0.1-8B.pt`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch
import torch.nn.functional as F

from ..llm_common import onnx_export as ox
from ..llm_common.qwen3 import Qwen3Trunk
from ...weightless_sharded import safetensors_source, torchzip_source
from . import ref

INPUT_NAMES = ["input_ids", "last_pos", "action"]
OUTPUT_NAMES = ["z"]
BASE_FILES = ["model-%05d-of-00005.safetensors" % i for i in range(1, 6)]


class ClmGraph(torch.nn.Module):
    def __init__(self, text_model, state_head, action_head):
        super().__init__()
        self.trunk = Qwen3Trunk(text_model)
        self.state_head = state_head
        self.action_head = action_head

    def forward(self, input_ids, last_pos, action):
        h = self.trunk(input_ids).float()
        e = F.normalize(h[torch.arange(h.shape[0], device=h.device), last_pos], dim=-1)
        zs = F.normalize(self.state_head(e), dim=-1)
        za = F.normalize(self.action_head(e), dim=-1)
        return torch.where(action.unsqueeze(-1) == 1, za, zs)


def rename(name):
    """Graph initializer name -> checkpoint tensor names (Qwen3-8B shards or the head zip)."""
    if name.startswith("trunk.m."):
        return ["model." + name[len("trunk.m."):]]
    return [name]   # state_head.* / action_head.*: their dotted paths in the head zip


def export(out_dir, base_dir, head_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_dir)
    lm = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    ck = torch.load(head_path, map_location="cpu", weights_only=False)
    s = ref.schema()
    from clm.heads import make_head
    cfg = ck["cfg"]
    kw = dict(width=cfg["width"], depth=cfg["depth"], proj=ck.get("projection_dim", 512),
              activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
              residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", 4096))
    sh, ah = make_head(**kw), make_head(**kw)
    sh.load_state_dict(ck["state_head"])
    ah.load_state_dict(ck["action_head"])
    graph = ClmGraph(lm.model, sh.float(), ah.float()).eval()

    # two rows of different lengths, one state and one option: every dynamic axis > 1
    pairs = s.build_pairs("The customer was charged twice for order A-104 and wants the duplicate refunded.",
                          {"a": {"type": "choice", "instructions": "Which team?",
                                 "criteria": {"billing": "charges and refunds", "tech": "bugs"}}})
    st, _, texts = pairs["a"]
    rows = [tok(st).input_ids, tok(texts[0]).input_ids]
    T = max(len(r) for r in rows) + 3
    ids = torch.full((2, T), tok.pad_token_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
    args = (ids, torch.tensor([len(r) - 1 for r in rows]), torch.tensor([0, 1]))
    with torch.no_grad():
        got = graph(*args)
        e = [F.normalize(lm.model(input_ids=torch.tensor([r])).last_hidden_state[0, -1], dim=-1) for r in rows]
        want = torch.stack([F.normalize(sh(e[0]), dim=-1), F.normalize(ah(e[1]), dim=-1)])
    print("eager graph vs HF + heads: %.2e" % float((got - want).abs().max()))

    R = torch.export.Dim("rows", min=1, max=4096)
    S = torch.export.Dim("seq", min=2, max=ref.MAX_TOKENS + 64)
    dyn = {"input_ids": {0: R, 1: S}, "last_pos": {0: R}, "action": {0: R}}
    tmp = ox.scratch_dir("clm-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    del graph, lm, got, want
    gc.collect()

    sources = [safetensors_source(f, os.path.join(base_dir, f), repo=ref.BASE, revision=ref.BASE_REVISION, filename=f)
               for f in BASE_FILES]
    sources.append(torchzip_source(ref.HEAD_FILE, head_path, repo=ref.REPO, revision=ref.REVISION, filename=ref.HEAD_FILE))
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(base_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=ref.SCALE_MAX))
    decision = {
        "engine": "onnx",
        "family": "clm",
        "layout": "clm-v1",
        "upstream": {"repo": ref.REPO, "revision": ref.REVISION, "head": ref.HEAD_FILE, "base": ref.BASE,
                     "base_revision": ref.BASE_REVISION, "code": "https://github.com/Contrastive-LM/CLM"},
        "contract": {
            "inputs": {"input_ids": {"dtype": "int64", "shape": ["rows", "seq"], "note": "one text per row; right-pad with any id"},
                       "last_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the row's last token"},
                       "action": {"dtype": "int64", "shape": ["rows"], "note": "1 = option text (action head), 0 = state text"}},
            "outputs": {"z": {"dtype": "float32", "shape": ["rows", kw["proj"]], "note": "L2-normalised projection"}},
            "positions": "0..seq-1, implicit",
            "attention": "causal; no mask input (right padding cannot reach earlier positions)",
        },
        "scale": scale,
        "max_text_tokens": ref.MAX_TOKENS,
        "special_tokens": {"pad": tok.pad_token_id, "add_special_tokens": False},
        "templates": {
            "state": "clm.schema.state_text: to_text(state).strip() + '\\n\\n' + to_text(instructions).strip()",
            "choice_option": "to_text(description), or the key when the description is null or ''",
            "score_option": "to_text(level)",
            "noul_options": ["false: {description or 'No. This is false: ' + instructions}",
                             "true: {description or 'Yes. This is true: ' + instructions}"],
            "to_text": "clm.schema.to_text: str verbatim; bool true/false; numbers Python str(); dict 'k: v' "
                       "(top level joined by a blank line, nested indented 2); list '- item' lines",
        },
        "option_logits": "scale * z[option row] . z[state row], in option order",
        "opset": ox.OPSET,
        "precision": "fp32 compute; Qwen3-8B weights BF16, widened by Cast; heads F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "none upstream: the probabilities are softmax(scale * cosine), scale = exp(logit_scale) capped at 100"}
    files = {
        "model": "clm-8b",
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya", "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights/base", ref.BASE, ref.BASE_REVISION, f, os.path.join(base_dir, f), location=f)
              for f in BASE_FILES],
            ox.file_entry("weights/head", ref.REPO, ref.REVISION, ref.HEAD_FILE, head_path, location=ref.HEAD_FILE)
            | {"note": "torch zip; the head tensors are stored uncompressed and referenced by byte offset"},
            ox.file_entry("tokenizer", ref.BASE, ref.BASE_REVISION, "tokenizer.json", os.path.join(base_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20),
          "unused", files["unused_checkpoint_tensors"], "weights_in_memory", decision["weights_in_memory"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="local snapshot of Qwen/Qwen3-8B at the pinned revision")
    ap.add_argument("--head", required=True, help="CLM_v0.1-8B.pt at the pinned revision")
    a = ap.parse_args()
    export(a.out, a.base, a.head)


if __name__ == "__main__":
    main()
