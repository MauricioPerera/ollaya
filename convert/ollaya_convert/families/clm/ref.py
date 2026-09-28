"""`clm-v1`: Contrastive-LM's CLM-v0.1-8B, the reference (PyTorch, fp32 compute on the BF16 weights).

    CLM_SRC=/path/to/CLM/src uv run python -m ollaya_convert.families.clm.ref

CLM (https://github.com/Contrastive-LM/CLM) scores a question's options by contrast, not by
generation:

    state text   = schema.state_text(state, question.instructions)       (context, blank line, question)
    option texts = schema.candidates(question)                            (the option's description, a
                                                                           score level, or "true: ..." for noul)
    embedding(t) = Qwen3-8B's final hidden state (after the last RMSNorm) at t's last token, L2-normalised
                   (upstream: vLLM `--runner pooling`, LAST pooling; the tokenizer adds no special tokens)
    z_state      = normalize(state_head(embedding(state text)))
    z_option     = normalize(action_head(embedding(option text)))
    logits       = min(exp(logit_scale), 100) * z_option . z_state
    probabilities = softmax(logits); answers = schema.answer_from_logits

Upstream serves the encoder in BF16 through vLLM. Ollaya's graph computes in fp32 on the BF16 weights
(as for decider and kev), so the goldens are that computation: every Linear, norm and attention in fp32,
the weights exactly the checkpoint's BF16 values. Upstream truncates a text to its last 2,048 tokens;
Ollaya rejects a longer text instead (TypeSafe routes never answer from a truncated state, #16).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = "Contrastive-LM/CLM-v0.1-8B"
REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
HEAD_FILE = "CLM_v0.1-8B.pt"
BASE = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_TOKENS = 2048
SCALE_MAX = 100.0


def schema():
    """Upstream's `clm.schema` (text building and answers), from CLM_SRC."""
    src = os.environ.get("CLM_SRC", os.path.expanduser("~/agents/clm-ref/src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    from clm import schema as s
    return s


def fp32_on_bf16(model):
    """Run every Linear and the embedding in fp32 on the BF16 weights, as the graph does."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.forward = (lambda m: lambda x: F.linear(x.float(), m.weight.float(),
                                                        None if m.bias is None else m.bias.float()))(mod)
    emb = model.embed_tokens
    emb.forward = (lambda e: lambda ids: F.embedding(ids, e.weight).float())(emb)
    for mod in model.modules():
        if type(mod).__name__.endswith("RMSNorm"):
            mod.forward = (lambda m: lambda x: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                                                         + m.variance_epsilon)) * m.weight.float())(mod)
    return model


class Reference:
    def __init__(self, base_root, head_path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..llm_common.qwen3 import Qwen3Trunk
        self.s = schema()
        self.tok = AutoTokenizer.from_pretrained(base_root)
        lm = AutoModelForCausalLM.from_pretrained(base_root, torch_dtype=torch.bfloat16)
        self.trunk = Qwen3Trunk(fp32_on_bf16(lm.model)).to(device).eval()
        self.trunk.inv_freq = self.trunk.inv_freq.float()
        self.device = device
        ck = torch.load(head_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        from clm.heads import make_head
        kw = dict(width=cfg["width"], depth=cfg["depth"], proj=ck.get("projection_dim", 512),
                  activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
                  residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", 4096))
        self.state_head, self.action_head = make_head(**kw), make_head(**kw)
        self.state_head.load_state_dict(ck["state_head"])
        self.action_head.load_state_dict(ck["action_head"])
        self.state_head.eval().to(device).float()
        self.action_head.eval().to(device).float()
        self.scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=SCALE_MAX))
        self.cache = {}

    def ids(self, text):
        return self.tok(text).input_ids

    @torch.no_grad()
    def embed(self, text):
        """L2-normalised last-token embedding (fp32); rejects texts over MAX_TOKENS."""
        if text in self.cache:
            return self.cache[text]
        ids = self.ids(text)
        if len(ids) > MAX_TOKENS:
            raise ValueError("text of %d tokens is longer than %d" % (len(ids), MAX_TOKENS))
        h = self.trunk(torch.tensor([ids], device=self.device))[0, -1].float()
        e = F.normalize(h, dim=-1)
        self.cache[text] = e
        return e

    @torch.no_grad()
    def answer(self, state, questions):
        """-> (pairs, logits per question, upstream answers)."""
        pairs = self.s.build_pairs(state, questions)
        logits, answers = {}, {}
        for qid, (st, keys, texts) in pairs.items():
            zs = F.normalize(self.state_head(self.embed(st)[None]), dim=-1)[0]
            za = F.normalize(self.action_head(torch.stack([self.embed(t) for t in texts])), dim=-1)
            lg = (self.scale * (za @ zs)).double().cpu().numpy()
            logits[qid] = lg
            answers[qid] = self.s.answer_from_logits(questions[qid], keys, lg.tolist())
        return pairs, logits, answers


if __name__ == "__main__":
    root = os.path.expanduser("~/agents/clm-work")
    r = Reference(os.path.join(root, "qwen3-8b"), os.path.join(root, HEAD_FILE))
    qs = {"urgency": {"type": "noul", "instructions": "Is this urgent?"},
          "department": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
          "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                          "criteria": ["Calm", "Frustrated", "Very angry"]}}
    _, lg, ans = r.answer("Customer: my invoice was charged twice and nobody answers the phone!", qs)
    for k, v in ans.items():
        print(k, v)
