"""Check the `jeeves-markers-v1` port (layout.py) against the upstream Encoder, id for id, without the model.

    JEEVES_SRC=<checkout> uv run python -m ollaya_convert.families.jeeves.check <model snapshot>

Every case of the shared set: upstream `parse_request` + `Encoder` (the no-thinking ids and readout
positions) against `JeevesLayout` on the tokenizer the runtime loads (`ref.TOKENIZER`) with `decision_for` below; rejections must
match.
"""
from __future__ import annotations

import argparse
import os
import sys

import tokenizers

from ..llm_common import cases
from . import ref
from .layout import JeevesLayout, LayoutError

SENTINEL = "@@JEEVES_CONTENT@@"


def decision_for(enc):
    """The layout's fields of decision.json from the upstream Encoder."""
    from loader.dataloader import DECIDE, OPT, OPT_END, Q, STATE, THINK_END

    text = enc.chat(SENTINEL)
    pre, post = text.split(SENTINEL)
    return {
        "template": {"pre": pre, "post": post},
        "markers": {"state": STATE, "question": Q, "option_open": OPT, "option_close": OPT_END, "decide": DECIDE,
                    "think_end": THINK_END},
        "marker_ids": {"state": enc.state_id, "question": enc.q_id, "option_open": enc.opt_id,
                       "option_close": enc.opt_end_id, "decide": enc.decide_id, "think_end": enc.think_end_id},
        "empty_think": "\n",
        "max_row_tokens": ref.MAX_LEN,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--td", type=int, default=200)
    ap.add_argument("--tokenizer", required=True, help="the tokenizer.json the runtime loads (ref.TOKENIZER)")
    a = ap.parse_args()
    enc = ref.encoder(a.snapshot)
    lay = JeevesLayout(tokenizers.Tokenizer.from_file(a.tokenizer), decision_for(enc))
    same = rejected = 0
    bad = []
    for cid, state, qs in cases.edge_cases() + cases.typed_decisions(a.td):
        try:
            up, _ = ref.encode(enc, state, qs)
            ue = None
        except ref.RequestError as e:
            up, ue = None, e
        try:
            ours, _ = lay.encode(state, qs)
            oe = None
        except LayoutError as e:
            ours, oe = None, e
        if ue is not None and oe is not None:
            rejected += 1
        elif (ue is None) != (oe is None):
            bad.append((cid, "rejection differs: upstream %s, port %s" % (ue, oe)))
        elif up != ours:
            first = next((i for i, (x, y) in enumerate(zip(up, ours)) if x != y), None)
            u, o = up[first]["ids"], ours[first]["ids"]
            at = next((i for i, (x, y) in enumerate(zip(u, o)) if x != y), min(len(u), len(o)))
            bad.append((cid, "question %s: %d vs %d ids, first difference at %d: %s vs %s" % (
                first, len(u), len(o), at, u[max(0, at - 3):at + 3], o[max(0, at - 3):at + 3])))
        else:
            same += 1
    print("identical requests %d | both reject %d | mismatches %d" % (same, rejected, len(bad)))
    for b in bad[:15]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
