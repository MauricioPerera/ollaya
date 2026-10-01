"""`jeeves-markers-v1`: the request -> token rows layout of PostHog/jeeves (no-thinking mode), written the
way the Rust port is (HF `tokenizers` and `decision.json` only). `check.py` and `goldens.py` compare it
id for id with the upstream Encoder.

One row per question:

    content = STATE + sanitize(render(state)) + "\\n" + Q + sanitize(render(instructions)) + "\\n" + block
    block   = "".join(OPT + sanitize(option) + OPT_END + "\\n" for option in options)
    ids     = tok(pre + content.strip() + post) + tok("\\n") + tok("</think>\\n\\n" + block + DECIDE)
    opt_pos = the OPT_END positions after the last </think>;  decide_pos = the last position

`pre` and `post` are Qwen's chat template around one user message with thinking on (the prompt ends in
"<think>\\n"). Every text is tokenized whole with special tokens parsed, as upstream does; sanitize
rewrites `<|name|>` to `<¦name¦>`, so user text cannot form the markers. render and the option texts
are Kev's (`prep/format.py`): noul reads "no"/"yes" (false, true), a choice "name" or "name: desc", a
score its levels.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from ..kev.layout import option_text, render

CONTROL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MAX_OPTIONS = 255


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def sanitize(text: str) -> str:
    return CONTROL_RE.sub(r"<¦\1¦>", text)


def options_of(qid, q):
    """inference.api.parse_question + prep.format.Question.options."""
    if not isinstance(q, dict):
        raise LayoutError("question %r must be an object" % qid)
    t = q.get("type")
    if t not in ("choice", "noul", "score"):
        raise LayoutError("question %r: type must be one of choice, noul, score" % qid)
    if set(q) - {"type", "instructions", "criteria"}:
        raise LayoutError("question %r: unknown fields" % qid)
    c = q.get("criteria")
    if t == "choice":
        if not (isinstance(c, dict) and 1 <= len(c) <= MAX_OPTIONS):
            raise LayoutError("question %r: choice criteria must be an object with 1..%d options" % (qid, MAX_OPTIONS))
        return t, [option_text(k, v) for k, v in c.items()]
    if t == "noul":
        if c is not None and not (isinstance(c, dict) and set(c) <= {"true", "false"}):
            raise LayoutError("question %r: noul criteria may only describe true and false" % qid)
        c = c or {}
        return t, [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
    if not (isinstance(c, list) and 1 <= len(c) <= MAX_OPTIONS):
        raise LayoutError("question %r: score criteria must be a list of 1..%d levels" % (qid, MAX_OPTIONS))
    return t, [render(x) for x in c]


class JeevesLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.pre, self.post = decision["template"]["pre"], decision["template"]["post"]
        m = decision["markers"]
        self.state, self.q, self.opt, self.opt_end, self.decide = m["state"], m["question"], m["option_open"], m["option_close"], m["decide"]
        self.think_end = m["think_end"]
        ids = decision["marker_ids"]
        self.opt_end_id, self.think_end_id = ids["option_close"], ids["think_end"]
        self.max_len = decision["max_row_tokens"]

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, state, questions):
        """-> (rows, meta): rows[q] = {"ids", "decide", "opts"}."""
        if not isinstance(questions, dict) or not questions:
            raise LayoutError("questions must be a non-empty object")
        state_text = sanitize(render(state))
        rows, meta = [], []
        for qid, q in questions.items():
            t, opts = options_of(qid, q)
            block = "".join(self.opt + sanitize(o) + self.opt_end + "\n" for o in opts)
            content = self.state + state_text + "\n" + self.q + sanitize(render(q.get("instructions"))) + "\n" + block
            # Qwen's template trims the message (`content|trim`), so the block's last newline goes.
            ids = self.enc(self.pre + content.strip() + self.post) + self.enc("\n") + self.enc(self.think_end + "\n\n" + block + self.decide)
            if len(ids) > self.max_len:
                raise LayoutError("question %r: %d tokens; the model reads at most %d" % (qid, len(ids), self.max_len))
            start = len(ids) - 1 - ids[::-1].index(self.think_end_id)
            opt_pos = [i for i in range(start, len(ids)) if ids[i] == self.opt_end_id]
            rows.append({"ids": ids, "decide": len(ids) - 1, "opts": opt_pos})
            meta.append({"qid": qid, "type": t, "k": len(opts)})
        return rows, meta
