"""Reference implementation of `cygnet-v1`: Cygnet (github.com/blockbrain-ai/cygnet-recipe), frozen
google/gemma-4-12B-it with a one-token option-letter readout. A port of `shim/cygnet_shim.py`
(`SYSTEM`, `build_prompt`) and `shim/decision_server.py` (`parse_question`), pinned below.

    user = state_text.rstrip() + "\\n\\n" + instructions.rstrip() + "\\n\\nOptions:\\n" + "A. text_0\\n..."
           + "\\n\\nAnswer with the letter of exactly one option, and nothing else:"
    prompt = Gemma's chat template over [system: SYSTEM, user: user], thinking off

state_text: the state when it is a string, "" when it is falsy (`body.get("state") or ""`), otherwise
`json.dumps(state, ensure_ascii=False, indent=1)`. instructions: `q.get("instructions") or ""`, JSON
(`ensure_ascii=False`) when not a string. Option texts (parse_question): a choice's description, its
label when blank, `"label: <json>"` for JSON; a score level, `"Level i"` when blank; a noul reads false
(A) then true (B), "No" and "Yes" when blank. The option logits are the letters' next-token logits
(upstream reads vLLM's logprobs with the output constrained to the letters, which is the same
softmax over them); p^(1/T) with T = 3.4 is softmax(logits / 3.4).

Differences (docs/families/cygnet.md): one pass reads up to 20 options (upstream groups wider choices
and composes the group passes; not ported); specials are parsed only in the template pieces; a
choice may be a list of labels (Ollaya's API form).
"""
from __future__ import annotations

import json

LAYOUT = "cygnet-v1"
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else \u2014 a single character, no words, no punctuation, no explanation."
)
ANSWER_LINE = "Answer with the letter of exactly one option, and nothing else:"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GROUP_SIZE = 20
MAX_SCORE_LEVELS = 10
TEMPERATURE = 3.4
UPSTREAM = {"recipe": "https://github.com/blockbrain-ai/cygnet-recipe", "commit": "3cf591c692dec649f7c134449814610307c7bb3a",
            "source": "shim/cygnet_shim.py, shim/decision_server.py", "base": "google/gemma-4-12B-it",
            "base_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"}


class CygnetError(ValueError):
    """A question the layout rejects (HTTP 400)."""


class TooManyOptions(CygnetError):
    """More options than one pass reads (HTTP 422)."""


def _text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _blank(v):
    return v is None or (isinstance(v, str) and not v.strip())


def state_text(state) -> str:
    state = state or ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def user_message(state, instructions, texts) -> str:
    instructions = instructions or ""
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state_text(state).rstrip(), "", instructions.rstrip(), "", "Options:"]
    lines += [f"{LETTERS[i]}. {t}" for i, t in enumerate(texts)]
    lines += ["", ANSWER_LINE]
    return "\n".join(lines)


def option_texts(qid, q):
    """parse_question's option texts in prompt order (noul: false, true) and the question type."""
    if not isinstance(q, dict):
        raise CygnetError(f"question {qid!r} must be an object")
    kind, crit = q.get("type"), q.get("criteria")
    ok = lambda d: isinstance(d, (str, dict, list)) or d is None  # noqa: E731
    if kind == "choice":
        if isinstance(crit, list) and crit:
            if not all(isinstance(c, str) for c in crit):
                raise CygnetError(f"question {qid!r}: choice labels must be strings")
            crit = dict.fromkeys(crit)
        if not isinstance(crit, dict) or not crit:
            raise CygnetError(f"question {qid!r}: criteria must be a non-empty object of options")
        texts = []
        for label, d in crit.items():
            if not ok(d):
                raise CygnetError(f"question {qid!r}: every option description must be text, JSON or null")
            texts.append(label if _blank(d) else d if isinstance(d, str) else f"{label}: {_text(d)}")
    elif kind == "score":
        if not isinstance(crit, list) or not crit:
            raise CygnetError(f"question {qid!r}: criteria must be a non-empty list of levels")
        if len(crit) > MAX_SCORE_LEVELS:
            raise CygnetError(f"question {qid!r}: a score takes at most {MAX_SCORE_LEVELS} levels")
        texts = []
        for i, d in enumerate(crit):
            if not ok(d):
                raise CygnetError(f"question {qid!r}: every level must be text, JSON or null")
            texts.append(f"Level {i}" if _blank(d) else _text(d))
    elif kind == "noul":
        crit = {} if crit is None else crit
        if not isinstance(crit, dict):
            raise CygnetError(f"question {qid!r}: criteria must be an object with 'true' and 'false'")
        sides = {}
        for key, d in crit.items():
            side = str(key).lower()
            if side not in ("true", "false") or side in sides:
                raise CygnetError(f"question {qid!r}: criteria takes only 'true' and 'false', once each")
            if not ok(d):
                raise CygnetError(f"question {qid!r}: every description must be text, JSON or null")
            sides[side] = d
        texts = ["No" if _blank(sides.get("false")) else _text(sides["false"]),
                 "Yes" if _blank(sides.get("true")) else _text(sides["true"])]
    else:
        raise CygnetError(f"question {qid!r}: type must be 'choice', 'score' or 'noul'")
    if len(texts) > GROUP_SIZE:
        raise TooManyOptions(f"question {qid!r}: {len(texts)} options; one pass reads at most {GROUP_SIZE}")
    return kind, texts


def compile_request(state, questions):
    """Every question's (qid, type, user message, n options), in request order. The request is rejected
    as a whole when any question is."""
    if not isinstance(questions, dict) or not questions:
        raise CygnetError("questions must be a non-empty object of named questions")
    out = []
    for qid, q in questions.items():
        kind, texts = option_texts(qid, q)
        out.append((qid, kind, user_message(state, q.get("instructions"), texts), len(texts)))
    return out
