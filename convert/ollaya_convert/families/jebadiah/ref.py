"""Reference implementation of `jebadiah-v1`: Jebadiah's prompt (frontier-infra/jebadiah-*-GGUF,
`scripts/jebadiah_prompt.py` + `scripts/ainode_prompt_verbatim.py`, AINode's own /v1/systemone
renderer at e5c08938, prompt_source_sha256 d2660ebe...).

    user   = "STATE:\\n" + serialize_state(state) + "\\n\\nQUESTION: " + instructions.strip()
             + "\\n\\nOPTIONS:\\n" + "A. opt_0\\nB. opt_1\\n..." + "\\n\\n" + ANSWER_INSTRUCTION
    prompt = "<|im_start|>system\\n" + SYSTEM + "<|im_end|>\\n<|im_start|>user\\n" + user
             + "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"

serialize_state: a string verbatim, anything else `json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`. Options (AINode's `translate_one`): noul `true` then `false` ("true: desc" or
"true"); a choice's criteria keys in order ("name: desc" or "name"); a score's levels (a list of
level names, or an object of level: description). A description is flattened to one line
(`" ".join(desc.split())`). Labels are A..Z, AA, AB, ... (bijective base 26); the option logits are
the labels' logits at the last token, and the author's per-type temperatures calibrate them.

Differences from the author's scripts (docs/families/jebadiah.md):
- The author tokenizes the whole prompt with special parsing; Ollaya parses specials only in the
  template pieces (PRE, POST), as jevk5-v1 does, so control-token text in a request stays text.
- The author's Renderer cuts the state when the prompt exceeds 2,048 tokens. Ollaya never answers
  from a cut state: such a question is rejected.
- A choice may also be a list of labels (read as {label: null}), as for every Ollaya model. A
  question wider than the single-token ainode labels (68 on the Qwen3.5 tokenizer: A..BP) uses the
  author's extended alphabet, up to TypeSafe's 255 options.
"""
from __future__ import annotations

import json

LAYOUT = "jebadiah-v1"
SYSTEM = "You are a decision function. Answer with the single letter of the best option and nothing else."
ANSWER_INSTRUCTION = "Answer with the label of one option and nothing else."
PRE = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_PROMPT_TOKENS = 2048
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
# TypeSafe's option limit.
MAX_OPTIONS_CAP = 255
UPSTREAM = {"prompt": "https://github.com/getainode/ainode", "commit": "e5c089386e0239c9eb270eeb490d181722b8da5b",
            "prompt_source_sha256": "d2660ebec28bd3f1704235bda88d24a397c1c62475e740519cb8ef2d08f25fdd",
            "source": "scripts/jebadiah_prompt.py, scripts/ainode_prompt_verbatim.py"}


class JebError(ValueError):
    """A question the layout rejects (HTTP 400)."""


class TooManyOptions(JebError):
    """More options than single-token labels (HTTP 422)."""


def option_label(index: int) -> str:
    n, out = index + 1, ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def serialize_state(state) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def option_text(name, description) -> str:
    if isinstance(description, str) and description.strip():
        return f"{name}: {' '.join(description.split())}"
    return name


def user_message(state, question: str, options, letters=None) -> str:
    """`letters`: the extended alphabet for a question wider than the ainode labels."""
    letters = letters or [option_label(i) for i in range(len(options))]
    lines = ["STATE:", serialize_state(state), "", f"QUESTION: {question}", "", "OPTIONS:"]
    lines += [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    lines += ["", ANSWER_INSTRUCTION]
    return "\n".join(lines)


def _pairs(qid, crit, kind):
    if not isinstance(crit, dict) or not crit:
        raise JebError(f"question {qid!r}: a {kind} question needs a non-empty 'criteria' object")
    out = []
    for name, desc in crit.items():
        if not isinstance(name, str) or not name.strip():
            raise JebError(f"question {qid!r}: every 'criteria' name must be a non-empty string")
        if desc is not None and not isinstance(desc, str):
            raise JebError(f"question {qid!r}: the 'criteria' description for {name!r} must be a string")
        out.append((name.strip(), desc))
    return out


def compile_question(qid, q, max_options):
    """AINode's translate_one plus Ollaya's list form -> (type, keys in wire order, option texts in
    prompt order, prompt position of each wire option)."""
    if not isinstance(q, dict):
        raise JebError(f"question {qid!r} must be an object")
    kind = q.get("type")
    if kind not in ("choice", "noul", "score"):
        raise JebError(f"question {qid!r}: 'type' must be one of choice, noul, score")
    instructions = q.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise JebError(f"question {qid!r} needs a non-empty 'instructions' string")
    crit = q.get("criteria")
    if kind == "noul":
        described = {}
        if crit is not None:
            if not isinstance(crit, dict):
                raise JebError(f"question {qid!r}: 'criteria' must be an object of {{true, false}}")
            for name, desc in crit.items():
                flat = name.strip()
                if flat not in ("true", "false"):
                    raise JebError(f"question {qid!r}: a noul's 'criteria' names only 'true' and 'false'")
                if desc is not None and not isinstance(desc, str):
                    raise JebError(f"question {qid!r}: the 'criteria' description for {flat!r} must be a string")
                described[flat] = desc
        options = [option_text(n, described.get(n)) for n in ("true", "false")]
        keys, wire = ["false", "true"], [1, 0]
    elif kind == "score":
        if isinstance(crit, list):
            names = []
            for level in crit:
                if not isinstance(level, str) or not level.strip():
                    raise JebError(f"question {qid!r}: every 'criteria' level must be a non-empty string")
                names.append(level.strip())
            options = list(names)
        elif isinstance(crit, dict):
            pairs = _pairs(qid, crit, "score")
            names, options = [n for n, _ in pairs], [option_text(n, d) for n, d in pairs]
        else:
            raise JebError(f"question {qid!r}: a score question needs 'criteria', a list or an object of levels")
        if not MIN_SCORE_LEVELS <= len(names) <= MAX_SCORE_LEVELS:
            raise JebError(f"question {qid!r}: a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
        if len(set(names)) != len(names):
            raise JebError(f"question {qid!r}: 'criteria' repeats a level name")
        keys, wire = [str(i) for i in range(len(names))], list(range(len(names)))
    else:
        if isinstance(crit, list):
            if not all(isinstance(c, str) for c in crit):
                raise JebError(f"question {qid!r}: choice labels must be strings")
            crit = dict.fromkeys(crit)
        pairs = _pairs(qid, crit, "choice")
        if len(pairs) < 2:
            raise JebError(f"question {qid!r}: a choice needs at least 2 options")
        options = [option_text(n, d) for n, d in pairs]
        # the answer carries the caller's key as written (AINode strips it only to validate)
        keys, wire = list(crit), list(range(len(pairs)))
    if len(options) > max_options:
        raise TooManyOptions(f"question {qid!r}: {len(options)} options; this model reads at most {max_options}")
    return kind, keys, options, wire, instructions.strip()


def compile_request(state, questions, n_labels, extended=None):
    """Every question's (qid, type, keys, user message, wire order, letters or None), in request order.
    `n_labels`: how many ainode labels (A, B, ..., AA, ...) are single tokens; `extended`: the author's
    extended alphabet (A..Z, then every single-token two-letter string in order) for wider questions.
    The request is rejected as a whole when any question is."""
    if not isinstance(questions, dict) or not questions:
        raise JebError("'questions' must be a non-empty object of {id: question}")
    extended = extended or []
    out = []
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid.strip():
            raise JebError("every question id must be a non-empty string")
        kind, keys, options, wire, question = compile_question(qid, q, MAX_OPTIONS_CAP)
        letters = None
        if len(options) > n_labels:
            most = max(n_labels, min(len(extended), MAX_OPTIONS_CAP))
            if len(options) > most:
                raise TooManyOptions(f"question {qid!r}: {len(options)} options; this model reads at most {most}")
            letters = extended[:len(options)]
        out.append((qid, kind, keys, user_message(state, question, options, letters), wire, letters))
    return out
