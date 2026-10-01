"""Check the `cygnet-v1` port (ref.py) against the author's shim, text for text.

    uv run python -m ollaya_convert.families.cygnet.check <cygnet-recipe checkout>

For every case of the shared set and every question on its own: the author's
`decision_server.parse_question` then `cygnet_shim.build_prompt` must give the same user message as
`ref.compile_request`, and a question the author rejects must be rejected by the port. Wider questions
than one pass (grouped upstream, TOO_MANY_OPTIONS here) and the port's list-form choices are counted
separately. The system message must equal `cygnet_shim.SYSTEM`.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

from ..llm_common import cases
from . import ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", help="a checkout of github.com/blockbrain-ai/cygnet-recipe at the pinned commit")
    ap.add_argument("--td", type=int, default=200)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("decision_server", os.path.join(a.recipe, "shim", "decision_server.py"))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    shim = ds.shim
    assert shim.SYSTEM == ref.SYSTEM, "SYSTEM differs"
    same = rejected = grouped = extension = 0
    bad = []
    for cid, state, qs in cases.edge_cases() + cases.typed_decisions(a.td):
        for qid, q in qs.items():
            try:
                qtype, instructions, items, _ = ds.parse_question(qid, q)
                up = shim.build_prompt(state or "", instructions, [(shim.LETTERS[i], lab, t) for i, (lab, t) in
                                                                   enumerate(items)]) if len(items) <= ref.GROUP_SIZE else None
                up_err = None
            except shim.Unprocessable as e:
                up, up_err, items = None, e, []
            try:
                _, _, user, _ = ref.compile_request(state, {qid: q})[0]
                our_err = None
            except ref.TooManyOptions as e:
                user, our_err = None, e
            except ref.CygnetError as e:
                user, our_err = None, e
            if up_err is not None and our_err is not None:
                rejected += 1
            elif up_err is not None:
                crit = q.get("criteria") if isinstance(q, dict) else None
                if isinstance(q, dict) and q.get("type") == "choice" and isinstance(crit, list):
                    extension += 1
                else:
                    bad.append((cid, qid, "author rejects: %s" % up_err))
            elif isinstance(our_err, ref.TooManyOptions) and up is None:
                grouped += 1
            elif our_err is not None:
                bad.append((cid, qid, "port rejects: %s" % our_err))
            elif up != user:
                bad.append((cid, qid, "user message differs"))
            else:
                same += 1
    print("identical user messages %d | both reject %d | grouped upstream (TOO_MANY_OPTIONS here) %d | "
          "list-form choices (Ollaya extension) %d | mismatches %d" % (same, rejected, grouped, extension, len(bad)))
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
