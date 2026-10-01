# cygnet (`cygnet-v1`)

**Cygnet** by blockbrain-ai ([cygnet-recipe](https://github.com/blockbrain-ai/cygnet-recipe), MIT) is not a
fine-tune: it is frozen `google/gemma-4-12B-it` (Apache-2.0), a system message, the options as letters, one
answer slot, and one calibration temperature (3.4). Upstream serves it with unmodified vLLM and a small
shim that reads the model's probability of each option letter with the output constrained to the letters.
Requested in #39.

| Tag | Weights | Temperature |
|---|---|---|
| `cygnet:12b`, `cygnet:latest` | `ggml-org/gemma-4-12B-it-GGUF@e3e68173` `gemma-4-12B-it-Q8_0.gguf`, converted from `google/gemma-4-12B-it@707f0a3b` (the revision Cygnet pins) | 3.4 |

Cygnet's published figures (203 of 231 on JevBench's public items) come from BF16 weights on vLLM. Ollaya
runs Gemma through llama.cpp, so it ships ggml-org's Q8_0 conversion of the same revision (12.7 GB; the BF16
file, 23.8 GB, does not fit a 24 GB GPU with its context). A quantized file is a slightly different model
numerically; the quality numbers below are measured on it.

## Prompt

`convert/ollaya_convert/families/cygnet/ref.py` ports `shim/cygnet_shim.py` (`SYSTEM`, `build_prompt`) and
`shim/decision_server.py` (`parse_question`) at `3cf591c6`:

```text
user = state_text.rstrip() + "\n\n" + instructions.rstrip() + "\n\nOptions:\n" + "A. text_0\nB. text_1\n..."
       + "\n\nAnswer with the letter of exactly one option, and nothing else:"
```

- **State.** A string as it is; nothing when the state is empty or falsy; anything else
  `json.dumps(indent=1, ensure_ascii=False)`, the rendering every Cygnet figure was measured with.
- **Options.** A choice's description, its label when blank, `label: <json>` for a JSON description; a score
  level, `Level i` when blank; a noul reads `false` (A) then `true` (B), `No` and `Yes` when undescribed.
- **Readout.** The next-token logits of the question's letters. Upstream reads vLLM's log-probabilities with
  the output constrained to those letters, which is the same softmax over them; `p^(1/3.4)` renormalised is
  `softmax(logits / 3.4)`.
- **Template.** Gemma's chat template over the system and user messages, thinking off, rendered by the
  pinned llama-server from the GGUF's own template (`decision.json`).

`python -m ollaya_convert.families.cygnet.check <cygnet-recipe checkout>` builds every question of the shared
case set both ways: 1,364 user messages identical to the shim's, the same 16 rejections. The system message
is the shim's byte for byte.

## Differences from upstream

- **Wide questions.** One pass reads up to 20 options, the upstream group size. Upstream reads a wider choice
  in groups of 20 and a final pass over the group winners; that is not ported, so more than 20 options is
  `TOO_MANY_OPTIONS`.
- **Weights.** Q8_0 on llama.cpp, not BF16 on vLLM (above).
- **Control tokens.** Specials are parsed only in the template pieces; user text stays text.
- **List-form choices.** A choice given as a list of labels reads as `{label: null}`, as for every Ollaya model.

## Parity (measured 2026-09-30)

Goldens: `export_llama.py cygnet` sends every test request through `ref.py` to a stock `llama-server` of the
pinned build (b11146) with the GGUF, one cold pass per question. 123 cases, 17 rejected by both, 502
questions.

| Model | Device | Decisions | Option logits max | Probabilities max | Five questions (runner, p50) |
|---|---|---|---|---|---|
| 12b Q8_0 | CUDA, RTX 4090 | 502/502 | 7.7e-6 | 4.1e-7 | 198 ms |

## Quality

- **Typed-decisions** (all 400 test states, 2,000 questions, argmax against the majority label, measured here
  through the goldens' path, `quality_llama.py`): **0.683**, ECE 0.148 at the shipped temperature 3.4 (0.289 at
  T = 1). Cygnet is frozen Gemma: nothing about it was trained on typed-decisions.
- **JevBench public items** (the authors' run, BF16 on vLLM, not this Q8_0 file): 203 of 231.

## Limits

- **Options.** Up to 20 per question; a score takes 1 to 10 levels.
- **Context.** 16,384 tokens per question; a longer prompt is rejected, not cut.
- **Memory.** About 13 GB for the weights plus the context: a 16 GB GPU holds it.
