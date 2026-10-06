# decima (`decima-late-interaction-v1`)

**Decima-small** by A. M. Madani ([amyrmahdy/decima-small](https://huggingface.co/amyrmahdy/decima-small),
[amyrmahdy/decima](https://github.com/amyrmahdy/decima), Apache-2.0) is `intfloat/multilingual-e5-small` (MIT)
fine-tuned with a small **late-interaction scorer**: the state and every option are encoded on their own, and
each option reads the state's tokens through cross-attention to get one score. Options never see each other, so
their order cannot change the answer. Score questions go through a cumulative-link ordinal head. 122M
parameters. Requested in #54.

| Tag | Weights | Temperature |
|---|---|---|
| `decima:small`, `decima:latest` | `amyrmahdy/decima-small@2e7f4d07` (Hub tag `v1.1.1`): `pytorch/encoder/model.safetensors` (471 MB) and `pytorch/head.safetensors` (19 MB), fp32 | 0.9356 (`pytorch/decima.json`, fitted by the author) |

The code is pinned to the GitHub tag `v1.1.1` (`2df60942`). The author also publishes an int8 ONNX export;
Ollaya's graph reads the fp32 checkpoint instead, so that the weights come unmodified from the files the
author's own code loads.

## Rows

`crates/ollaya-decision/src/decima.rs` ports `decima/systemone.py` (`to_question`) and the text and
tokenization steps of `decima/model.py`; `convert/ollaya_convert/families/decima/layout.py` is its Python twin.
Every question becomes one state row and one row per option, each encoded on its own:

```text
state row   [cls] tok("query: "   + normalize(text + "\n" + state))[..510]  [sep]
option row  [cls] tok("passage: " + normalize(text + " " + option))[..62]   [sep]
```

- **Text.** The question's instructions: a string as is, anything else as `json.dumps(ensure_ascii=False)`,
  the question id when they are absent or null. A noul question whose criteria give a `true` or `false`
  description appends `\nTrue if: ...\nFalse if: ...`, with U+2014 for the side without one.
- **Options.** A choice's `label: description` (the label alone when the description is empty), each score
  level's text, and `yes` and `no` for noul.
- **State.** A string as is; an object or array as `json.dumps(ensure_ascii=False)`. Numbers, booleans and
  null are rejected, as upstream's server rejects them.
- **normalize.** Unicode NFC, then Python's `str.strip()`.
- **Length.** A state row keeps its first 512 tokens and is flagged: `/v1/systemone` answers 422
  `STATE_TRUNCATED`, as upstream's server does, and `/api/decide` answers from the cut state (as upstream's
  `model.py` does) with `state_truncated: true`. An option row is cut to 64 tokens silently, as upstream cuts
  it.

`python -m ollaya_convert.families.decima.check` compares the port with upstream on the shared request set (the
edge cases, the 400 typed-decisions rows and 9 inputs Decima's mapping treats specially): 469 requests with
identical rows (12 of them with a cut state), 18 rejected by both, 3 refused first by Ollaya's shared question
rules, 0 differences.

## The graph

`python -m ollaya_convert.families.decima.export` writes one weightless graph that answers every question of a
request in one run. Its 259 weight tensors reference the two checkpoint files by byte offset; only the
encoder's pooler is left out.

```text
inputs   state_ids, state_mask     int64 [questions, state_len]   one state row per question, right-padded
         option_ids, option_mask   int64 [options, option_len]    every option row, questions in order
         option_state              int64 [options]                the question each option belongs to
outputs  scores                    float32 [options]              the raw score, before the temperature
         ordinal_g, ordinal_gap    float32 [options]              the ordinal head's two projections
```

The runner groups consecutive questions into one run as long as the padded tokens stay within the shared token
budget. Against upstream's PyTorch modules, on 295 questions: scores within 9.7e-6, projections within 4.2e-6.

## Answers

- **Choice and noul.** A softmax of the scores over the temperature. Upstream's noul order is `[yes, no]`; the
  runner turns it into the `[false, true]` order answers use.
- **Score.** The ordinal head, on the scores over the same temperature, so the score slot of
  `calibration.json` is 1:

  ```text
  s = scores / T                    expected = sum_k softmax(s)_k * k - (K - 1) / 2
  g = mean_k ord_g(z_k) + expected  (ord_g is linear, so the graph returns it per option)
  gaps = softplus(ord_gap(z)) + 1e-3,  theta_j = sum_{i<=j} gaps_i - sum(gaps) / 2   (j < K - 1)
  p_k = sigmoid(g - theta_{k-1}) - sigmoid(g - theta_k)   (1 for k = 0, 0 for k = K - 1)
  log p = log_softmax(log(max(p, 1e-7)))
  ```

## Differences from upstream

- **Option cache.** Upstream's `Decima` caches the encodings of up to 256 option sets; Ollaya encodes every
  row on every request, in the same run as the state.
- **int8.** Not used (above).

## Parity (measured 2026-10-06)

Goldens: `python -m ollaya_convert.families.decima.goldens` runs the author's `model.py` and `systemone.py` in
fp32 on the CPU: 140 requests, 18 of them rejected upstream, 3 with a cut state. `parity_decima` checks the
rejections and every row id for id, then the graph's outputs, the decisions and the `/v1/systemone` answers
against upstream's `system_one`.

| Device | ONNX Runtime | Questions | Rows | Decisions | Scores max | Probabilities max |
|---|---|---|---|---|---|---|
| x86-64 CPU | pyke 1.28.0 (static) | 581 | 2,922 identical | 581/581 | 1.1e-5 | 2.3e-6 |
| CUDA, RTX 4090 | Microsoft 1.28.2, CUDA 13 pack | 581 | 2,922 identical | 581/581 | 1.1e-5 | 1.3e-6 |
| CUDA, RTX 5090 | Microsoft 1.28.2, CUDA 13 pack | 581 | 2,922 identical | 581/581 | 1.2e-5 | 1.5e-6 |

On every device the `/v1/systemone` answers name the same choice as upstream's on every question, and their
numbers agree to the fourth decimal (1.0e-4, the rounding).

## Quality

- **Typed-decisions** (all 400 test states, 2,000 questions, argmax against the majority label, through
  Ollaya's runtime: `logits` example and `quality_runtime.py`, CPU): **0.432** (863 of 2,000), ECE 0.110
  at the author's temperature. By type: choice 0.412, score 0.356, noul 0.552.
- **The author's figure** for 1.1: 0.427 (95% interval 0.399 to 0.455). The author notes that every small model
  they tested, Decima included, is below this benchmark's majority-class baseline of 0.461: long, multi-fact
  business cases are hard at this size.

## Speed (measured 2026-10-06)

The triage preset (five questions) through the HTTP API, one request at a time
(`bench_latency.py`, the protocol of the 2026-10-01 sweep), on the RTX 4090 machine:

| Device | p50 | p90 | Load |
|---|---|---|---|
| CUDA, RTX 4090 | 7.3 ms | 8.5 ms | 0.84 s |
| CPU, i9-13900K | 146 ms | 148 ms | 0.58 s |

On this CPU no other model Ollaya ships is as fast: the next are `laya:multilingual` at 308 ms and `gliclass`
at 563 ms (2026-10-01). The author measured 20 ms for one question with four options on one laptop core with their int8
export and cached option encodings; Ollaya runs fp32 and encodes the options on every request.

## Limits

- **State.** 512 tokens per question, the question's text included. Longer: `/v1/systemone` answers
  `STATE_TRUNCATED`, `/api/decide` answers from the first 512 tokens.
- **Options.** Each option row holds the question's text and the option in 64 tokens, so long instructions cut
  the option's own text short, as upstream does. 2 to 255 choices, 2 to 10 score levels, 1 to 256 questions.
- **Languages.** Multilingual: the author evaluated 20 languages; the weakest are Swahili and Hindi.
