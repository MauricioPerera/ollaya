# clm (`clm-v1`): a contrastive decision model

**Contrastive-LM/CLM-v0.1-8B** (Apache-2.0) scores options by similarity instead of reading answer
logits. It is two small projection heads (a state head and an action head, 1536 wide, 3 layers, 512
out) on a frozen **Qwen/Qwen3-8B** encoder (Apache-2.0), trained with a bidirectional InfoNCE loss.
Code: [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM). Requested in #13.

## How it decides

For each question (upstream `clm.schema.build_pairs`):

- **State text:** the state rendered as prose (`to_text`: strings verbatim, objects as `key: value`
  fields, arrays as `- item` lines), a blank line, then the question's instructions.
- **Option texts:** a choice option's description (its label when the description is null or `""`),
  a score level's text, or for noul `false: No. This is false: <instructions>` and
  `true: Yes. This is true: <instructions>` (or the given descriptions).
- **Embedding:** Qwen3-8B's final hidden state (after the last RMSNorm) at the text's last token,
  L2-normalised. Upstream serves it with vLLM's pooling runner (LAST pooling); the tokenizer adds no
  special tokens.
- **Score:** `z = normalize(head(embedding))`, the state head for the state text and the action head
  for each option; option logit = `min(exp(logit_scale), 100) * z_option . z_state` (the scale is
  100.0 for this checkpoint); probabilities are the softmax over a question's options. No other
  calibration.

## The graph

One graph (7 MB), weightless: `input_ids [rows, seq]` (one text per row, right-padded), `last_pos
[rows]`, `action [rows]` (1 for an option text) → `z [rows, 512]`. Its weights reference the five BF16
shards of `Qwen/Qwen3-8B@b968826d…` (398 widening Casts, `weights_in_memory: bf16`, so 16 GB of
memory, not 32) and the F32 head tensors inside `CLM_v0.1-8B.pt@e939398d…`, a torch zip whose tensors
are stored uncompressed, by byte offset. Only `lm_head.weight` of the base goes unused.

The runtime computes the dot products itself and caches every projection by text and head (8,192 of
them, about 16 MB): options that repeat across requests, or a request sent again, skip the encoder.

## Differences from upstream

- **Long texts are rejected, not cut.** Upstream truncates a text to its last 2,048 tokens; Ollaya
  answers 422, as TypeSafe routes never answer from a truncated state (#16).
- **`instructions` is required,** as for every Ollaya model; upstream accepts a question without it.
- **Precision.** Upstream runs the encoder in BF16 through vLLM; Ollaya computes in fp32 on the same
  BF16 weights, as for decider and kev. The goldens are that computation (`families/clm/ref.py`).
  The model card's example (`billing` 0.93878) gives 0.988 both with this reference and with a plain
  transformers BF16 forward; the difference to the card is unexplained (an older head is the likely
  cause), and upstream publishes no fixtures to settle it.

## Parity (measured 2026-09-28)

Goldens: `ollaya_convert.families.clm.goldens` over the shared case set (typed-decisions rows, Laya's
edge cases and the decoder edge cases): 117 requests, 16 rejected upstream (choice criteria given as a
list, like kev), 480 questions.

| | texts and token ids | decisions | option logits max | probabilities max (p99) | wire answers |
|---|---|---|---|---|---|
| CUDA, RTX 4090 | 480 / 480 identical, 0 rejection mismatches | 100 % | 1.3e-4 | 2.9e-5 (1.8e-5) | max diff 7.6e-5 |
| CPU (24 cores) | 480 / 480 identical, 0 rejection mismatches | 100 % | 1.2e-4 | 2.1e-5 (1.3e-5) | max diff 6.5e-5 |

```sh
CLM_SRC=… uv run --with requests python -m ollaya_convert.families.clm.export --out out/clm-8b --base BASE --head HEAD.pt
CLM_SRC=… uv run --with requests python -m ollaya_convert.families.clm.goldens out/clm-8b --base BASE --head HEAD.pt
cargo run --release -p ollaya-runner --example parity_clm -- convert/out/clm-8b convert/out/goldens-clm-8b.jsonl cuda --latency
```

## Quality and speed

- **Typed-decisions** (all 400 states, 2,000 questions, argmax against the majority label, the shared
  `llm_common/eval_refs.py`): **0.357** (choice 0.248, score 0.331, noul 0.500), ECE 0.485. That is
  close to chance on choice and noul: zero-shot on these structured workflow states, CLM-v0.1-8B leans
  towards `true` (81 % of noul answers, against 51 % in the labels). Its authors report its strength on
  computer-use, gaming and tool-calling states, and as a fine-tuned verifier; typed-decisions is neither.
  Pick it for those, not as a general triage model.
- **RTX 4090, runner, parity fixtures:** five questions take 494 ms at the median with nothing cached
  (every state and option text through the encoder), and well under a millisecond when the same request
  comes again.
- **RTX 4090, HTTP API:** the triage preset (five questions) on a new short message takes 149 ms at
  the median once its questions are cached (only the five state texts run), and 0.8 ms for a repeated
  request.
- **CPU:** about 13 s per request with nothing cached (24-core x86): use a GPU with 20 GB or more.
