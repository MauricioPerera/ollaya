Decima-small by [A. M. Madani](https://huggingface.co/amyrmahdy/decima-small) is a small multilingual decision model: multilingual-e5-small (122M parameters) fine-tuned with a late-interaction scorer. It encodes the state and each option on its own, and every option reads the state to get its score, so the order of the options never changes the answer. Score questions go through an ordinal head. It never generates text.

> Needs an Ollaya release that runs the `decima-late-interaction-v1` layout.

## Models

| Tag | Weights | Typed-decisions accuracy | Five questions, RTX 4090 / CPU |
|---|---|---|---|
| `decima:latest`, `decima:small` | Decima-small 1.1, fp32, 490 MB | 0.432 | 7.3 ms / 146 ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya, with a calibration error (ECE) of 0.110 at the author's temperature. The author reports 0.427 for 1.1 on the same file, and notes that every small model they tested is below this benchmark's majority-class baseline (0.461): long, multi-fact business cases are hard at this size. Latency is the median request through the HTTP API on an RTX 4090 and on the CPU of the same machine (i9-13900K).

## Usage

```shell
ollaya run decima --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `decima`.

## How it works

- **Rows.** Each question becomes one state row (`query: ` and the question with the state) and one row per option (`passage: ` and the question with the option), exactly as the author's `decima/systemone.py` and `model.py` build them.
- **Scores.** The options of every question are scored in one pass. Choice and yes/no answers are a softmax of the scores at the author's fitted temperature (0.94); score questions use the author's cumulative-link ordinal head.
- **Engine.** ONNX Runtime, fp32, on an NVIDIA GPU (CUDA) or the CPU. The graph reads the author's own checkpoint files (490 MB), unmodified.
- **Parity.** Ollaya's runtime matches the author's own code (fp32) on 581 questions, on the CPU and on CUDA (RTX 4090 and RTX 5090): identical token rows, the same decision on every question, probabilities within 2.3e-6.

## Limits

- **State.** 512 tokens per question, the question included. `/v1/systemone` answers a longer state with `STATE_TRUNCATED`, as the author's server does; `/api/decide` answers from its first 512 tokens.
- **Options.** Each option row holds the question and the option in 64 tokens, so a long question cuts the option's own text short, as upstream does. 2 to 255 choices, 2 to 10 score levels.
- **Languages.** Multilingual: the author evaluated 20 languages; the weakest are Swahili and Hindi.
