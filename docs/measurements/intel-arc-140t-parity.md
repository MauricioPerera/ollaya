# Intel Arc 140T parity measurements

Measured on 2026-09-27 for [PR #27](https://github.com/ollaya-dev/ollaya/pull/27).
The full Winnow-E4B comparison against the reviewer's CUDA reference **fails** the gate.
Same-backend Vulkan parity passes for both measured models.

## Setup

- Windows x64; Intel Core Ultra 9 285H; 64 GiB system RAM.
- Intel Arc 140T integrated GPU, `Vulkan0`, driver `32.0.101.8860`.
  Vulkan reports about 37 GiB accessible memory. This is shared system memory, not a separate VRAM pool.
- Ollaya revision `5ad1f9f61d4d46b15d9f937be536aab597575fa7`, after merging main `91f873c02fd4b8f95649c49e2aeb43f146a7fe7a`.
  CUDA comparison runs used the same runner executable at checkout revision
  `80ffebb898a1c327d011d2750b8f1f03c59319ea`; the intervening commit changed documentation
  and reference-fixture filenames, not Rust inference code.
- Stock `llama-b11146-bin-win-vulkan-x64.zip`: version `0.5.0-dev`, build 11146,
  commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, Clang 20.1.8.
  The native parity runs used the stock libraries, not the local diagnostic build described below.
- Python reference encoder: laya 0.3.7. The default 40 typed-decisions rows and the full edge-case set were used.

| Model | Pinned author GGUF | Context | Plan | Temperature |
| --- | --- | ---: | --- | ---: |
| Winnow-E4B | `EldanRing/Winnow-E4B`, revision `734302fe5fbfeb3f21a7ece62653c9539be4aaf3`, `gguf/Winnow-E4B-Q8_0.gguf` | 8192, full sliding-window cache | prefix | 1.2574172017327816 |
| JevK5 | `alibiserikbay/JevK5-GGUF`, revision `ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a`, `jevk5-4b-v0.3-Q8_0.gguf` | 16384 | cold | 1.22 |

GGUF SHA-256:

- Winnow-E4B: `840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`.
- JevK5: `aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30`.

## Same-backend native parity

The reference prompt encoder ran through stock llama-server on **Vulkan0**, using the fixed evaluation plan.
`parity_llama` then compared Ollaya on **Vulkan0** with those fixtures, including prompt IDs,
split points, label candidates, state token counts, truncation and rejected requests.
Logit differences below are measured after log-softmax over the options, as in the parity tool.

| Model | Cases | Rejected cases | Matching decisions | Maximum logit difference | Maximum probability difference | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Winnow-E4B | 123 | 15 | 505/505 | 1.143e-5 | 2.299e-6 | PASS |
| JevK5 | 123 | 4 | 593/593 | 9.521e-6 | 1.563e-6 | PASS |

Fixture SHA-256:

- Winnow Vulkan: `de45022c172f5f590e3ca18e461102babeae48e73868d9eb294ec42aa4a57011`.
- JevK5 Vulkan: `2db2417a9c00ef5da3e7cd92838df6924dc285dbee6a86db1576ee64ff498aa0`.

Winnow used 108 prefix steps, 397 barriers and 505 question steps. JevK5 used 579 cold steps;
single-option questions require no inference. These passes establish fidelity to the same backend's
reference. They do not establish matching outputs between Vulkan and CUDA.

## CUDA-reference gate

The maintainer supplied the [exact CUDA reference](https://gist.github.com/cobanov/c808f61976a8210235d8faf8ae91aa15)
in [PR comment 5864110141](https://github.com/ollaya-dev/ollaya/pull/27#issuecomment-5864110141).
Its SHA-256 was verified as `b0b15b5646e8cda17364b6097a9613eb231a88a4813b52c129423313447695fe`.
The supplied decision and calibration JSON files exactly match the registry files used above.
The reference comes from stock b11146 CUDA on an RTX 4090, Linux, 2026-09-26.

| Arc implementation compared with CUDA | Matching decisions | Maximum logit difference | Maximum probability difference | Gate |
| --- | ---: | ---: | ---: | --- |
| Stock Vulkan llama-server, full 123 cases | 502/505 | 0.32614444 | 0.05600171 | FAIL |
| Ollaya Vulkan runner, full 123 cases | 502/505 | 0.3261 | 0.05600 | FAIL |

Both full comparisons cover the same 505 questions and 15 rejected cases. The stock comparison
found no differences in prompt IDs, split points, candidates, wire order, state token counts or
truncation. 502 questions exceed the 1e-3 logit tolerance. The runner reported the same decision
changes and numerical range, separating the backend difference from an Ollaya wrapper error.

The changed decisions were `preset/router/mask_text` / `domain`, `edge/many_options_40` / `intent`,
and `td/customer_service_000080` / `churn_risk`.

A targeted regression set contains the complete requests for those three cases, plus
`preset/email/mask_text`, which has the largest logit difference: four cases, 15 questions.
Its SHA-256 is `1253f4099d15b9e76ce6c48df85af8abb5d0901e6359e71b047c754f2a58f711`.

| Targeted setting | Matching decisions | Maximum logit difference | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Disable F16 | 12/15 | 0.3321 | 0.01769 |
| Force MMVQ | 12/15 | 0.3279 | 0.01805 |
| Conservative settings | 12/15 | 0.3299 | 0.01782 |
| Disable cooperative matrices and cooperative matrices 2 | 13/15 | 0.1517 | 0.02258 |
| Disable both cooperative matrix modes and force MMVQ | 12/15 | 0.1414 | 0.02262 |
| Stock Vulkan server, flash attention disabled | 12/15 | 0.32484553 | 0.01751260 |
| Ollaya CPU | 15/15 | 0.1915 | 0.03166 |

Every targeted setting fails the unchanged logit tolerance. They were not expanded into further
full runs, since the selected counterexamples already disprove a complete pass.

## CPU-reference diagnostic screen

The completed stock CPU versus stock Vulkan comparison for JevK5 covers all 123 cases,
593 questions and four rejected requests: 586/593 matching decisions, maximum normalized
logit difference 0.33286801 and maximum probability difference 0.08266701.
574 questions exceed the 1e-3 logit tolerance; prompt and state metadata match in all cases.
The full CPU fixture SHA-256 is `d82aca8fb3d23cafcf554eb8aa55b49a843401e0d4b123afc5040ab06b6d5216`.

A stock CPU reference was generated independently for JevK5. A fixed screen selected ten spread
cases, 50 questions, from an initial 18-case snapshot of that reference. The identical screen was
used for all five variants below. This is a diagnostic sample, not the complete CUDA gate.

| Vulkan setting | Matching decisions | Maximum logit difference | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Default | 48/50 | 0.1899 | 0.05146 |
| `GGML_VK_DISABLE_F16=1` | 48/50 | 0.1895 | 0.05059 |
| Conservative settings below | 48/50 | 0.1892 | 0.05132 |
| `GGML_VK_FORCE_MMVQ=1` | 48/50 | 0.1807 | 0.05070 |
| `GGML_VK_DISABLE_MMVQ=1` | 48/50 | 0.1899 | 0.05146 |

The conservative run disabled F16, fusion, cooperative matrices, cooperative matrices 2,
DOT2 and integer dot products, using the corresponding `GGML_VK_DISABLE_*` variables.
Every variant failed the unchanged 1e-3 logit tolerance and exact-decision requirement.
The 18-case CPU snapshot SHA-256 was
`10552c9365c0cbd06ff58687a913ecbc3555b472af6103693914ecac6e0f8a6b`;
the ten-case input SHA-256 was `035b4c149439cf9eb477913b8fdf7cc3d2e04b1ca215fd859117241a9d2a9410`
in both screen batches. CPU is not a substitute for the requested CUDA fixture.
Runs overlapped other work, so elapsed times are not presented
as performance benchmarks.

## Operation-level investigation

A local MSVC build of the pinned source enabled `GGML_VULKAN_CHECK_RESULTS` and disabled
`GGML_BACKEND_DL`. The check mode did not compile unmodified: its globals and two check
functions were private to `ggml-vulkan-debug.cpp`, while `ggml-vulkan.cpp` referenced them.
A local declaration/linkage patch enabled the diagnostic build. It changes no inference arithmetic
and is not included in the shipped libraries or this Ollaya PR.

The existing `test-backend-ops` Q8_0/F32 MUL_MAT cases with `m=16`, `n=1|8` and `k=256|4096`
passed 12 supported tests with default Vulkan settings and with forced MMVQ. Unsupported
permuted/broadcast cases were skipped. The operation test has its own NMSE tolerance of 5e-4;
it does not test the final model's option-logit gate.

The internal CPU check reported `avg_err` around 0.0035-0.0078 for default matrix-vector cases,
and around 1e-7 for their forced-MMVQ counterparts. Its denominator is `max(abs(reference), 1)`.
Each run initializes its own random tensors, so these are operation diagnostics rather than
paired model-input measurements. A large-batch case retained about 0.00655 error with forced MMVQ.
The pinned source selects a dequantized route for Intel Windows matrix-vector operations by default;
forcing MMVQ changes that route. The full-model sample above still fails after this change.

Attempts to run the check-enabled server on a Winnow input did not complete: the HTTP connection
was reset, including a retry with fusion, async execution and graph optimization disabled and
serialized submissions enabled. Those attempts provide no complete model parity result.
The exact cause of the remaining model drift has not been established.

There are concrete precision differences in the pinned source:

- [Vulkan matrix selection](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6097)
  forces F32 activations into an F16 input path for cooperative matrices with quantized weights.
  Disabling those matrices changes the path and reduces, but does not eliminate, the measured drift.
- [Vulkan activation quantization](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp#L114)
  stores its scale and sum as `f16vec2`; disabling F16 computation does not change this storage format.
- [CUDA's MMQ layout](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cuda/mmq.cuh#L59)
  selects four FP32 activation scales for Q8_0, and
  [the quantizer](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cuda/quantize.cu#L550)
  writes those scales without the F16 conversion.

These differences identify possible sources of drift. They do not prove that changing scale
storage alone will make the entire model pass. No numerical backend patch has been validated.

## Reproduction

From `convert/`, use the pinned author GGUF and the stock server from build b11146:

```powershell
uv run python -m ollaya_convert.families.llm_common.export_llama winnow `
  --server $SERVER --gguf $WINNOW --slug arc-vulkan-native `
  --repo EldanRing/Winnow-E4B --revision 734302fe5fbfeb3f21a7ece62653c9539be4aaf3 `
  --file gguf/Winnow-E4B-Q8_0.gguf --temperature 1.2574172017327816 `
  --upstream-commit 77d14580c6732ca2f3745750c1dc1fd446d8bcee `
  --n-ctx 8192 --device Vulkan0 --port 8098

uv run python -m ollaya_convert.families.llm_common.export_llama jevk5 `
  --server $SERVER --gguf $JEVK5 --slug arc-vulkan-native `
  --repo alibiserikbay/JevK5-GGUF --revision ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a `
  --file jevk5-4b-v0.3-Q8_0.gguf --temperature 1.22 `
  --n-ctx 16384 --device Vulkan0 --port 8097
```

Place the matching GGUF as `model.gguf` beside each generated `decision.json` and `calibration.json`.
From the repository root, set `OLLAYA_LIBRARY_PATH` to the stock install's `lib/ollaya` directory:

```powershell
cargo run --release -p ollaya-runner --example parity_llama -- `
  convert/out/winnow-arc-vulkan-native convert/out/winnow-arc-vulkan-native/goldens-vulkan.jsonl Vulkan0
cargo run --release -p ollaya-runner --example parity_llama -- `
  convert/out/jevk5-arc-vulkan-native convert/out/jevk5-arc-vulkan-native/goldens-vulkan.jsonl Vulkan0
```

To reproduce the CPU screen, generate JevK5 goldens with the same arguments and
`--device cpu --slug arc-cpu`. Select these records from the resulting `goldens-cpu.jsonl`
and run `parity_llama` with that ten-case fixture on `Vulkan0`, changing only the settings above:

```text
preset/triage/tr_billing
preset/triage/tr_outage
preset/triage/en_injection
preset/triage/ar_complaint
preset/triage/de_cancel
preset/triage/conversation
preset/triage/mask_text
preset/email/tr_billing
preset/email/en_pricing
preset/email/hi_refund
```

The operation diagnostic command was:

```powershell
test-backend-ops.exe test -b Vulkan0 -o MUL_MAT `
  -p 'type_a=q8_0,type_b=f32,m=16,n=(1|8),k=(256|4096),'
```

The export/replay tools now name Vulkan fixtures `goldens-vulkan.jsonl` and record `Vulkan0`
in their metadata. Before this correction they used the filename `goldens-cpu.jsonl` even though
the server and metadata selected Vulkan. The measured Vulkan fixtures above were renamed after
export without modifying their contents.

## Gate status

The requested CUDA-to-Arc comparison is complete for the stock Vulkan build and **fails**.
No tested setting meets the gate and no numerical backend correction has been validated.
Further work on activation representation or kernel precision would require backend changes
and a new validation cycle; Ollaya currently distributes the pinned upstream binaries.
Same-backend parity passes and latency measurements must not be used to claim that Vulkan
is ready for the repository's cross-backend requirements.
