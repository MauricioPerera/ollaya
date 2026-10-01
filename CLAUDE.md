# Ollaya

Runs open decision models locally, the way Ollama runs LLMs (see `README.md`).

- Layout: `crates/` is the Rust workspace (CLI, axum daemon, ONNX Runtime runner processes, registry client); `convert/` is Python build-time tooling (ONNX export, parity, goldens); `site/` is the static website (Hono JSX and Tailwind, Cloudflare static assets).
- Tests: `cargo test --workspace`. Runtime parity: `cargo run --release -p ollaya-runner --example parity -- <model-dir> <goldens.jsonl>`. Export parity and goldens: "Development" in `README.md`. Site: `cd site && npm run build`.
- Before committing Rust changes: `cargo fmt --all`, then `cargo clippy --workspace --all-targets --locked -- -D warnings`. CI runs `cargo fmt --all --check` first and stops there, so unformatted code fails main before clippy and the tests run.
- Invariant: parity tests must pass. Never loosen a tolerance to make a change pass.
- Invariant: Python is build-time only. Nothing at runtime depends on `convert/`.
- Invariant: never re-host model weights. They come unmodified from the author's Hugging Face repo, so never commit, upload or bundle them.
- Project-specific overrides for the skills in `.claude/skills/`: see `.claude/skills/PROJECT_NOTES.md`.

## Working efficiently on long tasks

- A release (`release.yml`: Windows build, CUDA Docker images, desktop app) takes about 45 minutes, and PR CI takes about 5–10. Don't idle-wait on them: start the next piece of work in parallel, or end the turn with a status update.
- Batch releases: ship finished work together in one version bump, not one release per feature. Each release costs about 45 minutes of pipeline time.
- Report progress every 20–30 minutes on long work, including under `/goal`: what is done, what is running, what is next. The user sees only final messages.
- Slow local steps: CPU parity on 2B+ decoders (4+ minutes), downloading multi-GB weights, sha256 checks in `package.py`. Run them in the background, and do CUDA parity first for a quick signal.
- Open idea (not measured yet): a build cache (for example sccache) in the release workflow, mainly for the Windows and CUDA Docker jobs.
