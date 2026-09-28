# ralph

A continuity runtime for stateful LLM inference. Ralph makes an inference session
outlive the process, GPU, or machine currently running it.

Your model process can die. Your inference session does not have to.

## What's here

- A Rust CLI (`ralph`) that talks to a local daemon over a Unix socket
- The daemon owns session lifecycle, SQLite-backed session metadata, and a vLLM
  process adapter (spawn, health-check, streaming generation)
- `ralph run <model> --name <name>` — start a model and create a session
- `ralph query <session> <prompt>` — query it, streamed, with safe Ctrl-C cancellation
- `ralph ps` / `ralph inspect <session>` — list and inspect sessions
- `ralph doctor` — check the local environment (GPU, vLLM, disk, SQLite)

## Building

```bash
cargo build
```

## Running against real vLLM

vLLM runs in an isolated Python virtualenv, never vendored into the repo:

```bash
python3 -m venv .venv
.venv/bin/pip install vllm
```

Then:

```bash
cargo run -- run Qwen/Qwen2.5-0.5B-Instruct --name demo
cargo run -- query demo "hello"
```

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

The real-vLLM end-to-end test is gated behind an env var so it never runs by accident
on a machine without a GPU:

```bash
RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
```
