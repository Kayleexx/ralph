# ralph

A continuity runtime for stateful LLM inference. Ralph makes an inference session
outlive the process, GPU, or machine currently running it.

Your model process can die. Your inference session does not have to.

## What's here

- A Rust CLI (`ralph`) that talks to a local daemon over a Unix socket
- The daemon owns session lifecycle, SQLite-backed session metadata (with automatic
  schema migrations), and a vLLM process adapter (spawn, health-check, streaming
  generation via chat completions)
- `ralph run [model] [--name name]` — start a model and create a session. With no
  model and a real terminal, shows a small interactive picker instead of failing
- `ralph query <session> <prompt>` — one-shot/scriptable query, streamed, with safe
  Ctrl-C cancellation
- `ralph chat <session>` — interactive back-and-forth with a session
- `ralph ps` / `ralph inspect <session>` — list and inspect sessions (typo-tolerant:
  `ralph inspect dem` suggests `demo`)
- `ralph doctor` — check the local environment (GPU, vLLM, disk, SQLite), safe to run
  before anything else exists

## Building

```bash
cargo build
```

To run `ralph` as a bare command from anywhere (not just `cargo run --`):

```bash
cargo install --path .
```

This installs to `~/.cargo/bin/ralph` — make sure that directory is on your `PATH`.
**After reinstalling, kill any running daemon** so the new build actually takes effect
(replacing the binary on disk doesn't restart an already-running daemon process):

```bash
pkill -f "ralph __daemon"
```

## Running against real vLLM

vLLM runs in an isolated Python virtualenv, never vendored into the repo:

```bash
python3 -m venv .venv
.venv/bin/pip install vllm
```

Ralph finds it by checking, in order: an activated venv (`$VIRTUAL_ENV`, works from any
directory once you `source .venv/bin/activate`), a `.venv` in the current directory, then
whatever `vllm` resolves to on `PATH`.

Then:

```bash
ralph doctor                                          # all six checks should PASS
ralph run Qwen/Qwen2.5-0.5B-Instruct --name demo       # or just `ralph run` for the picker
ralph query demo "hello"                               # one-shot
ralph chat demo                                        # interactive; /exit or Ctrl-D to leave
ralph ps
ralph --json inspect demo
```

Ctrl-C during `query`/`chat` cancels that one generation without touching the session —
`ralph inspect demo` will still show `state: active` right after.

There's no `ralph stop` yet (that's a later phase), so to free GPU memory when you're
done, kill the worker directly:

```bash
pkill -9 -f "vllm serve"
```

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

The real-vLLM end-to-end test is gated behind an env var so it never runs by accident
on a machine without a GPU. It exercises the full lifecycle — start, query, Ctrl-C
cancellation, killing the daemon and confirming session metadata survives the restart,
and killing the vLLM worker directly and confirming the session is reported `stopped`
(never silently `active`, never wrongly `failed`):

```bash
RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
```
