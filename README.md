# ralph

A local continuity runtime for stateful LLM inference. Ralph preserves session identity
and committed conversation history across vLLM worker crashes and daemon restarts.

## What's here

- A Rust CLI (`ralph`) that talks to a local daemon over a Unix socket
- SQLite WAL stores session metadata, accepted inputs, generated token IDs, and history
- Compatible sessions share one vLLM worker; it sleeps after five idle minutes and
  wakes for the next query
- `ralph run [model] [--name name]` — start a model and create a session. With no
  model and a real terminal, shows a small interactive picker instead of failing
- `ralph query <session> <prompt>` — one-shot/scriptable query, streamed, with safe
  Ctrl-C cancellation
- `ralph recover <session>` — validate durable history, start or reuse a compatible
  worker, and reconstruct context through real prefill before returning to active
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
A running daemon keeps its current build. Stop the specific daemon for the intended
data root before using a newly installed binary.

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
ralph doctor
ralph run Qwen/Qwen2.5-0.5B-Instruct --name demo
ralph query demo "Remember the word pineapple."
ralph chat demo
ralph ps
ralph --json inspect demo
ralph recover demo                                    # after worker loss
```

Ctrl-C during `query`/`chat` cancels that one generation without touching the session —
`ralph inspect demo` will still show `state: active` right after.

Worker loss moves a session to `recovering`. Automatic replacement has a durable
three-attempt budget; repeated failures end in `stopped`. `ralph recover demo` retries
explicitly and preserves the session ID and history. Repeating recovery on an active
session is a safe no-op.

Only one model may be starting or resident in a daemon, including sleeping workers.
Compatible sessions share that worker; another model receives a resource conflict.
Startup errors leave the session stopped with its history intact. `-v` exposes the
bounded startup-log tail and its path.

Durable history and token counts are committed together in SQLite WAL transactions.
Legacy sessions with missing token data report degraded recoverability and incomplete
portable state; recovery preserves that evidence and refuses to invent history.

Recovery reconstructs logical context; native KV checkpoints and cross-machine recovery
are not implemented. The tested model above fits the local GPU; Qwen3-0.6B's default
context exceeded its KV capacity in the verification environment.

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

GitHub Actions runs these CPU checks on pull requests and pushes to `main`, using
pinned Rust/actions, Swatinem dependency caching, and sccache. Only successful main
pushes save dependency caches; PRs use sccache in read-only mode. Superseded runs cancel.
GPU tests remain a separate hardware check.

To enforce review of CI changes, protect `main` with the required `rust` check from
GitHub Actions and code-owner approval, dismiss stale approvals, and disallow bypasses
and force pushes. The `.github/CODEOWNERS` file assigns CI review to `@Kayleexx`;
it needs those repository settings to enforce approval.

The real-vLLM tests use the locally built binary, isolated data roots, and
`Qwen/Qwen2.5-0.5B-Instruct`. They cover worker crashes before a query, before output,
after an exchange, and after a partial flush; daemon restarts during/after recovery;
Ctrl-C, disconnect, startup failure, and recovery cancellation. Teardown signals only
verified test-owned workers and direct-child daemons.

```bash
RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
```
