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
- `ralph checkpoint <session>` — record a durable pointer to the session's current KV
  cache directory and compatibility fingerprint, for a faster future resume
- `ralph pause <session>` / `ralph resume <session>` (`--fast-only` / `--portable`) —
  free the session's GPU worker, then reattach it, fast (native KV reuse) when a
  still-compatible checkpoint has real content, portable (full replay) otherwise
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

On the RTX 5050 Laptop, Qwen2.5-0.5B and Qwen3-0.6B can share the GPU using
measured BF16/eager profiles: Qwen2.5 selects 16k/448 MiB or 8k/256 MiB
context/KV, and Qwen3 selects 4k/960 MiB or 2k/512 MiB. Ralph picks the largest
profile that fits measured worker memory plus 150 MiB per worker and 1 GiB global
headroom. Compatible sessions reuse a worker; sleeping workers retain wake reservations.
The chosen budget and model revision persist through recovery, and resident budgets
never resize. Other models retain single-model operation; TinyLlama co-residency is
disabled pending investigation of its paired throughput regression.
Startup errors leave the session stopped with its history intact. `-v` exposes the
bounded startup-log tail and its path.

Durable history and token counts are committed together in SQLite WAL transactions.
Legacy sessions with missing token data report degraded recoverability and incomplete
portable state; recovery preserves that evidence and refuses to invent history.

Recovery reconstructs logical context; cross-machine recovery is not implemented. The
Qwen profiles were measured with vLLM 0.30.0 on the RTX 5050 Laptop; unmeasured
GPU/model identities are refused for these budgets.

`ralph checkpoint`/`pause`/`resume` use vLLM 0.30.0's built-in `OffloadingConnector` +
`TieringOffloadingSpec` filesystem tier as the native KV backend — confirmed by real
SIGKILL-and-restart testing to survive worker death, not just in-process sleep. Each
session gets its own KV directory, gated by Ralph's own compatibility fingerprint (model
revision, tokenizer revision, engine, engine version, GPU name), since vLLM's own
directory hashing does not cover those fields. A missing, incompatible, or
never-yet-populated checkpoint always falls back to a portable (full-replay) resume —
`ralph resume` never fakes a native restore, and `--fast-only` fails clearly instead of
silently falling back.

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
the small Qwen models. They cover worker crashes before a query, before output,
after an exchange, and after a partial flush; daemon restarts during/after recovery;
Ctrl-C, disconnect, startup failure, and recovery cancellation. Teardown signals only
verified test-owned workers and direct-child daemons.
Two-worker checks cover independent/concurrent queries, isolated worker recovery,
daemon reconciliation, context/VRAM rejection, and concurrent startup ownership.

```bash
RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_phase3 -- --ignored --test-threads=1
```

`e2e_phase3` covers checkpoint/pause/resume against real vLLM: a fast resume after real
generation populated the KV directory, a deleted-directory portable fallback, an
incompatible-fingerprint `--fast-only` rejection, and `--portable` always forcing full
replay.
