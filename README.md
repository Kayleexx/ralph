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
- `ralph hibernate <session>` — a stronger pause: same GPU release, but best-effort
  (logical-only hibernation is fine if the KV checkpoint can't be saved). A background
  idle sweep also auto-hibernates sessions left active and untouched for a while
- `ralph export <session> [--output path] [--force] [--with-accel]` / `ralph import
  <path.ralph> [--name name]` — a session's durable history (and optionally its KV
  checkpoint) as a portable, checksummed `.ralph` archive; import lands paused and
  resumes fast on a matching machine, portable otherwise
- `ralph handoff <session> <user@host> [--name name]` — move a session to another Ralph
  installation over SSH; the source stays authoritative until the destination ACKs the
  import, so a dropped connection or rejected transfer never leaves both machines
  thinking they own it
- `ralph drain <location> [--to user@host] [--yes]` — empty one GPU (`gpu0`): hand its
  active sessions to a destination, or hibernate them in place with no destination.
  Prints a plan first; `--yes` executes it
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
A running daemon keeps its current build; restart it to pick up a new install.

## Running against real vLLM

vLLM runs in an isolated Python virtualenv, never vendored into the repo:

```bash
python3 -m venv .venv
.venv/bin/pip install vllm
```

Ralph finds it by checking, in order: an activated venv (`$VIRTUAL_ENV`), a `.venv` in
the current directory, then whatever `vllm` resolves to on `PATH`.

Then:

```bash
ralph doctor
ralph run Qwen/Qwen2.5-0.5B-Instruct --name demo
ralph query demo "Remember the word pineapple."
ralph chat demo
ralph ps
ralph --json inspect demo
ralph recover demo                                    # after worker loss
ralph hibernate demo && ralph resume demo              # release/reattach the GPU worker
ralph export demo --with-accel && ralph import demo.ralph --name demo-copy
ralph drain gpu0                                      # hibernates every active session
ralph handoff demo user@gpu-box                       # move it to another installation
```

Ctrl-C during `query`/`chat` cancels that one generation without touching the session.
Worker loss moves a session to `recovering`; automatic replacement has a bounded retry
budget, and `ralph recover` retries explicitly and preserves session ID and history.

`checkpoint`/`pause`/`resume`/`hibernate` use vLLM 0.30.0's built-in `OffloadingConnector`
+ `TieringOffloadingSpec` filesystem tier as the native KV backend — confirmed by real
SIGKILL-and-restart testing to survive worker death, not just in-process sleep. Each
session gets its own KV directory, gated by Ralph's own compatibility fingerprint (model
revision, tokenizer revision, engine, engine version, GPU name) rather than vLLM's own
directory hashing, which doesn't cover those fields. A missing, incompatible, or
never-yet-populated checkpoint always falls back to a portable (full-replay) resume —
`--fast-only` fails clearly instead of silently falling back.

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

GitHub Actions runs these on pull requests and pushes to `main`. GPU/real-vLLM tests are
a separate, manual check:

```bash
RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_phase3 -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_phase4 -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_phase5 -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_phase6 -- --ignored --test-threads=1
```

These use the locally built binary, isolated data roots, and small Qwen models: worker
crashes, daemon restarts during/after recovery, Ctrl-C/disconnect/cancellation,
checkpoint/pause/resume/hibernate fast-vs-portable decisions, export/import round trips
(with and without KV state, plus corrupted-artifact rejection), and drain. `handoff`'s
own SSH round trip additionally needs `RALPH_E2E_HANDOFF_DEST` set to a reachable
destination (see `tests/e2e_phase6.rs`) and skips cleanly without one, the same way the
multi-GPU tests skip without a second GPU.
