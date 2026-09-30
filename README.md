# ralph

A local continuity runtime for stateful LLM inference. Ralph preserves session identity
and committed conversation history across vLLM worker crashes, daemon restarts, and GPU
memory pressure — a session is a durable thing with its own identity, not tied to any
one worker process that happens to be running it right now.

## Why

Running a local LLM directly through vLLM works fine until something goes wrong: a
worker crashes mid-conversation, you need to restart the daemon, or a second model
needs GPU memory another session is sitting on. Without Ralph, that's usually a lost
conversation. With Ralph, recovery is a command (`ralph recover`), and admission under
memory pressure is Ralph's own job (hibernate the cheapest idle session, not a manual
`kill`) — not something you have to engineer around yourself.

## Install

```bash
cargo install --path .                                # installs `ralph` to ~/.cargo/bin
python3 -m venv .venv && .venv/bin/pip install vllm    # vLLM, isolated, never vendored
```

`ralph doctor` checks GPU, vLLM, disk, and SQLite before you run anything else — safe
to run on a totally fresh machine, it never creates or mutates anything.

## 2-minute quickstart

```bash
ralph doctor
ralph run Qwen/Qwen2.5-0.5B-Instruct --name demo
ralph query demo "Remember the word pineapple."
ralph chat demo                                        # interactive; /exit or Ctrl-D
ralph ps
```

Crash survival, in one line: `kill -9` the worker process, then `ralph recover demo` —
the conversation and its history are still there.

## What's here

- A Rust CLI (`ralph`) that talks to a local daemon over a Unix socket
- SQLite WAL stores session metadata, accepted inputs, generated token IDs, and history
- Compatible sessions share one vLLM worker; it sleeps after five idle minutes and
  wakes for the next query
- `ralph run [model] [--name name] [--gpu index] [--policy ephemeral|warm|durable]
  [--continuity-target ms]` — start a model and create a session. With no model and a
  real terminal, shows a small interactive picker instead of failing. If a model
  doesn't fit, Ralph first tries to make room by hibernating the cheapest genuinely
  idle resident session (never one that's merely quiet) before rejecting
- `ralph migrate <session> --to <gpu>` — move an active session's worker to a
  different GPU on this machine, preserving session id/history. Always portable
  reconstruction (replay from durable history) — native KV never crosses GPUs
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
- `ralph completions <shell>` — print a completion script for bash/zsh/fish/elvish/
  powershell to stdout

## Building

`cargo install --path .` (see Install above) installs `ralph` to `~/.cargo/bin` — make
sure that's on your `PATH`. **A running daemon keeps whatever build was in memory when
it started**; a fresh build only takes effect after that daemon is restarted
(`kill $(cat <data-dir>/ralph/daemon.pid)`, then any command auto-starts a new one).

Shell completions: `ralph completions zsh > ~/.zsh/completions/_ralph` (or
bash/fish/elvish/powershell).

## Running against real vLLM

Ralph finds vLLM by checking, in order: an activated venv (`$VIRTUAL_ENV`), a `.venv` in
the current directory, then whatever `vllm` resolves to on `PATH` — never vendored into
the repo.

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

vLLM's automatic prefix caching is always explicitly enabled (`--enable-prefix-caching`,
separate from the disk-offload tier above) — an intentional Ralph decision rather than
relying on vLLM's own internal default, so repeated queries sharing a prompt prefix
reuse cached KV blocks in VRAM.

## Demos

### The strongest one: pressure-aware admission under real VRAM pressure

Real run, real GPU (RTX 5050 Laptop, 8151 MiB), a second process holding ~3 GiB to
force genuine pressure — not a mocked scenario:

```bash
ralph run Qwen/Qwen2.5-0.5B-Instruct --name coding
ralph query coding "Remember the secret word papaya. Reply with that word only."
# ... coding sits idle long enough to be make-room-eligible ...
ralph run Qwen/Qwen3-0.6B --name research
```

`coding` doesn't fit alongside `research` under the VRAM this run had free. Ralph
hibernated the one idle, cheap-to-rebuild session instead of rejecting the new one:

```json
{"name":"research", ..., "made_room_for":"coding"}
```

Measured on that run: 1595 MiB free before, 1321 MiB after (immediately; `research`'s
own model weights account for the rest); `coding` correctly landed `hibernated`, never
touched mid-query. Once VRAM was freed again (pausing `research`), resuming `coding`
replayed its durable history (portable — this was `coding`'s first-ever worker, so no
native checkpoint existed yet to restore from) and it answered "Papaya" — the exact
word from before the eviction, context fully intact. A **make-room decision that skips
an idle-but-recent session** is covered separately by
`make_room_prefers_ephemeral_over_durable_regardless_of_rebuild_cost` and
`a_recently_active_session_is_never_demoted_to_make_room`
(`src/daemon/tests_admission.rs`) — the policy/target logic that isn't exercised by
this single real run.

### The rest, in short, on one session named `demo`:

- **Crash survival**: `kill -9 <worker pid>` → `ralph recover demo` — history survives.
- **Fast vs. portable resume**: `ralph checkpoint demo` → `ralph pause demo` →
  `ralph resume demo` restores natively from the KV directory (`native: true`); a
  missing/corrupt/incompatible checkpoint (including a vLLM upgrade) falls back to a
  full replay instead (`native: false`) — context is correct either way.
- **Hibernate/resume**: `ralph hibernate demo` → `ralph resume demo` — same round trip,
  tolerant of a failed checkpoint write.

`ralph handoff`/`ralph drain` (moving a session to another machine or GPU) are their own
workflow, documented above.

## Troubleshooting

- **Start with `ralph doctor`** — checks GPU, vLLM, disk, and SQLite before anything
  else, safe to run even before any session exists.
- **A change you just built doesn't seem to be taking effect** — the daemon is still
  running the old build; see the install-flow note above.
- **`GPU OOM` / `cannot start ... safely` right after killing a worker** — the old
  worker process may not have released VRAM yet; check
  `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader`
  for a process that no longer has a matching Ralph session (`ralph ps`) and kill it
  directly.
- **Terminal width / tiny-terminal wrapping**: not implemented — output isn't
  responsive to terminal width today; use `--json` for scripting or a wide terminal
  for the human-readable tables.

## Status

Fully validated on real hardware (one RTX 5050 Laptop GPU, real vLLM 0.30.0, real
SIGKILL/crash injection — see Testing below): session identity/recovery, checkpoint/
pause/resume/hibernate (fast and portable paths), export/import, local SSH handoff,
drain, pressure-aware admission (`make_room`), continuity policies, and engine-restart/
upgrade continuity.

Structurally complete but **not runnable on this development machine** (one GPU only),
because it needs real second-GPU hardware: `ralph migrate`'s actual GPU0→GPU1 transfer
and its dual-GPU acceptance suite (`tests/e2e_migration.rs`). The suite is written to
run unchanged on a real 2-GPU box or a cloud target such as Kaggle T4x2 — it skips
cleanly and honestly here rather than claiming untested success. If you run it on real
multi-GPU hardware, a report back is genuinely useful.

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
RALPH_E2E_VLLM=1 cargo test --test e2e_checkpoint_resume -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_hibernate -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_export_import -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_handoff_drain -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_fault_injection -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_benchmarks -- --ignored --test-threads=1
RALPH_E2E_VLLM=1 cargo test --test e2e_migration -- --ignored --test-threads=1
```

These use the locally built binary, isolated data roots, and small Qwen models: worker
crashes, daemon restarts during/after recovery, Ctrl-C/disconnect/cancellation,
checkpoint/pause/resume/hibernate fast-vs-portable decisions, export/import round trips
(with and without KV state, plus corrupted-artifact rejection), and drain. `handoff`'s
own SSH round trip additionally needs `RALPH_E2E_HANDOFF_DEST` set to a reachable
destination (see `tests/e2e_handoff_drain.rs`) and skips cleanly without one, the same way
`e2e_migration` skips cleanly on any machine with fewer than two real GPUs (runnable
unchanged on a real 2-GPU box or a cloud environment such as Kaggle T4x2 — Ralph has no
dependency on any specific provider). `e2e_fault_injection` is the
fault-injection suite: corrupted/removed KV checkpoints, a daemon killed mid-pause, and
truncated durable history — disk-full and SQLite-busy handling are covered at unit
level instead (`src/daemon/tests_checkpoint.rs`, `src/storage/tests.rs`).
`e2e_benchmarks` is validation tooling (recovery/checkpoint/pause/resume/hibernate
latency, KV vs. history size, handoff downtime), not a regression gate — it prints a
JSON report rather than asserting thresholds.
