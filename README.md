# ralph

A local continuity runtime for stateful LLM inference. Ralph preserves session identity
and committed conversation history across vLLM worker crashes, daemon restarts, and GPU
memory pressure. A session is a durable thing with its own identity, not something tied
to whichever worker process happens to be running it right now.

## Why

Running a local LLM directly through vLLM works fine until something goes wrong: a
worker crashes mid-conversation, you need to restart the daemon, or a second model needs
GPU memory another session is sitting on. Without Ralph, that's usually a lost
conversation. With Ralph, recovery happens on your next message, no command to remember,
and admission under memory pressure is Ralph's own job: hibernate the cheapest idle
session instead of forcing you to manually `kill` something.

## Install

```bash
cargo install --path .                                # installs `ralph` to ~/.cargo/bin
python3 -m venv .venv && .venv/bin/pip install vllm    # vLLM, isolated, never vendored
```

`ralph doctor` checks GPU, vLLM, disk, and SQLite before you run anything else. It's safe
to run on a totally fresh machine since it never creates or mutates anything.

## 2-minute quickstart

```bash
ralph doctor
ralph run Qwen/Qwen2.5-0.5B-Instruct --name demo
ralph chat demo                                        # interactive; /exit or Ctrl-D
ralph ps
```

Crash survival needs nothing extra. Kill the worker mid-conversation
(`kill -9 <worker pid>`, or just let vLLM die) and keep typing in the same `ralph chat`:
Ralph recovers it before your next message goes through. `ralph query` (scriptable,
one-shot) gets the same transparent recovery, see the demo below.

## What's here

`chat` and `query` recover or resume a session automatically when needed. `recover`,
`pause`, `resume`, and `hibernate` below stay available when you want explicit, scripted
control instead.

- A Rust CLI (`ralph`) that talks to a local daemon over a Unix socket.
- SQLite WAL stores session metadata, accepted inputs, generated token IDs, and history.
- Compatible sessions share one vLLM worker. It sleeps after five idle minutes and wakes
  for the next query.
- `ralph run [model] [--name name] [--gpu index] [--policy ephemeral|warm|durable]
  [--continuity-target ms]`: start a model and create a session. With no model and a
  real terminal, this shows a small interactive picker instead of failing. If a model
  doesn't fit, Ralph first tries to make room by hibernating the cheapest genuinely idle
  resident session (never one that's merely quiet) before rejecting.
- `ralph migrate <session> --to <gpu>`: move an active session's worker to a different
  GPU on this machine, preserving session id and history. Always a portable
  reconstruction (replay from durable history), since native KV never crosses GPUs.
- `ralph query <session> <prompt>`: one-shot, scriptable query, streamed, with safe
  Ctrl-C cancellation.
- `ralph recover <session>`: validate durable history, start or reuse a compatible
  worker, and reconstruct context through real prefill before returning to active.
- `ralph checkpoint <session>`: record a durable pointer to the session's current KV
  cache directory and compatibility fingerprint, for a faster future resume.
- `ralph pause <session>` / `ralph resume <session>` (`--fast-only` / `--portable`):
  free the session's GPU worker, then reattach it. Fast (native KV reuse) when a
  still-compatible checkpoint has real content, portable (full replay) otherwise.
- `ralph hibernate <session>`: a stronger pause with the same GPU release, but
  best-effort: logical-only hibernation is fine if the KV checkpoint can't be saved. A
  background idle sweep also auto-hibernates sessions left active and untouched for a
  while.
- `ralph export <session> [--output path] [--force] [--with-accel]` / `ralph import
  <path.ralph> [--name name]`: a session's durable history (and optionally its KV
  checkpoint) as a portable, checksummed `.ralph` archive. Import lands paused and
  resumes fast on a matching machine, portable otherwise.
- `ralph handoff <session> <user@host> [--name name]`: move a session to another Ralph
  installation over SSH. The source stays authoritative until the destination
  acknowledges the import, so a dropped connection or rejected transfer never leaves
  both machines thinking they own it.
- `ralph drain <location> [--to user@host] [--yes]`: empty one GPU (`gpu0`) by handing
  its active sessions to a destination, or hibernating them in place with no
  destination. Prints a plan first; `--yes` executes it.
- `ralph chat <session>`: interactive back-and-forth with a session.
- `ralph ps` / `ralph inspect <session>`: list and inspect sessions. Typo-tolerant:
  `ralph inspect dem` suggests `demo`.
- `ralph doctor`: check the local environment (GPU, vLLM, disk, SQLite). Safe to run
  before anything else exists.
- `ralph completions <shell>`: print a completion script for bash, zsh, fish, elvish, or
  powershell to stdout.

## Building

`cargo install --path .` (see Install above) installs `ralph` to `~/.cargo/bin`, so make
sure that directory is on your `PATH`. **A running daemon keeps whatever build was in
memory when it started.** A fresh build only takes effect once that daemon is restarted
(`kill $(cat <data-dir>/ralph/daemon.pid)`, then any command auto-starts a new one).

Shell completions: `ralph completions zsh > ~/.zsh/completions/_ralph` (or
bash, fish, elvish, powershell).

## Running against real vLLM

Ralph finds vLLM by checking, in order: an activated venv (`$VIRTUAL_ENV`), a `.venv` in
the current directory, then whatever `vllm` resolves to on `PATH`. It's never vendored
into the repo.

Ctrl-C during `query` or `chat` cancels that one generation without touching the
session. Worker loss moves a session to `recovering`, and the next `query` or `chat`
message transparently recovers it first (`ralph recover` remains available for
explicit, scripted control). A background bounded-retry sweep also attempts replacement
on its own.

`checkpoint`, `pause`, `resume`, and `hibernate` use vLLM 0.30.0's built-in
`OffloadingConnector` plus a `TieringOffloadingSpec` filesystem tier as the native KV
backend, confirmed by real SIGKILL-and-restart testing to survive worker death, not
just in-process sleep. Each session gets its own KV directory, gated by Ralph's own
compatibility fingerprint (model revision, tokenizer revision, engine, engine version,
GPU name) rather than vLLM's own directory hashing, which doesn't cover those fields. A
missing, incompatible, or never-yet-populated checkpoint always falls back to a
portable, full-replay resume. `--fast-only` fails clearly instead of silently falling
back.

vLLM's automatic prefix caching is always explicitly enabled (`--enable-prefix-caching`,
separate from the disk-offload tier above). This is an intentional Ralph decision rather
than relying on vLLM's own internal default, so repeated queries sharing a prompt prefix
reuse cached KV blocks in VRAM.

## Demos

### The main one: a single `ralph chat` survives a real crash

One `ralph chat` process stays open the whole time. The worker is killed from another
terminal, and the very next message in that *same* chat transparently recovers it. Real
run, real GPU, real vLLM:

```
$ ralph run Qwen/Qwen2.5-0.5B-Instruct --name research
research ready · 23.1s

$ ralph chat research
you › Remember this word: marigold. Reply with just OK.
ralph › OK

# (worker killed from another terminal: a crash, not a command)

you › What word did I ask you to remember?
recovering research
ready · 22.0s
ralph › The word you asked me to remember is "marigold".
```

No `ralph recover` typed, no reopening chat. Same process, same conversation. Reopening
`chat` later, after the session has gone to sleep, behaves the same way:

```
$ ralph chat research
restoring research
ready · 1.2s
ralph › (conversation continues, context intact)
```

### Pressure-aware admission (a second demo)

Starting a second model that doesn't fit makes room by sleeping the idle one instead of
rejecting. Real run, with a second process holding about 1 GiB to force genuine pressure
on an 8 GB GPU (two small Qwen models otherwise both fit without any contention):

```
$ ralph run Qwen/Qwen3-0.6B --name coding
making room · sleeping research
coding ready · 26.0s
```

`research` landed `hibernated`, never touched mid-query, and resumed later with its
exact prior context intact. A **make-room decision that skips an idle-but-recent
session** is covered separately by
`make_room_prefers_ephemeral_over_durable_regardless_of_rebuild_cost` and
`a_recently_active_session_is_never_demoted_to_make_room`
(`src/daemon/tests_admission.rs`).

### Explicit control, for when you want it

The same things `chat` and `query` now do automatically are still available as their own
commands, for scripting or manual control:

- **Crash survival by hand**: `kill -9 <worker pid>`, then `ralph recover demo`.
- **Fast vs. portable resume**: `ralph checkpoint demo`, then `ralph pause demo`, then
  `ralph resume demo`. It restores natively from the KV directory when possible, and
  falls back to a full replay otherwise. Context is correct either way
  (`--verbose`/`--json` show which path was taken).
- **Hibernate/resume**: `ralph hibernate demo`, then `ralph resume demo`. Same round
  trip, tolerant of a failed checkpoint write.

`ralph handoff` and `ralph drain` (moving a session to another machine or GPU) are their
own workflow, documented above.

## Troubleshooting

- **Start with `ralph doctor`**. It checks GPU, vLLM, disk, and SQLite before anything
  else, and it's safe to run even before any session exists.
- **A change you just built doesn't seem to be taking effect**: the daemon is still
  running the old build, see the install-flow note above.
- **`GPU OOM` or `cannot start ... safely` right after killing a worker**: the old
  worker process may not have released VRAM yet. Check
  `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader`
  for a process that no longer has a matching Ralph session (`ralph ps`) and kill it
  directly.
- **Terminal width / tiny-terminal wrapping**: not implemented. Output isn't responsive
  to terminal width today, so use `--json` for scripting or a wide terminal for the
  human-readable tables.

## Status

Fully validated on real hardware (one RTX 5060 Laptop GPU, real vLLM 0.30.0, real
SIGKILL/crash injection, see Testing below): session identity and recovery,
checkpoint/pause/resume/hibernate (fast and portable paths), export/import, local SSH
handoff, drain, pressure-aware admission (`make_room`), continuity policies, and
engine-restart/upgrade continuity.

Structurally complete but **not runnable on this development machine** (one GPU only),
because it needs real second-GPU hardware: `ralph migrate`'s actual GPU0-to-GPU1
transfer and its dual-GPU acceptance suite (`tests/e2e_migration.rs`). The suite is
written to run unchanged on a real 2-GPU box or a cloud target such as Kaggle T4x2. It
skips cleanly and honestly here rather than claiming untested success. If you run it on
real multi-GPU hardware, a report back is genuinely useful.

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

GitHub Actions runs these on pull requests and pushes to `main`. GPU and real-vLLM tests
are a separate, manual check:

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
crashes, daemon restarts during and after recovery, Ctrl-C/disconnect/cancellation,
checkpoint/pause/resume/hibernate fast-vs-portable decisions, export/import round trips
(with and without KV state, plus corrupted-artifact rejection), and drain. `handoff`'s
own SSH round trip additionally needs `RALPH_E2E_HANDOFF_DEST` set to a reachable
destination (see `tests/e2e_handoff_drain.rs`), and skips cleanly without one, the same
way `e2e_migration` skips cleanly on any machine with fewer than two real GPUs (runnable
unchanged on a real 2-GPU box or a cloud environment such as Kaggle T4x2; Ralph has no
dependency on any specific provider). `e2e_fault_injection` is the fault-injection
suite: corrupted/removed KV checkpoints, a daemon killed mid-pause, and truncated
durable history. Disk-full and SQLite-busy handling are covered at unit level instead
(`src/daemon/tests_checkpoint.rs`, `src/storage/tests.rs`). `e2e_benchmarks` is
validation tooling (recovery/checkpoint/pause/resume/hibernate latency, KV vs. history
size, handoff downtime), not a regression gate: it prints a JSON report rather than
asserting thresholds.
