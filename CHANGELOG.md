# Changelog

Ralph is pre-1.0 (`0.1.0`); entries are grouped by theme rather than version number,
since nothing has been tagged/released yet.

## Unreleased — release polish

- shell completions (`ralph completions <shell>`)
- CI job building a Linux x86_64 release binary on a `v*` tag push
- documented compatibility promises for the `.ralph` archive format and the SQLite
  schema migration sequence
- README: expanded install/build notes, new troubleshooting section

## Core hardening

- disk-full handling before checkpoint/export writes, refused cleanly instead of
  attempted
- corrupted KV checkpoint content is now detected (a lightweight recursive
  content-integrity signature) and rejected before vLLM ever sees it, falling back
  to portable resume — real vLLM 0.30.0 does not reliably error on corrupted
  `OffloadingConnector` tier files itself
- daemon crash recovery extended: a session stuck mid-`pause`/`hibernate`/`resume`
  (not just mid-`handoff`/`drain`) now rolls back to a stable state on restart
  instead of sticking forever
- orphaned artifact sweep on daemon startup (stale `/dev/shm` offload segments,
  abandoned handoff-receive staging files, rowless session directories)
- real SQLite busy-contention and state-machine rollback-completeness test coverage

## Remote handoff, drain, and preemption

- `ralph handoff` — move a session to another Ralph installation over SSH with a
  two-phase commit (source stays authoritative until the destination ACKs import)
- `ralph drain` — empty one GPU by handing sessions off or hibernating them in place
- graceful SIGTERM checkpoint sweep before daemon shutdown

## Portable session artifacts

- `ralph export`/`ralph import` — checksummed `.ralph` archives, with or without KV
  acceleration state, importable on a different machine

## Hibernation

- `ralph hibernate` — a stronger pause that tolerates a failed KV checkpoint write
  (logical-only hibernation) rather than blocking on it
- idle-based auto-hibernation sweep

## Checkpoint, pause, resume

- `ralph checkpoint`/`ralph pause`/`ralph resume` — release and reattach a session's
  GPU worker, fast (native KV reuse) when a compatible checkpoint exists, portable
  (full replay) otherwise

## Durable recovery

- crash-safe token history, bounded automatic worker restart, `ralph recover`

## Session identity

- `ralph run`/`ralph query`/`ralph ps`/`ralph inspect` — sessions own their identity
  and durable history independent of any one vLLM worker process
