# CLAUDE.md — Ralph engineering rules

This file is engineering rules only. `RALPH_SPEC.md` is the single source of truth for
the product: what Ralph does, its CLI surface, session states, architecture, phases, and
edge cases. If anything here seems to conflict with `RALPH_SPEC.md`, the spec wins —
flag the conflict instead of resolving it silently.

## Core principles

- Rust is the primary implementation language.
- `unsafe` is forbidden across the entire codebase: `#![forbid(unsafe_code)]` at the crate root.
- Keep the codebase small, boring, explicit, and maintainable.
- Do not add architecture, abstractions, dependencies, services, or features unless the
  current phase (RALPH_SPEC.md §17) genuinely needs them. No speculative abstractions,
  no future-proofing.
- Prefer existing reliable primitives over inventing new subsystems.
- Implement the current phase only. Do not start future phases early.
- No placeholder implementations, fake production paths, or mocks for Ralph's core
  behavior (see RALPH_SPEC.md §14 — no fake "success" path for the main feature).
- Do not silently change product behavior described in RALPH_SPEC.md.
- Before adding anything, run it through RALPH_SPEC.md §18's scope guardrails.

## File / module rules

- No source file may exceed 400 lines; aim substantially below that.
- If a file approaches the limit, split by real responsibility, not arbitrary chunks.
- One module, one clear purpose. No `utils.rs` / `helpers.rs` / `common.rs` junk drawers,
  no god modules.
- Start with the smallest crate/module structure that works. Do not create multiple
  crates without a concrete need.
- Keep public APIs minimal.

## Rust quality

- Idiomatic stable Rust. Keep clean and passing before considering work done:
  - `cargo fmt --check`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --all`
- Warnings are errors.
- Prefer strong types over strings/booleans when they encode domain state (session
  states are the canonical example — RALPH_SPEC.md §8).
- Use enums for lifecycle/state transitions. Make invalid states hard to represent.
- Avoid unnecessary cloning/allocations, but correctness and clarity come before
  optimizing blindly.
- No macros unless they clearly improve the code. No clever code when straightforward
  code works.

## Error handling

- `unwrap()`, `expect()`, `panic!`, `unreachable!`, `todo!`, `unimplemented!` are
  forbidden in production code paths.
- Errors are always propagated or handled intentionally; never discard one with
  `let _ = ...` unless failure is provably irrelevant, and say why in a comment.
- Never turn a recoverable error into a panic.
- Define typed domain errors with `thiserror` where callers need to distinguish failure
  kinds. Use `anyhow` only at application/CLI/daemon boundaries, with `.context(...)` /
  `.with_context(...)` at I/O/process/system boundaries.
- User-facing errors must answer: what failed, what state Ralph is left in, what the
  user can do next (this is the CLI UX contract in RALPH_SPEC.md §7.2, principle 3).
  Preserve underlying causes for verbose/debug output.
- Tests may use `unwrap`/`expect` when it makes the test clearer and the failure itself
  represents a broken test invariant.

## State and durability

Ralph is stateful infrastructure; correctness around state outranks convenience.

- Worker failure must never automatically become session failure.
- Session state transitions are explicit and validated.
- Persistent state changes are crash-safe; never silently lose or overwrite session state.
- Acceleration state is disposable; durable logical state is authoritative. Never
  restore acceleration state without validating its compatibility fingerprint.
- Interrupted operations must leave ownership/state unambiguous.
- Avoid hidden side effects. Make retry/idempotency behavior explicit where relevant.
- Ground every design decision here in RALPH_SPEC.md §13's invariants — they're the spec's
  contract, not engineering style, so implement to them rather than reinterpreting them.

## Concurrency

- Do not hold locks across `.await`.
- Prefer message passing or narrower ownership over shared mutable state; avoid
  unnecessary `Arc<Mutex<_>>`.
- Document non-obvious synchronization invariants.
- Cancellation must leave state consistent. Concurrent operations on the same session
  must have clearly defined behavior (per-session locks — RALPH_SPEC.md §16.1).
- Do not "fix" races with sleeps. Tests for async state changes wait on explicit
  conditions, never arbitrary sleeps.

## Dependencies

- Dependencies must earn their place: prefer mature, focused crates.
- Before adding one, check whether std/Tokio/current dependencies already solve it.
- No large frameworks for tiny functionality. No paid services or cloud dependencies.
- Core development must run locally on the workstation described in RALPH_SPEC.md §3/§14.1.

## Comments and docs

- Minimal comments. Don't narrate obvious code.
- Comment only: non-obvious invariants, tricky correctness constraints,
  safety/recovery reasoning, external protocol quirks.
- Prefer clear names and structure over comments. Public APIs get concise docs only
  when their contract isn't obvious. No AI-style essay comments.
- Bad: `// increment the counter` / `counter += 1;`
- Good: `// Ownership is committed only after the destination durably acknowledges the session.`
- Remove stale comments when the code they describe changes.

## Testing

- Test behavior, not implementation details.
- Every bug fix includes a regression test where practical.
- State-machine transitions and failure paths are first-class tests, not afterthoughts.
- Use real vLLM integration for core end-to-end behavior when the phase requires it — no
  mocked engine pretending to prove crash recovery works.
- Tests are deterministic; no sleep-based synchronization.
- Follow RALPH_SPEC.md §14's testing strategy and §16.15's fault-injection list for what
  each phase must cover — don't re-derive it here.

## CLI / UX

- Ralph is CLI-first; follow RALPH_SPEC.md §7 for the actual UX contract (defaults,
  output modes, exit codes, Ctrl-C safety, idempotency).
- Normal output stays concise and human-readable; errors stay actionable; internal
  implementation details don't leak into normal output.
- Preserve scripting support: stable exit codes, `--json` where specified.
- Prefer sensible defaults over extra flags. Do not grow a huge command hierarchy.

## Workflow for every task

Before editing:
1. Read the relevant section of RALPH_SPEC.md.
2. Inspect the existing implementation.
3. Identify the smallest change required.
4. Do not implement unrelated cleanup or features.

After editing:
1. `cargo fmt`.
2. `cargo clippy --all-targets --all-features -- -D warnings`.
3. Run relevant tests, then the full suite when reasonable.
4. Confirm every touched source file is ≤400 lines.
5. Remove dead code, debug prints, temporary files, stale comments, unused dependencies.
6. Summarize what changed and any unresolved issue honestly.

## Git

- Do not commit or push unless explicitly asked.
- Do not add `Co-authored-by` lines.
- Do not rewrite unrelated user changes.
- Do not run destructive git commands without explicit permission.

## Anti-slop rule

If a solution needs several new abstractions/modules/dependencies to solve a small
problem, stop and find a simpler design. Ralph should feel sophisticated because its
correctness is strong, not because its codebase is complicated.
