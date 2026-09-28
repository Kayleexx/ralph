# Phase 2 verification

Worktree: `.claude/worktrees/debug-duplicate-name`. Existing work and user session
state were preserved. No dependencies, Phase 3 work, timeout changes,
memory-setting changes, or offload workarounds were introduced.

## Original startup failures

The existing Qwen3-0.6B log records a 261.012964-second weight download, followed by a
KV-capacity failure: 40,960 tokens needed 4.38 GiB, but only 2.94 GiB was available.
A longer startup timeout would not fix that capacity failure.

The Qwen2.5 log records 3.78 GiB free versus 4.53 GiB requested. Its startup overlapped
Qwen3's download and initialized CUDA process. That process is a plausible contributor;
there is no contemporaneous GPU allocation snapshot proving the complete breakdown.

## Recovery behavior

- Startup is reserved before spawning. Registry locks are released before network or
  process waits. Compatible model/tokenizer identities reuse a worker; a different
  resident model, including a sleeping worker, is rejected with the session preserved.
- Private ownership records store boot/start identity, process group, inherited nonce,
  endpoint, and model revision immediately after spawn. Reconciliation validates model
  and process identity, cleans owned orphans, and requires explicit recovery.
- Startup failure, cold-start cancellation, and worker loss stop the verified process group and
  reap the direct child before replacement. Retry logs are appended, and verbose error
  envelopes expose a bounded tail with the originating exception.
- Session recovery is locked and follows stopped → recovering → active. Active recovery
  is a no-op. The three-attempt automatic budget survives replacement and daemon restart;
  successful generation resets it while the last failure remains inspectable.
- Accepted user tokens and an assistant reservation are committed before generation.
  Cumulative assistant content and token IDs are committed with session counts. Retried
  writes cannot duplicate output; failed writes never advance the committed offset.
- Context preflight uses the installed `/tokenize` API and streamed token IDs. Recovery
  validates durable history and invokes the installed zero-output echo/prefill API
  before activation. Stream errors, cancellation, and disconnect attempt a final flush.
- SQLite WAL handles interrupted commits. Invalid ordering, roles, completion status,
  token IDs/counts, missing reservations, and per-turn payload checksums fail closed.
  Legacy missing history is reported as degraded/incomplete without altering the evidence.

The first manual check also found that EngineCore's process-title update erased its
ownership environment marker. The verifier refused the signal. vLLM workers now use
`SPT_NOENV=1`, the native [setproctitle environment-preservation option](https://github.com/dvarrazzo/py-setproctitle#environment-variables).
Hardware startup now verifies every live group member before proceeding, as well as
before signaling it. The earlier demo's worker was cleaned using its recorded live
leader identity and verified direct-child ancestry; its data was retained.

## Environment

RTX 5050 Laptop GPU, 8,151 MiB reported total; NVIDIA driver 610.57.04;
vLLM 0.30.0; Rust 1.96.1. Recovery model:
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.

## Checks

- `cargo fmt --check`: passed.
- `cargo clippy --all-targets --all-features -- -D warnings`: passed.
- `cargo test --all`: 109 unit tests and 12 CLI contract tests passed; all three
  hardware cases are intentionally ignored in this regular invocation.
- Every Rust source file under `src` and `tests` is at most 400 lines (largest: 392).
- `git diff --check`: passed. Final tracked and newly created source files were reviewed.

## Real GPU acceptance

The required command passed after the recovery, ownership, and durability changes:

```sh
RALPH_E2E_VLLM=1 cargo test --test e2e_vllm -- --ignored --test-threads=1
```

**3 passed, 0 failed, 0 ignored, 247.83 seconds.** Output was retained at
`/tmp/ralph-hardware-acceptance.log`.

The cases cover verified SIGKILL before a query, before the first client output token,
after a committed exchange, and during generation after a durable partial flush;
replacement and context reconstruction; unchanged session identity and no duplicate
turns; daemon restart during and after recovery; surviving owned orphan cleanup;
Ctrl-C query and recovery cancellation; client disconnect; bounded missing-model
startup failure ending stopped; and rejection of a different resident model.

## Manual demo

After the GPU suite, the worktree CLI was exercised command by command against
`Qwen/Qwen2.5-0.5B-Instruct`, without mocks, in this retained isolated data root:

`/home/mitali/.local/share/ralph-phase2-manual-20260928T231111Z-af6e7e`

- Session: `manual-recovery`, ID `01M3N4JQ156YSF09M509GJSMGV`.
- A large prompt containing 350 repeated sentences and a request to remember
  “lighthouse” returned “lighthouse”; two turns and 3,516 tokens were durable.
- Boot/start identity, process group, and ownership nonce were checked for all live
  members. SIGKILL stopped group 294309 (members 294309, 294403, 294404).
- Ralph reported recovering with unchanged history and token count. `ralph recover`
  returned the same ID, active, on replacement worker 295089, with identical history.
- The continued query recalled “lighthouse”; four durable turns and 3,534 tokens
  remained, with no duplicate turns. Inspection reported safe/complete portable state.
- The verified demo daemon (294283) and replacement group were stopped; databases,
  logs, history snapshots, `manual-result.json`, and `teardown.json` were retained.

## Remaining limits

No native KV or RNG checkpoint is implemented in Phase 2. Recovery reconstructs the
logical conversation from committed history; it does not promise identical sampling
or reconstruct output that never committed. Legacy missing token data remains
explicitly degraded. Payload checksums detect accidental corruption, not malicious
rewrites; migration cannot prove the earlier integrity of pre-checksum rows.

Qwen3's full default context still requires more KV capacity than these unchanged
engine settings provide. That resource limitation is separate from session recovery.

## Real capacity-failure verification

The additional Qwen3 startup probe initially exposed a diagnostic defect: an 80-line
limit omitted the originating error above a 125-frame wrapper traceback, and vLLM
spelled the wrapper “Engine core”. The diagnostic reader now preserves its bounded
16 KiB byte window and excludes both wrapper spellings. A focused regression,
formatting, strict Clippy, and the complete regular suite passed after this final fix.
The already-passing recovery tests and manual demo preceded this diagnostic-only fix.

The real startup probe was repeated successfully in 29.46 seconds. Normal JSON reported:

> Qwen/Qwen3-0.6B: To serve at least one request with the model's max seq len
> (40960), (4.38 GiB KV cache is needed, which is larger than the available
> KV cache memory (2.96 GiB).

Resource exit code was 6; session `01M3N4RFP4HPHNCJ5J56H4Z494` ended stopped with no
PID, zero accepted tokens, and no worker ownership record remaining. Verbose JSON
included a 16,512-byte diagnostic including the log path. Its verified direct-child
daemon was stopped. Logs and `capacity-result.json` remain at:

`/home/mitali/.local/share/ralph-phase2-capacity-20260928T231420Z-9f64e3`

The earlier failed diagnostic probe's files were also preserved at
`/home/mitali/.local/share/ralph-phase2-capacity-20260928T231248Z-f61694`.

All required Phase 2 recovery acceptance checks passed. No known Phase 2 recovery
issue remains in the tested configuration; the model capacity and legacy-data limits
above remain explicit. The installed user daemon and existing session data were untouched.
