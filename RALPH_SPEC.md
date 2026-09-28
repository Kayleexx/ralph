# Ralph

> **A continuity runtime for stateful LLM inference.**
>
> Ralph makes an inference session outlive the process, GPU, or machine currently running it.

This file is the single source of truth for the project. Claude Code/Codex should follow it exactly. Do not add unrelated features or expand the architecture unless this document is updated first.

**Spec version:** 3  
**Last updated:** 2026-09-28  
**Priority:** correctness and recoverability first, user experience second, optimization third.

---

## 1. What Ralph is

Ralph treats an LLM inference **session** as the durable object.

Today, the lifetime of an inference session is usually tied too closely to the inference worker that owns it. If the worker crashes, restarts, is upgraded, loses its GPU, or the workload needs to move elsewhere, the session may lose expensive computed state and need to start over.

Ralph separates:

1. **Durable logical state** — small, portable state needed to reconstruct the session.
2. **Acceleration state** — large, engine-specific state that makes resume fast, mainly KV cache and related runtime metadata.

The rule is:

> **Fast when possible, recoverable always.**

If native acceleration state can be restored safely, Ralph uses it. If not, Ralph reconstructs the session from durable logical state.

---

## 2. Core user promise

A session should survive:

- inference engine crashes
- engine restarts
- machine restarts
- GPU changes
- worker replacement
- engine upgrades
- preemption
- manual pause/resume
- local → remote handoff

The user should operate on a session, not on a specific vLLM process.

---

## 3. What Ralph is NOT

Ralph is deliberately narrow.

Do **not** turn it into:

- a new inference engine
- a GPU virtualization layer
- a Firecracker/Kata sandbox
- a Kubernetes replacement
- an autoscaler
- an agent orchestrator
- a profiler/benchmark dashboard
- a model hosting SaaS
- a custom KV database
- a distributed consensus system
- a general container platform
- a clone of Morph, Daytona, Modal, Dynamo, Cedana, or LMCache

Ralph may integrate with existing systems later, but it should not rebuild them.

For the first implementation:

- one Linux workstation first
- one NVIDIA GPU first
- vLLM first
- local filesystem first
- SSH for remote handoff later
- no cluster control plane
- no paid GPU/cloud/storage/database/service dependency
- every core Ralph feature must be developable and testable on one machine with one GPU

Primary development machine:

```text
OS: Arch Linux
CPU: Ryzen 7
GPU: NVIDIA RTX 5050
RAM: 16 GB
Storage: local NVMe
```

This hardware is the baseline development target, not a special case. Ralph must remain useful on a normal developer workstation.

Multi-GPU or multi-node hardware may be used for optional integration/performance validation, but must never be required to build, test, or use Ralph's core workflow.

---

# 4. Mental model

```text
                    RALPH SESSION
                         │
              ┌──────────┴──────────┐
              │                     │
       DURABLE STATE          ACCELERATION STATE
       portable               fast but disposable
              │                     │
       token history              KV cache
       model identity             block/runtime metadata
       model revision             engine-specific state
       sampling config
       generated tokens
       generation position
       adapter / LoRA identity
       seed / RNG metadata
              │                     │
              └──────────┬──────────┘
                         │
                    vLLM worker
                         │
                        GPU
```

The vLLM worker is disposable.

The session is not.

---

# 5. Core concepts

## 5.1 Session

A Ralph session represents one long-lived inference context.

Example:

```text
name: research
id: 01J...
model: Qwen/Qwen3-8B
engine: vllm
context_tokens: 62,144
state: active
location: local/gpu0
```

A session has:

- stable session ID
- human-readable name
- model and exact model revision
- tokenizer identity/revision
- input token history
- generated token history
- sampling configuration
- adapter / LoRA identity if present
- generation position
- timestamps
- engine compatibility metadata
- optional acceleration snapshot

---

## 5.2 Durable state

Durable state is the canonical source of truth.

It must be sufficient to reconstruct the session even when native KV state is unavailable.

Suggested contents:

```text
sessions/<session-id>/
├── manifest.json
├── tokens.bin
├── generation.log
├── sampler.json
├── compatibility.json
└── accel/
```

`manifest.json` contains:

```text
session id
session name
model id
model revision
tokenizer id/revision
engine
engine version
created_at
updated_at
state
last known location
token count
adapter metadata
```

Do not store model weights inside every session.

Model weights are shared/immutable and referenced by identity.

---

## 5.3 Acceleration state

Acceleration state exists only to avoid recomputing work.

Initially this means:

- KV cache
- engine-specific cache metadata
- any information required for a safe native restore

Acceleration state is allowed to be:

- absent
- evicted
- incompatible
- regenerated

Losing acceleration state must **never** make the logical session unrecoverable.

---

## 5.4 Compatibility fingerprint

Every acceleration checkpoint gets a fingerprint.

At minimum:

```text
model revision
tokenizer revision
engine
engine version
KV format/version
KV dtype
GPU architecture / relevant compatibility info
tensor parallel configuration
adapter identity
```

On resume:

```text
compatible
    -> fast restore

not compatible
    -> portable reconstruction
```

Never blindly load native KV state.

---

# 6. Primary features

## 6.1 Run

Start a model under Ralph.

```bash
ralph run Qwen/Qwen3-8B --name research
```

Expected result:

```text
✓ session research created
✓ vLLM started
✓ ready on gpu0
```

Ralph owns the session identity even though vLLM performs inference.

---

## 6.2 Query

```bash
ralph query research "explain tensor parallelism"
```

Requirements:

- streaming output
- all accepted input tokens are persisted
- generated output is durably recorded in batches
- the session metadata is updated
- a crash must not silently corrupt session history

---

## 6.3 List sessions

```bash
ralph ps
```

Example:

```text
NAME        MODEL        TOKENS    STATE      LOCATION
research    qwen3-8b     62k       active     gpu0
coding      qwen3-8b     91k       paused     disk
```

Keep the output short.

---

## 6.4 Inspect

```bash
ralph inspect research
```

Shows useful lifecycle information only:

```text
model
engine
token count
session state
current location
native checkpoint present?
native checkpoint compatible?
last durable flush
recovery path
```

Do not turn this into a general profiler.

---

## 6.5 Recover

This is Ralph's first killer feature.

```bash
ralph recover research
```

If the vLLM worker dies:

1. detect that the worker is gone
2. read durable session state
3. check for compatible acceleration state
4. if compatible, restore it
5. otherwise restart vLLM and rebuild state by prefill
6. continue the session

Recovery should have two paths:

### Fast recovery

```text
durable state
+ compatible KV
→ restore
→ continue
```

### Portable recovery

```text
durable token history
→ restart engine
→ re-prefill tokens
→ rebuild KV
→ continue
```

The portable path is slower but must work.

---

## 6.6 Pause / resume

```bash
ralph pause research
ralph resume research
```

Pause means:

- stop active generation
- flush durable state
- persist acceleration state if supported
- free GPU resources when possible
- keep the Ralph session

Resume means:

- choose the fastest safe restoration path
- reattach the session to a running worker
- continue

---

## 6.7 Hibernate

Hibernate is a stronger pause.

```bash
ralph hibernate research
```

Goal:

- release expensive GPU state
- keep the session alive
- move recoverable state to cheaper storage

Initial storage hierarchy:

```text
GPU → host RAM → local NVMe
```

Object storage is optional later.

Do not build a complicated global scheduler.

A simple policy is enough:

```text
active        -> GPU
recently idle -> RAM
long idle     -> NVMe
```

---

## 6.8 Export / import

Ralph sessions should become portable artifacts.

```bash
ralph export research
```

Produces:

```text
research.ralph
```

The artifact contains:

- manifest
- logical token history
- sampler state
- generation log
- compatibility metadata
- optional native acceleration checkpoint

It does **not** contain duplicate model weights by default.

Import:

```bash
ralph import research.ralph
```

If native acceleration state is incompatible, Ralph imports the logical session and rebuilds it when resumed.

---

## 6.9 Handoff / move

Later:

```bash
ralph handoff research user@gpu-box
```

Goal:

```text
machine A
   ↓
flush durable session
   ↓
transfer Ralph session
   ↓
machine B
   ↓
fast restore if compatible
or
portable rebuild
```

V1 remote transport should be SSH.

No custom distributed protocol.

No cluster controller.

---

## 6.10 Drain

```bash
ralph drain gpu0
```

Drain means:

- stop assigning new Ralph sessions to the target GPU
- safely pause/checkpoint sessions currently using it
- move/hibernate them if a destination is available
- return only when the GPU is safe to stop

Do not implement a Kubernetes-like scheduler.

The command can initially work with explicitly configured destinations.

---

## 6.11 Graceful preemption

Ralph should support termination hooks.

Example:

```bash
ralph serve --checkpoint-on SIGTERM
```

On termination:

1. stop accepting new generation
2. flush token/generation log
3. save acceleration state if time permits
4. persist metadata
5. exit

On the next machine:

```bash
ralph recover
```

This enables preemption-safe inference without building a cloud platform.

---

## 6.12 Engine restart / upgrade continuity

Later:

```bash
ralph restart-engine
```

or:

```bash
ralph upgrade-engine
```

Flow:

```text
persist sessions
↓
stop old engine
↓
start new engine
↓
check native-state compatibility
↓
restore KV if compatible
otherwise rebuild
↓
resume sessions
```

A failed upgrade must not destroy the logical sessions.

Automatic package management is not required.

---

# 7. CLI surface and UX contract

Ralph is infrastructure, but it must feel like a small developer tool rather than an ops console.

The CLI is part of the product. A technically correct feature with confusing UX is not considered finished.

## 7.1 Command surface

Keep the command surface tiny.

Required final commands:

```bash
ralph run
ralph query
ralph ps
ralph inspect

ralph pause
ralph resume
ralph hibernate

ralph checkpoint
ralph recover

ralph export
ralph import

ralph handoff
ralph drain
```

Support command:

```bash
ralph doctor
```

`ralph doctor` is allowed because it reduces setup/debugging friction. It should check only Ralph dependencies and environment readiness; it must not become a general system profiler.

Possible internal/admin command:

```bash
ralph daemon
```

Do not expose a huge command tree.

---

## 7.2 UX principles

### Principle 1 — sane defaults

The common path should require the fewest flags possible.

Good:

```bash
ralph run Qwen/Qwen3-8B --name research
ralph query research "hello"
ralph pause research
ralph resume research
```

Avoid requiring users to specify internal ports, worker IDs, cache directories, process IDs, or engine implementation details.

### Principle 2 — Ralph speaks in sessions

Never make the user operate on:

```text
PID 38219
worker-7fc1
cache-block-18
unix socket paths
```

unless they explicitly request verbose/debug output.

User-facing identity is:

```text
research
```

Internal IDs still exist for correctness.

### Principle 3 — concise success, actionable failure

Normal success output should be 1–4 lines.

Example:

```text
✓ research paused
  4.2 GB GPU memory released
```

An error should answer:

```text
what failed?
what state is the session in now?
did Ralph preserve the session?
what can the user do next?
```

Example:

```text
× fast restore unavailable
  saved KV: vLLM 0.12 / sm_90
  current:  vLLM 0.13 / sm_120

  session is safe. Ralph can rebuild it from 62,144 tokens.

→ run: ralph resume research --portable
```

Do not dump raw Rust errors, Python tracebacks, SQLite errors, or vLLM stack traces by default.

### Principle 4 — never make color required

Color and symbols are optional presentation enhancements.

Every state must remain understandable:

- without color
- when redirected to a file
- in CI
- through SSH
- in a terminal with Unicode disabled

### Principle 5 — no surprise destruction

Ralph must never silently delete the only durable copy of a session.

Operations that replace or overwrite user-owned artifacts must:

- refuse by default
- explain the conflict
- accept explicit `--force` when safe

### Principle 6 — automatic fallback should feel boring

Users should not normally choose between native KV restore and portable reconstruction.

Default:

```bash
ralph resume research
```

Ralph automatically chooses:

```text
native restore if safe
otherwise portable reconstruction
```

Advanced users may force a path:

```bash
ralph resume research --fast-only
ralph resume research --portable
```

`--fast-only` must fail safely rather than silently rebuilding.

### Principle 7 — progress only when useful

For operations longer than roughly one second, show a progress indicator when attached to a TTY.

Examples:

```text
rehydrating research  41,882 / 62,144 tokens
```

or:

```text
transferring research  1.8 / 2.4 GB
```

Rules:

- do not fake percentages when the total is unknown
- use elapsed time/activity instead
- no animated spinner when stdout is not a TTY
- never bury an error behind a spinner

### Principle 8 — Ctrl-C must be safe

For `ralph query`:

- first Ctrl-C requests generation cancellation
- already accepted/generated durable state remains valid
- the Ralph session stays usable
- CLI exits only after cancellation is acknowledged or a short timeout expires

For checkpoint/export/handoff/recovery:

- Ctrl-C stops the current attempt
- source session must remain recoverable
- partial destination artifacts are marked incomplete and are never treated as valid

### Principle 9 — idempotent lifecycle commands

Where safe, lifecycle commands should be idempotent.

Examples:

```text
pause(paused)       -> success/no-op
hibernate(hibernated) -> success/no-op
recover(active)     -> success/no-op with explanation
resume(active)      -> success/no-op
```

Do not turn harmless repeated commands into scary failures.

### Principle 10 — every long operation exposes the recovery state

If handoff/checkpoint/recovery fails halfway, Ralph must say where the valid session still exists.

Example:

```text
× handoff interrupted
  destination did not acknowledge import
  research remains active and recoverable on this machine
```

---

## 7.3 First-run experience

Running any normal command should automatically connect to the local daemon.

If the daemon is not running, Ralph should start it automatically when possible.

The user should not normally need:

```bash
ralph daemon start
```

If prerequisites are missing, show a short fix path.

Example:

```text
× no compatible NVIDIA GPU found

Ralph found:
  NVIDIA driver: 590.XX
  CUDA runtime:   available
  GPU:            none visible to this process

→ check CUDA_VISIBLE_DEVICES or run: ralph doctor
```

`ralph doctor` should check:

```text
Ralph data directory
SQLite access
free disk space
vLLM availability
Python/runtime required by adapter
NVIDIA driver
visible GPUs
model-cache access
SSH client when remote features are used
```

It should report pass/warn/fail, not mutate the system.

---

## 7.4 Session naming UX

`--name` is recommended but optional.

If omitted, Ralph generates a readable unique name and prints it clearly.

Example:

```text
✓ session created: quiet-otter
```

Rules:

- names must be unique locally
- reject ambiguous duplicate names
- allow exact session ID as an alternative
- allow a unique ID prefix for advanced recovery/debug usage
- never silently select between two ambiguous sessions

If a name is already used:

```text
× session "research" already exists

→ ralph inspect research
→ choose another name with --name
```

---

## 7.5 Output modes

Global flags:

```bash
--json
--quiet
--no-color
-v / --verbose
--yes
```

Rules:

### Human mode

Default. Compact, readable, stable wording where practical.

### JSON mode

For scripts/CI.

Requirements:

- machine-readable JSON only on stdout
- diagnostics on stderr
- stable field names within a major version
- no spinner/progress animation
- non-zero exit status on failure

### Quiet mode

Print only essential result data.

### Verbose mode

May expose:

- worker IDs
- paths
- engine details
- compatibility fingerprint
- internal recovery decisions

Verbose mode must still avoid printing prompt content by default.

---

## 7.6 Exit codes

Keep a small documented set.

Suggested:

```text
0   success / safe no-op
1   generic operation failure
2   invalid CLI usage
3   session not found / ambiguous session
4   invalid session state for operation
5   compatibility failure when fallback was forbidden
6   resource failure (GPU memory/disk/etc.)
7   remote/transport failure
8   persistent-state corruption detected
```

Do not create dozens of codes.

---

## 7.7 Helpful command behavior

### `ralph run`

Should:

- resolve exact model revision before marking session ready
- verify enough basic disk space before model download/checkpoint setup
- fail early when no usable GPU is available
- avoid leaving a fake ACTIVE session when vLLM never became healthy
- print the created session name

### `ralph query`

Should:

- stream by default
- support stdin for scripting/large prompts
- preserve the session if the terminal disconnects
- distinguish generation cancellation from worker failure
- never duplicate committed generated tokens after recovery

### `ralph ps`

Default columns only:

```text
NAME  MODEL  TOKENS  STATE  LOCATION
```

Optional verbose columns only with `-v`.

If there are no sessions:

```text
no sessions yet

→ ralph run <model> --name <name>
```

### `ralph inspect`

Focus on lifecycle/recoverability, not performance trivia.

Must clearly show:

```text
recoverability: safe / degraded / broken
fast restore:   available / unavailable / incompatible
portable state: complete / incomplete
```

### `ralph pause`, `resume`, `hibernate`, `recover`

Must show the chosen path only when useful.

Good:

```text
✓ research resumed from KV checkpoint in 1.8s
```

or:

```text
KV checkpoint incompatible; rebuilding from 62,144 tokens...
✓ research resumed in 9.4s
```

### `ralph export`

Default output:

```text
./research.ralph
```

Refuse overwrite unless `--force` is provided.

### `ralph import`

Validate the full artifact before making it visible as a valid session.

If the session name conflicts:

```text
× session "research" already exists

→ import with: --name research-copy
```

### `ralph handoff`

Source remains authoritative until the destination:

1. receives data
2. validates data
3. creates/imports session
4. acknowledges readiness

Only then may Ralph pause/remove source acceleration state depending on requested mode.

### `ralph drain`

Print a compact plan before executing when more than one session is affected.

Example:

```text
draining gpu0

research   -> gpu1
coding     -> hibernate
chat-7     -> finish

3 sessions
```

For a destructive or irreversible variant, require explicit confirmation or `--yes`.

---

## 7.8 UX anti-patterns

Do not ship CLI output like:

```text
Error: anyhow::Error(SqliteFailure(Error { code: DatabaseBusy ... }))
```

Do not require:

```bash
ralph session lifecycle state transition --session-id ...
```

Do not constantly print:

```text
INFO daemon...
INFO worker...
INFO storage...
```

Normal Ralph usage should feel closer to Git/Docker than to reading service logs.

---

# 8. Session states

Use an explicit state machine.

```text
CREATED
  ↓
STARTING
  ↓
ACTIVE
  ↓
PAUSING
  ↓
PAUSED
  ↓
RESUMING
  ↓
ACTIVE
```

Other states:

```text
HIBERNATED
RECOVERING
MOVING
FAILED
STOPPED
```

Important rule:

A failed worker must not automatically mark the session itself as failed.

Example:

```text
worker = dead
session = recoverable
```

---

# 9. Recovery decision tree

When Ralph resumes/recover a session:

```text
1. Is the logical session valid?
   no  -> fail clearly
   yes -> continue

2. Is compatible native acceleration state available?
   yes -> restore native state
   no  -> continue

3. Is token history complete?
   yes -> reconstruct via prefill
   no  -> fail clearly because durable state is incomplete
```

This decision tree should remain simple.

---

# 10. Architecture

## 10.1 High-level

```text
                    ┌──────────────┐
                    │  Ralph CLI   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Ralph daemon │
                    └──────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
     ┌───────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
     │ Session Mgr  │ │ State Mgr│ │ Engine      │
     │              │ │          │ │ Adapter     │
     └───────┬──────┘ └────┬─────┘ └──────┬──────┘
             │             │               │
             │      ┌──────▼──────┐        │
             │      │ Storage     │        │
             │      │             │        │
             │      │ metadata    │        │
             │      │ token log   │        │
             │      │ accel state │        │
             │      └─────────────┘        │
             │                             │
             └──────────────────────┬──────┘
                                    │
                              ┌─────▼─────┐
                              │   vLLM    │
                              └─────┬─────┘
                                    │
                                  GPU
```

---

## 10.2 Components

### CLI

Responsibilities:

- parse commands
- talk to local Ralph daemon
- stream user-visible output
- never own session state directly

### Ralph daemon

Responsibilities:

- lifecycle management
- worker monitoring
- session state transitions
- recovery orchestration
- storage coordination
- engine adapter coordination

One daemon per machine is enough.

### Session manager

Responsibilities:

- session IDs/names
- lifecycle state machine
- active worker mapping
- recovery status
- destination mapping for handoff

### State manager

Responsibilities:

- durable token/generation persistence
- checkpoint creation
- compatibility checks
- portable reconstruction
- acceleration state lifecycle

### Engine adapter

Stable internal interface such as:

```text
start_model()
stop_model()
create_session()
generate()
pause()
resume()
checkpoint_acceleration_state()
restore_acceleration_state()
health()
```

First adapter:

```text
vLLM
```

Do not design a giant plugin system before a second engine is actually supported.

### Storage

Use simple local storage first.

Recommended:

- SQLite for metadata/state transitions
- append-only files for generation/token logs
- filesystem directories for checkpoint blobs

SQLite should use WAL mode.

Do not put huge KV blobs inside SQLite.

---

# 11. Implementation language / repo shape

Ralph should stay small.

Recommended:

```text
ralph/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── cli.rs
│   ├── daemon.rs
│   ├── session.rs
│   ├── state.rs
│   ├── storage.rs
│   ├── recovery.rs
│   ├── compatibility.rs
│   └── engine/
│       ├── mod.rs
│       └── vllm.rs
├── tests/
├── examples/
└── docs/
```

Prefer one Rust crate initially.

Do not split Ralph into many crates until there is a real need.

A tiny Python helper/bridge is acceptable only if vLLM exposes required functionality more safely/easily through Python.

The user-facing CLI and lifecycle logic should remain in Rust.

---

# 12. Data safety rules

Ralph handles potentially private prompts and generations.

Minimum requirements:

- session directories mode `0700`
- session files mode `0600`
- never print full prompt history in logs by default
- never include prompt text in error messages unless explicitly requested
- atomic metadata updates
- checksums for exported session artifacts
- version every persistent format
- incomplete writes must be detectable

Do not add encryption/key management in early phases unless needed.

---

# 13. Important invariants

These must always hold.

### Invariant 1

The durable logical session is the source of truth.

### Invariant 2

Acceleration state may be deleted at any time without making a valid logical session permanently unusable.

### Invariant 3

Ralph must never load acceleration state whose compatibility fingerprint fails.

### Invariant 4

Accepted generated tokens must eventually be durably logged.

### Invariant 5

A worker crash is not the same as a session failure.

### Invariant 6

Every state transition must either complete or be recoverable after daemon restart.

### Invariant 7

Do not silently discard session state.

---

# 14. Testing strategy

Every phase must have:

- unit tests for state transitions
- failure-path tests
- at least one real vLLM end-to-end test
- no fake "success" path for the main feature
- crash/restart testing where relevant
- at least one interruption test for every long-running operation added in that phase
- TTY and non-TTY CLI behavior tests where the phase changes user-visible output
- JSON output tests for commands added in that phase
- verification that failure leaves the session in a documented recoverable or terminal state

## 14.1 Local-first development rule

The complete core Ralph development workflow must work on the primary development machine:

```text
Ryzen 7
RTX 5050
16 GB RAM
Arch Linux
local NVMe
```

No paid service is allowed to be a requirement for development, testing, demos, or normal use.

The local machine must be enough to build and validate:

```text
serialization                ✓
session ownership            ✓
transfer protocol            ✓
compatibility checks         ✓
crash recovery               ✓
pause/resume                 ✓
hibernation                  ✓
RAM/NVMe tiering             ✓
engine restart               ✓
export/import                ✓
handoff semantics            ✓
preemption handling          ✓
drain semantics              ✓
```

Use small real models during development. Ralph tests the lifecycle mechanism, not model intelligence.

Normal development should prefer models small enough to:

- load comfortably on the RTX 5050
- leave headroom for KV/cache experiments
- start quickly enough for repeated crash/recovery tests
- support real vLLM inference

Large production models are not required to prove Ralph's correctness.

`ralph doctor` must clearly report unsupported CUDA/vLLM combinations rather than failing later with raw backend errors.

## 14.2 Main single-GPU end-to-end scenario

```text
start Ralph
↓
start model
↓
create long session
↓
generate output
↓
kill vLLM with SIGKILL
↓
recover session
↓
continue generation
```

Later:

```text
checkpoint
↓
delete acceleration state
↓
recover through portable path
```

and:

```text
export
↓
delete/stop original local session
↓
import
↓
resume
```

These are mandatory local tests.

## 14.3 Local handoff simulation

Real handoff semantics must be testable without owning two GPUs or two computers.

Run two isolated Ralph instances on the same machine:

```text
Ralph A
data root: ~/.local/share/ralph-a
endpoint: 127.0.0.1:<port-a>

        real transfer path
        TCP/SSH/loopback

Ralph B
data root: ~/.local/share/ralph-b
endpoint: 127.0.0.1:<port-b>
```

The instances must have independent:

- metadata databases
- session roots
- daemon identities
- ownership records
- transfer staging areas

On the single RTX 5050, GPU ownership can be sequential:

```text
A owns session on GPU
↓
A freezes + persists
↓
A transfers
↓
A releases GPU
↓
B acquires GPU
↓
B restores/rebuilds
```

This is a valid handoff correctness test.

It must exercise the real production handoff protocol, not a special fake test path.

Use it to validate:

```text
ownership transfer
checksums
two-phase/commit-style ownership handoff
rollback
interrupted transfer
destination rejection
source authority
split-brain prevention
Ctrl-C behavior
```

## 14.4 Optional free multi-GPU integration target

When a real two-GPU test is useful, the preferred optional free environment is a Kaggle notebook/runtime exposing **2× NVIDIA T4 GPUs**, when that free accelerator remains available.

This environment is for integration testing only.

Example topology:

```text
GPU 0 / T4
  ↓
worker A
  ↓
Ralph session

        move / restore

GPU 1 / T4
  ↓
worker B
```

Use it to validate:

```text
GPU0 → GPU1 correctness
two workers alive concurrently
migration downtime
checkpoint/restore latency
KV transfer/restore behavior
multi-GPU race conditions
fault injection during migration
```

Important rules:

- Kaggle is optional, never required.
- Ralph must not contain Kaggle-specific product logic.
- Tests requiring two GPUs must detect hardware capability and skip cleanly when only one GPU is present.
- CI must not require a paid GPU runner.
- Free-tier availability/limits may change; Ralph's architecture must not depend on them.
- T4 measurements are not representative of H100/H200/B200-class datacenter performance.

## 14.5 Multi-node testing without paid infrastructure

True network behavior can first be tested with the two local Ralph endpoints above because the real transfer protocol still crosses sockets/SSH.

For a genuine physical two-machine test, an optional zero-cost setup is:

```text
developer RTX 5050 machine
        ↓ SSH/network
another Linux PC with an NVIDIA GPU
```

This can be the user's second machine, a contributor's machine, or any temporarily available compatible host.

No feature may require renting a cloud GPU to pass its core acceptance criteria.

## 14.6 What cannot be fully validated locally

A single RTX 5050 cannot produce trustworthy measurements for:

```text
real GPU A → GPU B performance
simultaneous multi-GPU saturation
NVLink/NVSwitch behavior
datacenter interconnect behavior
large-cluster concurrency
H100/H200/B200 production throughput
```

Those are advanced benchmark/integration environments, not prerequisites for implementing Ralph.

Ralph must separate:

```text
correctness claim
from
hardware-specific performance claim
```

Never extrapolate workstation or T4 results into unsupported datacenter claims.

## 14.7 Fault-injection philosophy

For any move/recovery/storage operation, deliberately inject failures at meaningful boundaries:

```text
before durable commit
after durable commit
during acceleration-state save
during transfer
after destination validation
before ownership commit
after ownership commit
during worker startup
during restore
```

The expected result must always be one of:

```text
source remains authoritative
destination becomes authoritative
session is explicitly terminal because durable state is corrupt
```

Never allow ambiguous dual ownership.

# 15. Performance metrics

Ralph is not a profiler product, but we need internal measurements.

Track:

```text
time to recover with native state
time to recover through replay/prefill
checkpoint size
logical session size
KV checkpoint size
pause latency
resume latency
handoff transfer bytes
downtime during recovery/move
```

Benchmarks are for validating Ralph, not a user-facing analytics suite.

Record the hardware/environment beside every benchmark result.

At minimum:

```text
GPU model
GPU count
driver/CUDA version
engine + version
model + revision
context length
KV dtype
storage medium
local vs loopback vs physical network
```

Results from the RTX 5050 or optional free T4x2 environments are valid for Ralph development/integration comparisons, but must not be presented as representative datacenter-GPU performance.

---

# 16. Edge cases and failure handling

This section defines expected handling for all known edge cases inside Ralph's current scope.

The general rule is:

> **When Ralph is uncertain, preserve durable state, refuse unsafe acceleration state, and tell the user exactly what remains recoverable.**

No failure should silently convert a recoverable session into data loss.

---

## 16.1 Session identity and lifecycle edge cases

### Duplicate session name

Handling:

- reject creation/import
- do not rename silently
- suggest `--name`

### Unknown session

Handling:

- fail with exit code 3
- show close exact-prefix matches only when useful
- never fuzzy-select and execute a state-changing operation

### Ambiguous session prefix

Handling:

- refuse
- print matching names/IDs

### Command conflicts with current state

Examples:

```text
resume(ACTIVE)
pause(PAUSED)
hibernate(HIBERNATED)
recover(ACTIVE)
```

Handling:

- safe repeated lifecycle operations are no-op success
- impossible transitions fail without mutating state

### Operation already in progress

Examples:

```text
resume while MOVING
handoff while CHECKPOINTING
pause while RECOVERING
```

Handling:

- acquire per-session operation lock
- second mutating command fails fast with a useful message
- read-only `ps`/`inspect` remain available
- never run two state transitions concurrently on the same session

### Stale operation lock after daemon crash

Handling:

- locks must include daemon/process identity plus operation journal state
- on restart, reconcile against durable transition journal
- never permanently strand a session because a PID lock file exists

---

## 16.2 Query/generation edge cases

### User presses Ctrl-C during generation

Handling:

- request engine cancellation
- flush all tokens already considered committed
- session remains valid
- no duplicated tokens on next query

### Client terminal disconnects

Handling:

- CLI connection loss must not corrupt the session
- configured behavior should default to cancelling generation after detecting disconnect, while preserving committed output
- later versions may allow detached generation explicitly; do not make it implicit

### Worker dies before first token

Handling:

- input is already durable before execution begins
- mark attempt interrupted
- recover/retry safely

### Worker dies after some tokens were streamed

Hard case.

Handling:

- distinguish `streamed` from `durably committed`
- persist generated output frequently enough to bound replay ambiguity
- after recovery, never emit duplicate committed tokens as if they were new
- if exact continuation cannot be guaranteed because sampler/RNG state was lost, report that generation resumes from the last committed token boundary

### Request exceeds model context window

Handling:

- fail before modifying canonical session state when detectable
- report current tokens, requested tokens, and model limit
- do not auto-truncate conversation history silently

### Invalid sampling parameters

Handling:

- validate before recording the request as accepted
- report the invalid field directly

### Adapter/LoRA disappears between queries

Handling:

- session becomes degraded, not destroyed
- refuse to continue under a different adapter silently
- explain the missing adapter identity/path

---

## 16.3 Durable-state persistence edge cases

### Process crashes during metadata write

Handling:

- use transactional SQLite updates
- file writes use temp file + fsync + atomic rename where relevant
- transition journal reconciles on daemon restart

### Power loss during token-log append

Handling:

- records are framed/versioned/checksummed or otherwise able to detect a torn tail
- discard only incomplete tail data
- preserve all previously committed records

### SQLite busy/locked

Handling:

- use WAL mode
- bounded retry with jitter/backoff
- never expose raw `database is locked` as the only user message
- if retries fail, leave session state unchanged and report temporary storage failure

### Metadata says ACTIVE but worker is gone after reboot

Handling:

- reconcile during startup
- set session to recoverable/interrupted state
- do not pretend it is ACTIVE

### Durable token log missing

Handling:

- if native checkpoint exists but canonical durable state is incomplete, do not claim the session is safely portable
- `inspect` reports `recoverability: degraded` or `broken`
- never delete the remaining evidence automatically

### Corrupt durable state

Handling:

- detect checksum/format/transaction inconsistency
- fail closed
- preserve files for manual recovery
- never overwrite corrupt data while attempting automatic repair

---

## 16.4 Checkpoint / acceleration-state edge cases

### Checkpoint interrupted halfway

Handling:

- write into an incomplete/staging location
- mark valid only after final metadata/checksum commit
- ignore incomplete checkpoint on resume
- durable logical session remains canonical

### Checkpoint exists but fingerprint is incompatible

Handling:

- never load it
- default to portable reconstruction
- explain the mismatch in `-v` or `inspect`

### Checkpoint file corrupt

Handling:

- quarantine/ignore acceleration state
- portable reconstruction if logical state is complete
- no session loss

### KV save fails because disk is full

Handling:

- do not fail the logical session
- surface that only fast recovery was lost
- preserve durable log first
- hibernate may fail if it cannot release GPU safely without a recoverable path

### KV checkpoint larger than available disk

Handling:

- estimate when possible before starting
- fail early or fall back to logical-only checkpoint
- tell user what capability is lost

### Native backend does not support KV serialization

Handling:

- Ralph still supports portable recovery
- feature capability reported clearly
- no fake `checkpoint succeeded` message implying fast restore exists

---

## 16.5 GPU / engine edge cases

### No GPU visible

Handling:

- fail early for commands that require execution
- existing durable sessions remain inspectable/exportable

### GPU OOM while starting model

Handling:

- session remains CREATED/STOPPED or recoverable, not ACTIVE
- terminate failed worker cleanly
- report model, requested GPU, free/total memory when safely obtainable
- suggest hibernating other Ralph sessions when relevant

### GPU OOM during generation

Handling:

- classify worker/session state accurately
- persist durable boundary
- attempt worker recovery only when policy says safe
- never loop forever automatically

### GPU disappears / driver reset

Handling:

- treat worker as failed
- invalidate acceleration state whose correctness is uncertain
- recover using another available path when requested/possible

### vLLM fails to become healthy

Handling:

- bounded startup timeout
- collect a short diagnostic summary
- stop failed child process
- do not mark session ACTIVE

### vLLM crashes repeatedly

Handling:

- bounded automatic restart attempts
- exponential-ish backoff
- after threshold, stop retrying and report FAILED worker with session still durable
- avoid infinite crash loops

### Engine API response times out

Handling:

- distinguish timeout from confirmed failure
- health-check worker before deciding it is dead
- avoid launching duplicate workers for the same session accidentally

### Engine version changes unexpectedly

Handling:

- recompute compatibility fingerprint
- acceleration state may become invalid
- logical session remains valid

---

## 16.6 Model/tokenizer compatibility edge cases

### Model revision moved or tag changed

Handling:

- store immutable revision/digest when possible
- never silently resume against a different model revision

### Model missing locally on restore

Handling:

- if normal model resolution/download is configured, obtain exact revision
- otherwise report exact model identity required
- session remains valid while unavailable

### Model download interrupted

Handling:

- use underlying cache's safe partial-download behavior
- never mark model ready prematurely

### Tokenizer revision mismatch

Handling:

- treat as incompatible with native/logical replay assumptions where relevant
- obtain exact tokenizer revision or fail clearly
- do not silently retokenize historical text into a different token sequence

### Model context limit changed after upgrade

Handling:

- if existing token history no longer fits, refuse resume under that runtime configuration
- explain the conflict
- do not truncate automatically

### Sampling implementation changed

Handling:

- exact byte-for-byte future output may not be guaranteed after portable reconstruction
- preserve historical generated tokens
- document resume boundary and deterministic limitations

---

## 16.7 Pause / resume / hibernate edge cases

### Pause while generation is active

Handling:

- stop accepting new query work for the session
- cancel or reach a safe generation boundary
- flush durable state
- only then transition to PAUSED

### Resume while GPU capacity is insufficient

Handling:

- leave session PAUSED/HIBERNATED
- do not destroy its state
- report resource shortage

### Hibernation cannot save acceleration state

Handling:

- if logical state is complete, allow logical-only hibernation when policy permits
- explain that resume will require prefill

### RAM tier full

Handling:

- fall through to NVMe if configured and safe
- otherwise remain on current tier
- no arbitrary eviction of canonical state

### NVMe tier full

Handling:

- evict disposable acceleration state according to policy
- never evict the only canonical durable state

### User resumes while automatic hibernation is happening

Handling:

- serialize through per-session transition lock
- either cancel hibernation safely or finish it then resume
- never race both state transitions

---

## 16.8 Export / import edge cases

### Output file already exists

Handling:

- refuse by default
- require `--force`

### Export interrupted

Handling:

- write temporary artifact
- rename only after complete validation/checksum
- clean stale temp artifacts opportunistically

### Import corrupt artifact

Handling:

- validate before adding session to canonical DB
- fail without partial visible session

### Import from newer unsupported Ralph format

Handling:

- fail clearly with artifact version and supported range
- never guess interpretation

### Import from older supported format

Handling:

- explicit migration path
- keep original artifact untouched

### Imported session name collides

Handling:

- require `--name` or explicit overwrite behavior if ever supported
- never overwrite an existing session silently

### Artifact references unavailable model/adapter

Handling:

- import logical state successfully if structurally valid
- mark execution dependency missing
- `inspect` explains what is required before resume

### Artifact contains malicious paths

Handling:

- never trust archive paths
- reject path traversal, symlinks/links that escape extraction root, and unexpected file types
- enforce size limits before extraction when possible

---

## 16.9 Remote handoff edge cases

### SSH authentication fails

Handling:

- source unchanged
- actionable transport error

### Destination Ralph missing or incompatible

Handling:

- capability/version handshake before transferring large blobs
- source unchanged

### Destination lacks exact model

Handling:

- destination can resolve exact model revision if configured
- otherwise abort before source relinquishes authority

### Destination lacks disk space

Handling:

- preflight when possible
- fail before transfer or before commit
- source unchanged

### Network drops midway

Handling:

- partial destination is invalid/staged
- source remains canonical
- retries may resume transfer only if integrity scheme supports it safely

### Destination validates data but fails to start engine

Handling:

- handoff is not complete
- source remains recoverable
- optionally leave imported-but-paused session at destination if clearly reported

### Both source and destination could become active

Split-brain risk.

Handling:

- handoff uses an explicit ownership/commit protocol
- source remains owner until destination ACK
- after committed move, source must not continue generation without a new explicit action
- do not add distributed consensus; SSH handoff is a controlled two-endpoint transaction

### User Ctrl-C during handoff

Handling:

- preserve source authority
- destination staging may be cleaned later

---

## 16.10 Drain edge cases

### One session cannot move/hibernate

Handling:

- drain does not claim success
- clearly list blocked sessions
- unaffected sessions may complete safely

### New query arrives during drain

Handling:

- target GPU stops accepting new session placements immediately
- behavior for existing sessions is explicit

### Drain destination becomes unavailable

Handling:

- preserve affected source session
- continue/abort according to safe plan
- final result lists incomplete moves

### User interrupts drain

Handling:

- sessions already safely moved stay moved
- remaining sessions stay valid
- target is not marked drained unless it actually is

---

## 16.11 Daemon / concurrency edge cases

### Ralph daemon crashes

Handling:

- running vLLM worker may continue temporarily
- on daemon restart, discover/reconcile known worker where possible
- transition journal resolves incomplete operations

### Two CLI processes mutate same session

Handling:

- daemon serializes mutation
- one wins lock; other gets deterministic busy/conflict response

### Two Ralph daemons accidentally start

Handling:

- single-instance lock/socket ownership
- second daemon exits safely
- stale lock is recoverable

### System clock changes

Handling:

- do not base correctness on wall-clock ordering
- use monotonic clocks for durations/timeouts
- wall clock only for human timestamps

### Machine reboot

Handling:

- durable DB/logs survive
- sessions previously ACTIVE reconcile to interrupted/recoverable
- Ralph does not auto-start expensive model workloads unless explicitly configured later

---

## 16.12 Storage pressure and cleanup edge cases

### Ralph data disk almost full

Handling:

- warn before risky operation
- prioritize durable logical writes over acceleration cache
- prevent new large checkpoints when safety margin is crossed

### Cleanup/GC runs while session is active

Handling:

- GC only removes objects proven unreachable/disposable
- use references/leases so active operations cannot lose files underneath them

### Old acceleration snapshots accumulate

Handling:

- retain latest valid snapshot plus any snapshot required by in-progress operation
- delete old acceleration snapshots according to bounded policy
- canonical logical state is excluded from automatic destructive GC

### Permission changes on data directory

Handling:

- fail safely
- do not recreate a separate hidden state store elsewhere

---

## 16.13 Security and privacy edge cases

### Prompt accidentally appears in logs

Handling:

- prohibited by default
- logs refer to session ID/name, token counts, byte sizes, and errors
- prompt logging requires an explicit debug-only opt-in if ever implemented

### Session artifact permissions

Handling:

- local session dirs `0700`
- private files `0600`
- exported files should be created private by default

### Remote host key changes

Handling:

- defer to SSH security semantics
- never auto-disable host-key checking to make handoff easier

### Untrusted imported artifact

Handling:

- parsing/extraction must be defensive
- no code execution from artifact metadata
- paths and sizes validated

---

## 16.14 UX edge cases

### Command used in non-interactive script

Handling:

- no prompts when stdin/stdout is non-TTY unless explicitly requested
- operation requiring confirmation fails with instruction to pass `--yes`

### Terminal width is tiny

Handling:

- tables degrade gracefully
- truncate nonessential display fields, never identity/state meaning
- `--json` remains exact

### Unicode unsupported

Handling:

- `--no-color` or environment capability can use ASCII equivalents

### Slow operation with unknown total

Handling:

- show phase + elapsed time, not invented percentage

### Verbose output requested

Handling:

- show technical details without leaking prompt/generation content

---

## 16.15 Required fault-injection tests

By the end of Phase 7, automated/manual tests must cover at least:

```text
kill vLLM during generation
kill vLLM during recovery
kill Ralph daemon during state transition
SIGKILL Ralph during checkpoint metadata commit
truncate token-log tail
corrupt KV checkpoint
remove KV checkpoint
fill checkpoint disk
force SQLite busy contention
GPU OOM on model start
GPU OOM during request where reproducible
resume with incompatible engine fingerprint
resume with missing model
resume with tokenizer mismatch
Ctrl-C query
Ctrl-C checkpoint
Ctrl-C handoff
network drop during handoff
SSH auth failure
import corrupt artifact
import path-traversal artifact
reboot/restart reconciliation
parallel pause/resume commands
stale operation lock
hibernate with RAM tier full
export target already exists
drain with one blocked session
```

A phase is not done merely because the happy path works.

---

# 17. Seven implementation phases

Do not start the next phase until the current phase's acceptance criteria pass.

---

## Phase 1 — Foundation: Ralph owns the session

### Goal

Build the smallest real usable shell around vLLM.

### Build

- Rust CLI
- local daemon
- session ID/name
- SQLite metadata
- session directories
- vLLM process lifecycle
- health checking
- `ralph run`
- `ralph query`
- `ralph ps`
- `ralph inspect`
- exact model/tokenizer revision stored in manifest
- streaming generation
- auto-start/connect local daemon
- human output + `--json` output for Phase 1 commands
- global `--quiet`, `--no-color`, and `-v/--verbose` behavior
- stable user-facing error envelope
- safe Ctrl-C behavior for `query`
- `ralph doctor` basic environment checks
- duplicate/ambiguous session-name handling
- single-daemon and per-session mutation locks

### Commands

```bash
ralph run Qwen/Qwen3-8B --name demo
ralph query demo "hello"
ralph ps
ralph inspect demo
```

### Acceptance criteria

- real vLLM process starts
- user can query it through Ralph
- restarting Ralph daemon does not lose session metadata
- process IDs are internal, not the user-facing identity
- no mocks in the main E2E demo
- CLI works correctly in TTY and piped/non-TTY mode
- `--json` emits JSON only on stdout
- no raw traceback/database error is shown in normal mode
- Ctrl-C during query leaves session usable
- duplicate names and concurrent mutations fail clearly and safely

---

## Phase 2 — Durable logical sessions + crash recovery

### Goal

Prove Ralph's core thesis.

### Build

- durable token history
- generated-token log
- batched fsync/flush policy
- worker crash detection
- recovery state
- engine restart
- portable reconstruction through replay/prefill
- `ralph recover`
- daemon restart recovery
- corrupted/incomplete log detection
- torn-tail recovery for append-only logs
- startup reconciliation when DB says ACTIVE but worker is gone
- bounded worker restart loop
- committed-vs-streamed token tracking
- context-window preflight
- safe handling of client disconnect and Ctrl-C

### Main demo

```bash
ralph run Qwen/Qwen3-8B --name demo
ralph query demo "<large prompt>"

kill -9 <vllm-pid>

ralph recover demo
ralph query demo "continue"
```

### Acceptance criteria

- kill vLLM with SIGKILL during/after a session
- Ralph detects worker loss
- session remains present
- new worker starts
- token context is reconstructed
- user can continue using the same Ralph session
- no native KV persistence is required yet
- worker crash before first token and after partial output are both tested
- daemon restart during/after recovery is tested
- incomplete/torn log never causes silent token duplication or truncation
- repeated worker crash does not create an infinite restart loop

This is the first public release-worthy milestone.

---

## Phase 3 — Fast checkpoint + pause/resume

### Goal

Avoid full reconstruction when compatible acceleration state can be preserved.

### Build

- acceleration checkpoint interface
- compatibility fingerprint
- native KV persistence using the simplest supported vLLM/LMCache path
- `ralph checkpoint`
- `ralph pause`
- `ralph resume`
- fast vs portable resume decision
- explicit fallback when native state is missing/incompatible
- checksum/version metadata
- staging + atomic commit for acceleration checkpoints
- disk-space preflight where practical
- interrupted-checkpoint cleanup
- safe handling of unsupported KV serialization
- explicit `--fast-only` and `--portable` modes

### Resume UX

```text
$ ralph resume demo

native checkpoint compatible
✓ restored session
```

or:

```text
native checkpoint unavailable
rehydrating from 61,442 tokens...
✓ restored session
```

### Acceptance criteria

- compatible checkpoint resumes without full replay
- deleting the KV checkpoint still allows portable resume
- incompatible fingerprint never loads native state
- session stays correct in both paths
- corrupt/incomplete KV checkpoint falls back safely
- disk-full during checkpoint preserves the logical session
- incompatible fingerprint is never loaded
- Ctrl-C during checkpoint leaves no valid-looking partial snapshot

---

## Phase 4 — Hibernation + local storage tiers

### Goal

Make idle sessions stop wasting GPU memory.

### Build

- session states: active / paused / hibernated
- host RAM tier if practical through backend support
- NVMe checkpoint tier
- manual hibernate/resume
- basic idle policy
- storage quotas
- eviction of disposable acceleration state
- keep durable logical state protected
- serialize resume-vs-hibernate races
- RAM-full fallback behavior
- NVMe-full behavior
- GC/reference safety for active operations

### Commands

```bash
ralph hibernate demo
ralph resume demo
```

### Simple policy

```text
active     -> GPU
short idle -> RAM
long idle  -> NVMe
```

Do not build a global scheduler.

### Acceptance criteria

- hibernating releases GPU resources
- hibernated session remains recoverable
- deleting acceleration state does not delete logical session
- session can return to active state successfully
- RAM tier full and NVMe tier full are tested
- hibernation failure never deletes the only recoverable state
- resume during hibernation resolves deterministically

---

## Phase 5 — Portable `.ralph` session artifacts

### Goal

Make sessions portable between installations.

### Build

- versioned `.ralph` format
- `ralph export`
- `ralph import`
- manifest validation
- checksum validation
- acceleration state optional
- model weights excluded by default
- compatibility inspection on import
- import with portable reconstruction when native state is incompatible
- temporary-file + atomic finalization for export
- archive path-traversal and unsafe-link defense
- artifact size/format validation
- session-name collision UX

### Commands

```bash
ralph export research
ralph import research.ralph
ralph resume research
```

### Acceptance criteria

- export/import on the same machine works
- artifact without KV state works
- imported artifact survives Ralph restart
- corrupted artifact fails clearly
- version mismatch fails clearly or follows an explicit migration path
- interrupted export leaves no valid-looking artifact
- corrupt import creates no partial visible session
- path-traversal artifact is rejected
- missing model/adapter produces an importable-but-not-runnable session with clear inspect output

---

## Phase 6 — Remote handoff, drain, and preemption

### Goal

Let sessions move when compute changes.

### Build

- SSH transport
- `ralph handoff`
- destination capability check
- transfer only required session data
- native state transfer when compatible
- logical-state fallback otherwise
- `ralph drain`
- SIGTERM checkpoint hook
- graceful preemption workflow
- explicit destination configuration
- safe failure/rollback if transfer breaks halfway
- capability/version handshake before large transfer
- destination disk/model preflight where possible
- source-ownership commit protocol to prevent split-brain
- network-drop and Ctrl-C handling
- two-local-endpoint test mode using separate Ralph data roots
- sequential single-GPU handoff path for development
- hardware capability detection for optional multi-GPU tests
- optional two-GPU integration test script that works on a free T4x2-style environment when available

### Commands

```bash
ralph handoff research user@gpu-box
ralph drain gpu0
```

### Acceptance criteria

- handoff protocol works between two isolated local Ralph endpoints with separate state roots
- the single-GPU test can transfer ownership A → B sequentially without requiring a second GPU
- machine A can export/handoff a session to machine B through SSH when a second machine is available
- machine B resumes the same logical session
- interrupted transfer leaves source recoverable
- `drain` finishes with target GPU safe to stop
- SIGTERM leaves session recoverable
- SSH auth failure leaves source unchanged
- network failure midway leaves source authoritative
- destination startup failure does not commit the move
- drain never claims success while blocked sessions remain
- multi-GPU tests skip cleanly on one-GPU machines
- optional free two-GPU validation can move a real session GPU0 → GPU1
- no acceptance criterion requires a paid cloud/GPU service

No Kubernetes. No Raft. No custom cluster manager. No paid infrastructure dependency.

---

## Phase 7 — Hardening + second engine boundary + release

### Goal

Turn Ralph from a demo into a credible open-source systems tool.

### Build

- robust daemon crash recovery
- engine restart/upgrade continuity
- rollback-safe state transitions
- benchmark suite
- concurrency/race testing
- disk-full handling
- corrupted checkpoint handling
- storage cleanup/GC
- better compatibility reporting
- structured logs
- stable persistent format
- documentation
- install flow
- example demos
- CI
- release binaries
- stable documented exit codes
- shell completion generation if trivial through the CLI framework
- complete TTY/non-TTY/JSON UX test suite
- fault-injection suite from Section 16.15
- first-run/install documentation and troubleshooting examples

Only after vLLM support is stable:

- formalize the engine adapter
- add one second engine, preferably SGLang
- prove portable logical sessions survive engine changes
- do not promise cross-engine native KV compatibility

### Final demo set

#### Demo 1: crash survival

```text
vLLM dies
→ Ralph recovers
→ same session continues
```

#### Demo 2: fast vs portable resume

```text
KV present
→ fast restore

KV deleted
→ token replay/prefill
→ still restores
```

#### Demo 3: hibernate

```text
GPU session
→ hibernate
→ GPU memory freed
→ resume
```

#### Demo 4: machine handoff

```text
laptop
→ .ralph / SSH handoff
→ remote GPU
→ same session continues
```

#### Demo 5: engine restart/upgrade

```text
old worker
→ persist sessions
→ restart/replace worker
→ compatible restore or rebuild
→ continue
```

### Final acceptance criteria

Ralph is finished when:

- sessions outlive workers
- recovery works without KV state
- compatible acceleration state makes recovery faster
- sessions can pause/hibernate/resume
- sessions can be exported/imported
- sessions can move through SSH
- a GPU can be drained safely
- engine restart does not destroy logical sessions
- at least one real-world model/demo is documented
- failure behavior is deterministic and tested
- all Section 16 edge cases applicable to implemented features have explicit tests or documented unsupported behavior
- CLI remains concise, scriptable, and safe under interruption
- core development and demos work on the single RTX 5050 workstation
- no paid GPU/cloud service is required for any core acceptance criterion
- optional multi-GPU tests detect hardware and skip cleanly when unavailable
- no unnecessary orchestration layer has been added

---

# 18. Scope guardrails for Claude Code / Codex

Before implementing any feature, ask:

1. Does this directly improve session continuity?
2. Does it support survive, sleep, or move?
3. Can it be built using an existing engine/storage primitive instead of creating a new subsystem?
4. Is it required in the current phase?
5. Does it keep the normal CLI simpler rather than expose another internal concept?
6. If it fails halfway, is the user's recoverable state unambiguous?
7. Can the core path be developed and tested on one RTX 5050 workstation?
8. Does it avoid making a paid service or special datacenter topology mandatory?

If the answer is no, do not build it.

Do not add:

```text
web dashboard
Kubernetes operator
agent framework
custom CUDA kernels
custom inference engine
Firecracker
distributed consensus
general sandboxing
autoscaling
benchmark product
profiling product
multi-cloud abstraction
```

unless this specification is explicitly changed.

---

# 19. Product positioning

Short:

> **Ralph is a continuity runtime for stateful LLM inference.**

Technical:

> **Ralph decouples inference session lifetime from inference worker lifetime.**

User-facing:

> **Your model process can die. Your inference session does not have to.**

Core verbs:

```text
survive
sleep
move
```

Everything Ralph builds should fit under one of those three verbs.

---

# 20. Final product shape

At the end, the normal Ralph workflow should feel this small:

```bash
ralph run Qwen/Qwen3-8B --name research

ralph query research "..."

ralph pause research
ralph resume research

ralph hibernate research

ralph recover research

ralph export research
ralph import research.ralph

ralph handoff research user@gpu-box

ralph drain gpu0
```

The implementation underneath may become sophisticated.

The product should not feel sophisticated.

A Ralph feature is only finished when:

```text
happy path is short
failure path preserves state
error tells the user what happened
Ctrl-C is safe
JSON mode is scriptable
repeating the command is deterministic
```

That is the design rule for Ralph.
