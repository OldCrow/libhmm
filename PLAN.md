# libhmm — Plan / Status

## Decided [DERIVED]
- v4 template parameterization (`BasicHmm<Obs>`, `BasicEmissionDistribution<Obs>`)
  preserves the v3 scalar API via type aliases; multivariate support was added
  without breaking existing consumers.
- Two-tier SIMD strategy: 11/16 distributions runtime-dispatched via
  `DoubleVecOps`; 5 remain tier-1 by design (lgamma dependency, gather
  complexity, or trivial cost — see AGENTS.md Architecture for the
  per-distribution rationale).
- JSON is the recommended I/O format going forward; XML is deprecated,
  scalar-only, retained for reading existing files only.
- Threading is not used in the production path — this is a deliberate,
  settled decision, not an oversight. The Phase 4 refactor (v3.0.0-alpha)
  replaced a Plan-A four-level hierarchy (`WorkStealingPool`-based
  parallelism) with Plan B (per-distribution batch SIMD). `ThreadPool` was
  later moved out of the library entirely and into `tools/`, since no
  production code ever instantiated it — only two diagnostic tools
  (`analyze_overhead`, `debug_parallel`) use it, to measure thread-pool
  overhead empirically. See GitHub Milestones below: issue #48 tracks a
  *prospective* future reintroduction of threading for parallel E-step
  accumulation — that issue is open/unstarted, not a contradiction of this
  decision.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-07-14.
- GitHub is the collaborator-facing source for issues and milestones; this
  PLAN.md is the agent-facing durable project state. Keep both in sync.
- When creating, closing, reopening, retitling, or moving a GitHub issue or
  milestone, update this section in the same change set or note why it could
  not be updated.
- Reconcile this section against live GitHub state when either is true:
  (a) the task at hand involves reading the backlog to decide what to work
  on next, or creating/closing/retitling/moving an issue or milestone, or
  (b) more than 7 days have passed since the "Last reconciled" date above.
  Skip the check for tasks that don't touch the backlog or this file at
  all — a per-session or per-task refresh regardless of relevance is
  wasted effort in one direction and a rubber stamp in the other. Update
  the "Last reconciled" date whenever this section is actually re-checked,
  whether or not anything had drifted.
- Convention: open (actionable) milestones/issues are fully itemized here;
  closed/historical ones are summarized as counts only, since their content
  is immutable and retrievable on demand via `gh` — no need to keep it
  loaded in every session's context.

## GitHub Milestones [DERIVED]
- v4.3.0 — Training & Core Usability (open, #1): 6 open / 0 closed.
  - #43 OPEN — feat: BasicHmm::clone() — deep copy for restarts, checkpointing, and ensemble methods.
  - #44 OPEN — feat: HMM-level sequence sampling — sample(hmm, T, rng).
  - #45 OPEN — feat: multi-restart training — fit_best_of_n() for robust EM convergence.
  - #46 OPEN — feat: HMM topology constraints — left-to-right, banded, and skip topologies.
  - #48 OPEN — perf: parallel E-step accumulation across sequences using ThreadPool.
  - #58 OPEN — perf: extend tier-2 runtime ISA dispatch to FB/BW/transcendental TUs (wheel portability without performance cost).
- v4.4.0 — Algorithm Coverage (open, #2): 3 open / 0 closed.
  - #47 OPEN — feat: GMMDistribution — Gaussian Mixture Model emission for multimodal states.
  - #51 OPEN — feat: online/streaming forward calculator — incremental α update for real-time inference.
  - #52 OPEN — feat: N-best Viterbi decoding — return top-k most probable state paths.

## GitHub Issues Without Milestone [DERIVED]
- Open issues without milestone:
  - #50 OPEN — feat: Hidden Semi-Markov Model (HSMM) with explicit duration distributions.
  - #53 OPEN — feat: Input-Output HMM (IOHMM) — covariate-conditioned transition probabilities.
- Closed issues without milestone: 19 as of 2026-07-14 (fetch via
  `gh issue list --state closed --json number,title,milestone -q
  '.[] | select(.milestone == null)'` if ever needed).

## In Progress [OPEN]
- (none currently tracked outside the GitHub milestone backlog above —
  populate as work actually starts)

## Known Gaps [OPEN]
- Distribution fit-quality improvements: see docs/GOLD_STANDARD_CHECKLIST.md
  for the prioritized list (this file doesn't duplicate it; confirmed
  2026-07-14 that its scope is narrowly fit-quality/interface-completeness
  tracking only, with no broader project task-tracking that should move
  here instead).
- clang-tidy is available but disabled in CI (`ENABLE_CLANG_TIDY=OFF`) —
  no decision recorded on whether/when to enable it.

## Cross-Repo Dependencies [OPEN]
pylibhmm pins this repo via `FetchContent` (`GIT_TAG v4.2.4` in
`pylibhmm/CMakeLists.txt`, with a local-source-tree override for
side-by-side development). Confirmed in sync as of 2026-07-14: libhmm
`main` is at tag v4.2.4, matching pylibhmm's pin exactly. Before cutting a
new libhmm release or making a breaking API change, verify pylibhmm's pin
is updated to match — check pylibhmm's PLAN.md for its current pinned tag
rather than assuming it's current.

## Next Steps
- Work through the v4.3.0 — Training & Core Usability backlog (6 open
  issues above) before starting v4.4.0 — Algorithm Coverage.
- Decide whether issue #48 (parallel E-step accumulation) should proceed;
  if so, record the reversal of the "threading not used" decision above
  when work begins, rather than leaving both statements to coexist
  silently.
