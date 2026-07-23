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
- Special functions (regularized incomplete gamma/beta, inverse erf) are
  implemented from public-domain references only — Abramowitz & Stegun, NIST
  DLMF, Lentz (1976), Winitzki (2008) — never Numerical Recipes, whose code is
  proprietary and non-redistributable. Keeps the distributed tree MIT-clean
  (`src/distributions/distribution_base.cpp`). Residual "Numerical Recipes"
  strings in `CHANGELOG.md` and `benchmarks/docs/` describe the external
  LAMP_HMM comparator only, not libhmm code, and are intentionally left as-is.

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
  - #62 OPEN — chore: decide whether to enable clang-tidy in CI.
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
- clang-tidy is available but disabled in CI (`LIBHMM_ENABLE_CLANG_TIDY=OFF`) —
  tracked as GitHub issue #62 (2026-07-14).
- JOSS submission deferred (2026-07-19): JOSS rejected the paper for
  insufficient open-source/research uptake of libhmm (newer scope
  requirement), with an explicit invitation to resubmit once the library
  is in use in open-source or research projects. PR #20 closed without
  merging; `joss-paper` retained as a long-lived paper branch (JOSS
  accepts submission from a named branch). Published arXiv paper:
  arXiv:2605.29208 (v2, 2026-06-13), source tagged `arxiv-v2` on
  `joss-paper`. Resubmission checklist when uptake exists: merge `main`
  into `joss-paper`, refresh benchmarks/figures/version references,
  gather citation/usage evidence (CITATION.cff on `main` supports this),
  open new PR + JOSS submission.

## Cross-Repo Dependencies [OPEN]
pylibhmm pins this repo via `FetchContent` (`GIT_TAG v4.2.4` in
`pylibhmm/CMakeLists.txt`, with a local-source-tree override for
side-by-side development). As of the v4.2.5 release (2026-07-19) libhmm
`main` is at tag v4.2.5, so pylibhmm's pin now lags by one patch release.
v4.2.5 is a license-hygiene patch with no API or behavior change, so the
lag is safe; bump pylibhmm's `GIT_TAG` to v4.2.5 at the next convenient
sync. Before cutting a new libhmm release or making a breaking API change,
verify pylibhmm's pin rather than assuming it's current.

## Local Machine State [DERIVED]
Confirmed 2026-07-14: `main` fully in sync with `origin/main` (clean,
no ahead/behind). `joss-paper` branch (PR #20 closed 2026-07-19 after
JOSS deferral — see Known Gaps; branch retained for resubmission) matches
`origin/joss-paper` exactly. A local-only stash on `joss-paper`
containing only regenerated LaTeX build artifacts (`.aux`, `.blg`,
`.fdb_latexmk`, `.fls`, `.pdf` — no `.tex` source changes) was dropped
as safe-to-discard output. 8 stale local branches left over from
squash-merged PRs (#30, #31, #33, #34, #57, #59, #60, #61) were deleted
locally — confirmed merged via `gh pr list` before deletion, not
unmerged work.

## Build-Stack Standardization (2026-07-23) [DERIVED]
Cross-repo effort tracked in `~/Development/BUILD-STANDARDIZATION-PLAN.md`
(house style: `~/Development/CMAKE-HOUSE-STYLE.md`). Phases 0-3A complete,
CI-green, no library API/behavior change: `66a7568` (install-export repair —
`install(TARGETS ... EXPORT)`, GNUInstallDirs, `AnyNewerVersion` ->
`SameMajorVersion`), `610cdf4` (GNUInstallDirs + pkg-config + kebab
`libhmm-config.cmake` + consumer-example CI smoke test), `5445a0a`
(CMakePresets.json schema 6, CMake minimum 3.25, AGENTS.md CMake-standard
section), `8b0b6f7` (`LIBHMM_*`-prefixed options with one-release
deprecation shim, target-scope includes/warnings, `LIBHMM_WERROR`,
`BUILD_SHARED_LIBS` removed — coordinated with pylibhmm `7a06b42`). CHANGELOG.md
`[Unreleased]` section and AGENTS.md CMake-standard section updated to match.

## Next Steps
- Work through the v4.3.0 — Training & Core Usability backlog (6 open
  issues above) before starting v4.4.0 — Algorithm Coverage.
- Decide whether issue #48 (parallel E-step accumulation) should proceed;
  if so, record the reversal of the "threading not used" decision above
  when work begins, rather than leaving both statements to coexist
  silently.
