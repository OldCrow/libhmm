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
Last reconciled against live GitHub state: 2026-08-16.
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
  - #63 OPEN — chore: bulk-apply `[[nodiscard]]` in the three linalg headers
    (clang-tidy `modernize-use-nodiscard` cluster surfaced by #62's advisory
    CI job); a prerequisite for reconsidering blocking clang-tidy CI.
  - #70 OPEN — chore: audit compensated accumulation paths for FP-contraction
    sensitivity. The libhmm counterpart of libstats #84; both trace to corvus's
    cross-compiler finding that GCC's default `-ffp-contract=fast` fuses inside
    a compensated sequence. Was never listed here — this section is derived
    from GitHub rather than maintained by hand, so re-derive it rather than
    trusting it between passes.
  - #72 OPEN — fix(math): `log_bessel_i0` has a ~1900 ULP step discontinuity at
    x = 700. See Numerical Defect Triage below.
  - #73 OPEN — fix(von-mises): `getCircularVariance()` returns NaN for
    κ ≥ 713.99 and cancels ~log₂(2κ) bits below it. See Numerical Defect
    Triage below.
  - #74 OPEN — accuracy: SIMD `cos_pd` is ~2e-10 at every tier, and there is no
    `sin_pd`. See Numerical Defect Triage below.
  - #75 OPEN — fix(build): `LIBHMM_HAS_CXX17_BESSEL` is `PRIVATE` to
    `hmm_objects`, so test TUs and installed consumers compile the Tier 2
    Bessel fallback while the library ships Tier 1. See Numerical Defect
    Triage below.
- Closed issues without milestone: 20 as of 2026-07-18 (#62 closed — clang-tidy
  CI decision recorded, see Known Gaps below; fetch full list via
  `gh issue list --state closed --json number,title,milestone -q
  '.[] | select(.milestone == null)'` if ever needed).

## In Progress [OPEN]
- (none currently tracked outside the GitHub milestone backlog above —
  populate as work actually starts)

## Numerical Defect Triage (2026-08-16) [DERIVED]
**#72 and #73 are FIXED and pushed (84bc997).** #74 and #75 remain open; both
are scope calls recorded below, not neglect.

Three accuracy defects were found by carrying libstats' closed
`spike/corvus-bessel` findings across to this repo, rather than by running a
spike here. libhmm's `math/bessel.h` and libstats' `core/bessel.h` share a
design — the same two-tier `LIBHMM_HAS_CXX17_BESSEL` / A&S split, the same
`x > 700` asymptotic — so libstats' measurements transferred directly and were
then re-confirmed here against mpmath at dps 60 and against real
`std::cyl_bessel_i`.

All three are **independent of the corvus adoption question**: they are
in-repo defects in code libhmm owns, and adopting corvus would close none of
them outright (#73 in particular cancels regardless of how accurate the
Bessel ratio is — the defect is the formulation).

- **#72 / #73 — DONE 2026-08-16.** #72 was a two-term extension of the
  asymptotic bracket (≤ 0.79 ULP over x ∈ [700, 20000]). #73 had a hard-bug
  half (NaN above κ = 713.99, reachable through `fit()` on concentrated
  angles, since `kappa_from_r_bar` returns 1e6 for R̄ ≥ 1) and a conditioning
  half, both closed by `detail::one_minus_bessel_ratio` — a tier-independent
  series above κ = 30 (≤ 2 ULP) and the direct form below it (≤ 52 ULP).
  **The residual bound is the direct branch, and it is not improvable in
  place**: 52 ULP is the 2κ amplification acting on correctly-rounded inputs,
  and the series diverges below κ ≈ 25, so nothing sits between them. Record
  that before anyone re-opens it looking for a tighter crossover.
- **#75 — found while writing #72's tests, and it is why #72 survived.**
  `LIBHMM_HAS_CXX17_BESSEL` is `PRIVATE` to `hmm_objects`, so no test TU has
  ever compiled Tier 1; the existing `BesselFunctions.*` tests exercise the
  A&S fallback on every platform, which is why their tolerances are 1e-4…1e-7.
  It is also an ODR violation (the same `inline` functions get two different
  bodies in one binary) and it reaches installed consumers, who get Tier 2
  silently. The #72/#73 tests work around it by going through the public
  `VonMisesDistribution` API; that is a workaround, not a fix. Fixing it means
  a generated config header, which is a build-surface change and was kept out
  of the fix commit deliberately.
- **#74 is NOT tractable now, and that is a scope judgement, not a deferral
  by neglect.** Fixing SIMD cos means authoring a kernel across four ISA
  tiers, and this repo has no oracle or per-tier ULP-gate infrastructure to
  validate one against — `tests/performance/` holds a single test file. The
  bounds recorded in #74 are the kernels' own header claims, not independent
  measurements. Building the gate is the first step and is its own piece of
  work. It also interacts with #58 and with the corvus question.

## Known Gaps [OPEN]
- Distribution fit-quality improvements: see docs/GOLD_STANDARD_CHECKLIST.md
  for the prioritized list (this file doesn't duplicate it; confirmed
  2026-07-14 that its scope is narrowly fit-quality/interface-completeness
  tracking only, with no broader project task-tracking that should move
  here instead).
- clang-tidy CI job (advisory, non-blocking, `continue-on-error: true`)
  per the #62 decision; the build-integration option is
  `LIBHMM_ENABLE_CLANG_TIDY` (`OFF` by default) after the Phase 3A option
  rename. Six checks were
  disabled in `.clang-tidy` as architecturally mismatched with libhmm's
  design (pragma-once convention, intentional SIMD intrinsics/pointer
  arithmetic, the v4 template+virtual pattern, a false-positive move-ctor
  idiom); see AGENTS.md CI/Validation for the full rationale. Residual
  warnings (~1344) are dominated by a mechanical `modernize-use-nodiscard`
  cluster in the linalg headers, tracked as #63 — promoting the job to
  blocking is contingent on that landing and a few noise-free cycles.
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
pylibhmm consumes this repo via `FetchContent` against a pinned release
tag, with a local-source-tree override for side-by-side development.

**The pin value is deliberately not restated here.** `GIT_TAG` in
`pylibhmm/CMakeLists.txt` is the single source of truth; a copy in this
repo has no way to notice when it goes wrong, and a previous copy here
did exactly that (asserted a one-release lag that did not exist, and
prescribed a bump that had already happened). Read the pin from that
file — never from this one.

The invariant this repo owns: before cutting a new release or making a
breaking API change, check pylibhmm's pin and coordinate the bump.
Option renames count as breaking for this purpose — pylibhmm's
`FetchContent` path force-sets libhmm option names, so a rename here
requires a matching change there. Staleness of the pin itself is
checked mechanically by pylibhmm's monthly CI canary (its issue #15),
not by prose in either repo.

[OPEN] **Whether libhmm adopts corvus as a dependency is undecided**, and the
decision is tracked in `corvus/PLAN.md`, not here — same rule as the pylibhmm
pin. State as of 2026-08-16, recorded so a session does not re-derive it:

- corvus is at v0.5.0 with a core/generator/test freeze; only machine-blocked
  legs (Kaby, M1 quiet bench) stand before v1.0.0. Nothing here gates on it.
- **libstats has already settled adoption as intent**, on the strength of the
  wider special-function surface rather than any single family. Its spike
  established the mechanism cross-repo: clang-cl-built corvus links into an
  MSVC consumer, the dispatch tier follows the compiler that builds corvus's
  TUs (not the delivery mechanism), and AVX2 vs AVX3_ZEN4 corvus produce
  byte-identical output. A libhmm spike would not need to re-answer any of it.
- **The libhmm case is different from libstats' and stronger.** libstats'
  spike targeted eight cold scalar Bessel call sites — an accuracy win, not a
  throughput one, and explicitly not enough to justify a dependency on its
  own. libhmm's case is `lgamma`: AGENTS.md Architecture states outright that
  Poisson, Binomial and NegativeBinomial are tier-1 *because* no portable
  vectorized lgamma exists, and that they "would become tier-2 if a
  vectorized lgamma is added to `simd_double_ops_*.cpp`". That is three of
  sixteen distributions moving from a scalar loop into the per-state,
  per-`compute()` batch path. corvus ships exactly that kernel, audited per
  tier, MIT, portable via Highway.
- **The open design question, and it should be settled before any spike, not
  during one:** libhmm dispatches through its own CPUID-built `DoubleVecOps`
  function-pointer table across five per-ISA TUs; corvus brings Highway's own
  runtime dispatch. Adoption means two dispatch mechanisms in one binary.
  #58 (extend tier-2 runtime ISA dispatch to the FB/BW/transcendental TUs) is
  the natural place to decide the shape.
- Secondary surface, beyond lgamma: digamma/trigamma (`psi_functions.h`,
  ~2e-14), incomplete gamma/beta and erfinv (`distribution_base.cpp`), and
  i0/i1 — all four are in corvus's audited set.

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
Cross-repo effort tracked in the fleet standards repo:
[record](https://github.com/OldCrow/standards/blob/main/records/BUILD-STANDARDIZATION-PLAN.md),
[house style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md).
Phases 0-3A complete,
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
- **#75 next.** It is the cheapest of the three remaining and it is a
  prerequisite for the other two being testable: until the Bessel tier
  propagates, no test can assert against the code that actually ships, and the
  same trap applies to any future header-level accuracy work.
- Then decide #74's prerequisite: whether to build the oracle/ULP-gate
  infrastructure the SIMD kernels currently have no way to be validated
  against. That decision also unblocks any accuracy claim this repo might
  want to publish, and it is the same question a corvus adoption would force.
- Not yet started, and not scheduled: a corvus adoption spike. If one is run,
  aim it at `lgamma` and the batch path, and settle the two-dispatch-mechanism
  question (#58) first — see Cross-Repo Dependencies.
- Work through the v4.3.0 — Training & Core Usability backlog (6 open
  issues above) before starting v4.4.0 — Algorithm Coverage.
- Decide whether issue #48 (parallel E-step accumulation) should proceed;
  if so, record the reversal of the "threading not used" decision above
  when work begins, rather than leaving both statements to coexist
  silently.
