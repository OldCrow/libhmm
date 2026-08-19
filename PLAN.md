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
Last reconciled against live GitHub state: 2026-08-19.
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
Renumbered 2026-08-16 for the v4.3.0 release, on libstats' rule: a milestone
title names the version its work actually ships in, and no shipped work gets
relabelled with a version it did not ship in. The release contents went into a
NEW closed milestone (#3); the two feature milestones each moved up one minor
version. Milestone NUMBERS did not change, only titles — #1 is now v4.4.0.

- v4.3.0 — Numerical Correctness & Build Contract (CLOSED, #3): 0 open / 6
  closed — #63, #70, #72, #73, #75, #76. Shipped 2026-08-16.
- v4.4.0 — Training & Core Usability (open, #1): 3 open / 5 closed.
  - #43 CLOSED 2026-08-18 — BasicHmm::clone(); shipped on dev/v4.4.0.
  - #44 CLOSED 2026-08-18 — sample(hmm, T, rng); shipped on dev/v4.4.0.
  - #45 CLOSED 2026-08-19 — fit_best_of_n() multi-restart; shipped on dev/v4.4.0.
  - #46 OPEN — feat: HMM topology constraints — left-to-right, banded, and skip topologies.
  - #48 OPEN — perf: parallel E-step accumulation across sequences using ThreadPool.
  - #78 OPEN (filed 2026-08-19) — usability: reject never-initialised
    (all-zero pi/trans) models at calculator/trainer entry instead of the
    confusing "no valid observation sequences" downstream failure. Decision
    on the issue: (b) descriptive throw + (c) docs; uniform default init (a)
    rejected. Fix in flight on dev/v4.4.0; SegmentalKMeansTrainer is
    deliberately exempt (it self-initialises pi/trans).
  - #58 CLOSED 2026-08-18 — tier-2 dispatch extension; shipped on dev/v4.4.0.
  - #74 CLOSED 2026-08-18 — clean-room cos/sin + ULP gates; moved onto this
    milestone at closure (it ships in v4.4.0), shipped on dev/v4.4.0.
- v4.5.0 — Algorithm Coverage (open, #2): 3 open / 0 closed.
  - #47 OPEN — feat: GMMDistribution — Gaussian Mixture Model emission for multimodal states.
  - #51 OPEN — feat: online/streaming forward calculator — incremental α update for real-time inference.
  - #52 OPEN — feat: N-best Viterbi decoding — return top-k most probable state paths.

## GitHub Issues Without Milestone [DERIVED]
- Open issues without milestone:
  - #50 OPEN — feat: Hidden Semi-Markov Model (HSMM) with explicit duration distributions.
  - #53 OPEN — feat: Input-Output HMM (IOHMM) — covariate-conditioned transition probabilities.
  - #77 OPEN — accuracy: log1p_batch doc claims small-|x| accuracy its
    add-then-log implementation cannot deliver (filed 2026-08-18, untriaged).
- Closed issues without milestone: 20 as of 2026-07-18. Note #63/#70/#72/#73/
  #75/#76 were moved ONTO the new v4.3.0 milestone at release, so they are no
  longer in this section; their detail lives in Numerical Defect Triage below.
  Fetch the full closed-unmilestoned list via `gh issue list --state closed
  --json number,title,milestone -q '.[] | select(.milestone == null)'` if ever
  needed.

## In Progress [OPEN]
### v4.4.0 milestone (branch dev/v4.4.0)
- **#58 and #74 are COMPLETE on dev/v4.4.0** (2026-08-18), not yet merged to
  main or released. Full design/decomposition record: this section's text at
  commit cd1b141 (git history); outcome detail in Numerical Defect Triage
  below. Sub-agent decomposition ran T0-T4 (design/verify here, three Sonnet
  implementation agents); two agent errors caught by verification: the NEON
  relocation landed in the non-AArch64 #else stub section (masked on every
  x86 leg, broke every macOS link), and a corvus AGENTS.md rule ("MSVC caps
  at AVX2") leaked into a brief — countermeasures recorded in session memory.
- **#45 COMPLETE on dev/v4.4.0** (2026-08-19). Both pre-implementation design
  points settled as recorded here:
  - "Randomise emissions from prior" → **random-subsample refit**. No prior
    machinery exists and a per-family randomise() virtual was out of scope.
    Scalar restarts refit each state via the existing unweighted fit() on a
    small random subsample (with replacement, m = clamp(pool/(4N), 2, 32)) of
    the pooled observations — small samples carry the variance that makes
    starts diverse; large subsamples would fit every state to near-identical
    pooled statistics. MV restarts run kmeans_init() with fresh k-means++
    seeding, per the issue. π/A keep their cloned values (first E-step
    re-estimates them once emissions differ).
  - 90%-of-seeds acceptance → **made structural, not statistical**. Restart 0
    trains from the caller's current parameters unrandomised, so
    fit_best_of_n ≥ single run holds by construction (the single run IS one
    of the candidates). The acceptance test asserts that deterministic
    invariant on every platform; the separate multimodal-recovery test uses
    10σ-separated modes from a symmetric-emissions trap (a fixed point plain
    EM cannot leave) with margins wide enough for any conforming RNG stream.
  - Per-restart exceptions discard that restart; rethrown only if all fail.
    Restarts serial; parallel execution deferred to #47's ThreadPool work.
- Remaining v4.4.0 issues: #46 topology (independent), #48 parallel E-step
  (decide the threading reversal first — see Next Steps).
- Merge dev/v4.4.0 → main and release when the milestone empties.

## Numerical Defect Triage (2026-08-16) [DERIVED]
**#72, #73, #75 and #76 are FIXED and pushed** (84bc997, 5274c6d, 819f4d8).
CI green on all 9 jobs at 819f4d8, with both tiers exercised: the Linux legs
run Tier 1 (the installed-consumer check reports it), macOS runs Tier 2.

#74 remains open; that is a scope call recorded below, not neglect.

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
- **#76 — DONE 2026-08-16.** Tier 1 forwarded straight
  to `std::cyl_bessel_i`, whose domain is x ≥ 0. The implementations disagree
  outside it: libstdc++ throws `std::domain_error`, MSVC's STL returns the
  even/odd continuation that Tier 2 also implements. Because the wrappers are
  `noexcept`, the libstdc++ throw reaches `std::terminate` — an abort a
  consumer cannot catch. Fix normalizes the sign in the Tier 1 wrappers so
  both tiers hold one domain. Verified under libstdc++ (mingw g++ 15, the same
  standard library as the failing CI legs) and MSVC 47/47 locally.
  **This was the #75 fix doing its job**: the `I₀(-2) == I₀(2)` assertion has
  been in the suite all along and states the intended contract; it simply
  never exercised Tier 1. It turned `main` red for one commit, which is the
  correct outcome and not a reason to soften the guard. Expect more of this
  shape — defects that were always there and are only now reachable by a test.
  The follow-up sweep folded into the #74 pass (done 2026-08-18): the C++17
  special-math family — the only `std::` math functions that throw — appears
  ONLY in `math/bessel.h`, and every call site there is |x|-normalized by
  this fix; overflow (κ > ~713) returns inf without throwing and the #73
  ratio/log-split paths avoid it anyway. No other `noexcept` wrapper in the
  repo fronts a throwing `std::` function.
- **#75 — DONE 2026-08-16. Found while writing #72's tests, and it is why #72
  survived.** `LIBHMM_HAS_CXX17_BESSEL` was `PRIVATE` to `hmm_objects`, so no
  test TU had ever compiled Tier 1 — the existing `BesselFunctions.*` tests
  exercised the A&S fallback on every platform, which is why their tolerances
  are 1e-4…1e-7. It was also an ODR violation and it reached installed
  consumers. Now a generated `libhmm/config.h`; the convention is recorded in
  AGENTS.md CMake standard, since it applies to any future configure-time fact
  a public header branches on.
  **The durable lesson is about the guard, not the fix**: a one-sided
  assertion ("Tier 2 is within 1.6e-7") passes on a Tier 1 build too, so it
  cannot detect the regression. Both guards decide independently, via
  `__cpp_lib_math_special_functions`, whether the compiler has the C++17
  special math functions and require libhmm to agree — and the test was
  confirmed to fail against the pre-fix build before being trusted.
  Scope note: `LIBHMM_HAS_CXX17_BESSEL` was the *only* compile definition in
  the entire build, so the "audit whether other PRIVATE definitions leak into
  public headers" half of the issue is closed rather than deferred.
- **#74 — DONE 2026-08-18 on dev/v4.4.0, jointly with #58.** The oracle/
  ULP-gate prerequisite was built rather than deferred: self-checking mpmath
  generators (scripts/gen_trig_cleanroom_table.py, gen_trig_ulp_vectors.py),
  checked-in correctly-rounded references (5000 points + specials, incl. a
  near-k·π/2 stress walk), and per-tier gates calling the per-ISA batch
  symbols directly (tests/performance/test_trig_ulp_gates.cpp). The kernel is
  libstats' clean-room quadrant-reduction cos ported to all four tiers, plus
  sin_pd/sin_batch from the quadrant table. Measured (Zen 4, MSVC): max 1 ULP
  / mean ~0.03 ULP on scalar, SSE2, AVX2, AVX-512 — the max-1 points are
  hard-to-round ties the platform libm also misses by exactly 1 (verified
  against mpmath), i.e. the kernel is faithfully rounded; NEON gated in CI.
  Old kernel was ~2e-10 absolute (~9×10⁵ ULP). Gate-can-fail verified: a
  one-bit coefficient perturbation drives the gate to ~98 ULP and red.
  VonMisesDistribution.BatchMatchesScalar tightened 1e-9 → 1e-12.

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
  idiom); see AGENTS.md CI/Validation for the full rationale.
  **#63 CLOSED 2026-08-16** — 69 `[[nodiscard]]` attributes across the three
  linalg headers, and the cluster is now gone.
  **The "~1344 residual warnings" figure was an artefact and is retired.**
  `run-clang-tidy` re-reports each header diagnostic once per including TU, so
  one flagged line in a widely-included header shows up 20+ times; the private
  `BasicMatrix3D::index()` helper alone accounted for 23 of them. Counting
  unique `file:line:col + check` tuples, the real residual is **86 sites**,
  led by `modernize-use-auto` (16) and `modernize-concat-nested-namespaces`
  (9) — all mechanical, none architectural. Always de-duplicate before
  quoting a count from that job.
  That changes the #62 decision materially: promoting clang-tidy to blocking
  was deferred on an apparent scale problem that does not exist. 86 mechanical
  sites is a tractable target, so it is now a scheduling question. Worth
  reopening #62's assessment rather than waiting on "a few noise-free
  cycles".
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
- Work through the remaining v4.4.0 — Training & Core Usability backlog
  before v4.5.0 — Algorithm Coverage. #58, #74, #43, #44 and #45 are done on
  dev/v4.4.0. Next: #46 topology constraints — keep tied-states OUT of scope
  (the scope hazard flagged in the 2026-08-18 estimate), then the #48
  threading-reversal decision.
- Decide whether issue #48 (parallel E-step accumulation) should proceed;
  if so, record the reversal of the "threading not used" decision above
  when work begins, rather than leaving both statements to coexist
  silently. Note #48 touches the same FB/BW TUs #58 just restructured.
- Not yet started, and not scheduled: a corvus adoption spike. If one is
  run, aim it at `lgamma` and the batch path. The two-dispatch-mechanism
  question is now SETTLED by #58: corvus would slot behind DoubleVecOps
  table entries like everything else — see Cross-Repo Dependencies.
- Triage #77 (log1p_batch small-|x| doc/implementation gap) — filed
  2026-08-18, unmilestoned.
