# libhmm defect hunt — 2026-10-12

Durable record (DOC-CONVENTIONS: single-repo evidence lives in docs/). Session summary first, per-agent sections after.

## Issues filed 2026-10-12 (one per fix unit; bodies carry inputs, expected vs got, guard)
| # | subject | summary item | milestone |
|---|---|---|---|
| #112 | MAP-BW static_cast crash on Poisson/Binomial/NegBin | A1 | v4.4.2 — Safety & contract patch |
| #113 | fit(data, weights) size unchecked — SEGV/OOB all classes | A2 | v4.4.2 — Safety & contract patch |
| #114 | SegmentalKMeans SEGV on all-empty lists; π sum 0.5 | A3 | v4.4.2 — Safety & contract patch |
| #125 | VonMises sampler hang κ∈[1e-9,1e-8]; NaN-param samplers loop | A4 | v4.4.2 — Safety & contract patch |
| #115 | clone()/copy ctor race with cold cache fill | A5 | v4.4.2 — Safety & contract patch |
| #116 | Binomial O(n) table; DoS via JSON n | A6 | v4.4.2 — Safety & contract patch |
| #117 | moved-from + reset() crash | A7 | v4.4.2 — Safety & contract patch |
| #126 | FB posterior drift ~T²·ε | B8 | Fit accuracy & kernel hygiene patch |
| #127 | gammap accuracy near x≈a | B9 | Post-corvus kernel patch |
| #128 | NegBin CDF; Poisson CDF normal-approx switch | B9 | Post-corvus kernel patch |
| #129 | CDF lower tails via 1−upper; Pareto/VonMises/Uniform CDF range | B9 | Fit accuracy & kernel hygiene patch |
| #130 | Bessel Tier 2 bounds; digamma/trigamma switch | B10 | Post-corvus kernel patch |
| #131 | ChiSquared ν floor; Beta small-shape fit | B11 | Fit accuracy & kernel hygiene patch |
| #132 | Weibull/Rayleigh/Gaussian fit at extreme scale | B11 | Fit accuracy & kernel hygiene patch |
| #118 | non-finite data/weights install NaN; NaN policy per class | B11 | Fit accuracy & kernel hygiene patch |
| #133 | Beta sampler exact 0.5 / −inf-scoring draws | B12 | Fit accuracy & kernel hygiene patch |
| #134 | accepted extreme parameters → NaN / wrong logpdf | B13 | Fit accuracy & kernel hygiene patch |
| #135 | moment accessors (Weibull var, NegBin mode, underflow) | B14 | Fit accuracy & kernel hygiene patch |
| #136 | FullCov Σ+1e-5·I vs getCovariance | B14 | Fit accuracy & kernel hygiene patch |
| #137 | BW `a <= 0.0` zero-transition test | B15 | Fit accuracy & kernel hygiene patch |
| #119 | from_json positional keys | C16 | v4.4.2 — Safety & contract patch |
| #120 | validation parity: from_json_mv, legacy readers, setPi/setTrans | C17, C18 | v4.4.2 — Safety & contract patch |
| #121 | subnormal round trip (ERANGE) | C19 | v4.4.2 — Safety & contract patch |
| #138 | batch vs scalar IEEE edges; exp_batch body/tail | C20 | Fit accuracy & kernel hygiene patch |
| #122 | SegKM/kmeans_init non-finite obs; topology after Viterbi/SegKM | C21, C22 | v4.4.2 — Safety & contract patch |
| #123 | legacy I/O minor cluster | C22 | v4.4.2 — Safety & contract patch |
| #124 | dispatch mutant undetectable; traits test unregistered; timing labels | D23, D24 | v4.4.2 — Safety & contract patch |
| #139 | #88 cast guard survives mutation on x86 (m88); needs AArch64 evidence or an ISA-independent guard | D25 | v4.4.2 — Safety & contract patch |
Milestones assigned 2026-10-12 (three-way split): v4.4.2 — Safety & contract patch (#7) ships first; Fit accuracy &
kernel hygiene patch (#5) is now corvus-independent and follows; Post-corvus kernel patch (#8) waits on the spike (#106).
#116 is split: n cap/JSON bound on #7, the lgamma path on #8 (comment on the issue).
Extensions noted on existing issues: #73 (Bessel Tier 2), #92 (StudentT one-step fit), #95 (topology after
train), #97 (NaN-param samplers), #99 (exp_pd floor), #100 (gammap tolerance), #101/#108 (test infra), #103 (erfinv numbers).

## Where the evidence lives
Probe sources and raw outputs: the `libhmm-hunt` worktree, `hunt/<agent>/` (untracked). They are inputs to the
guard work, not shipped code; copy a probe into `tests/` when its guard lands. T6 mutant worktrees are throwaway.

HEAD 58695c3 (v4.4.1 + dependabot). Kaby Lake Mac, macOS 13.7.8, AppleClang/libc++, AVX2 tier only.
Six agents, ~17.6k raw hits → 81 agent-confirmed items, ~70 distinct after de-duplication across agents.
Orchestrator re-ran 14 of the headline probes from source; every one reproduced ("VERIFIED" below).
All defects pre-existing at HEAD. No library file was edited; nothing committed; no issues filed.
Per-agent detail follows this summary; probe sources under hunt/<agent>/.

### A. Crash, hang, resource — reachable from the public API
1. MAP Baum-Welch with Poisson/Binomial/NegBinomial emissions → SIGSEGV (isDiscrete() then static_cast to
   DiscreteDistribution). Ordinary inputs. VERIFIED. [S1]
2. fit(data, weights) never checks weights.size(): SEGV or silent heap OOB read in all 16 classes + IC.
   Same class as #86, one layer up. VERIFIED. [F1]
3. SegmentalKMeans (scalar+MV) SEGV on an all-empty ObservationLists; π sums to 0.5 on {valid, empty}. VERIFIED. [H2,H3]
4. VonMises::sample hangs for κ∈[1e-9,1e-8]; Poisson(NaN)/VonMises(NaN).sample() never return. VERIFIED. [M1,F2]
5. clone()/copy ctor (every class but FullCov) racing a shared instance's first const evaluation: Release crash
   5/5 (Discrete), TSan races in Binomial and DiagonalGaussian. Violates the documented const-safety contract.
   Found independently by two agents. [S3,C1,C2,C3]
6. Binomial builds an O(n) table in the ctor/noexcept path: 24 s + 4 GB at n=5e8, ~17 GB at INT_MAX; the JSON
   loader accepts n to INT_MAX, so a 60-byte document is a DoS. [N8,F10,M6]
7. Moved-from NegBin/Discrete (+3 MV classes under ASan) after reset() → SEGV. Low reach. VERIFIED. [S2]

### B. Accuracy in core algorithms and special functions
8. Forward-backward posteriors drift ~T²·ε (scalar+MV, both modes): |γ−0.5| = 1.2e-5 at T=1e6 for identical
   emissions; feeds every M-step. VERIFIED. [H1]
9. CDF family: gammap off 1e-8 rel at x≈a (6.7e-5 at a=1e5); NegBin CDF grossly wrong (1e-5 vs 0.632; 6.8e-61 vs
   0.999999); Poisson CDF silently switches to a normal approx at k,λ>100 (err 2e-3 at λ=1000, 0 at λ=700);
   lower tails flush to 0 via 1−upper in Gaussian/LogNormal/Exponential/Weibull/Rayleigh (F(−10)=0);
   VonMises CDF 0.084 vs 0.99 at κ=1e6. VERIFIED (Gamma, Gaussian). [N1–N4, M side]
10. Bessel Tier 2 (every macOS build) exceeds its header bounds: log I0 4.7e-7 (claims <1.6e-7); 1−I1/I0 rel
    8.5e-6 below x=30 — the #73 52-ULP bound holds on Tier 1 only. digamma/trigamma series stops early at x=6
    (trigamma(6) rel 1e-9 vs claimed 2e-14). [N10,N11]
11. Fits: ChiSquared returns ν≈1e-4 for every ν≲0.4; Beta misses the MLE at small shapes (0.36 nat/obs);
    Weibull resets to (1,1) for data scale ≳1e6/≲1e-40 and overflows x^k on tight data; Rayleigh σ=0/inf and
    Gaussian σ→1e-6 floor on overflow; StudentT fit() is one ECME step (extends #92); non-finite data/weights
    install NaN/inf parameters in Gaussian/Rayleigh/VonMises/Poisson/Discrete/MV classes. [M10–M16, F2, F7, S4]
12. Samplers: Beta returns exactly 0.5 (22.5% at (0.001,0.001)) and exact 0/1 that its own logpdf scores −inf
    (62% at that shape). VERIFIED. [M2,M3]
13. Extreme-parameter logpdf: scalar vs batch disagree (Weibull 1 nat, StudentT 12 nats); accepted tiny σ gives
    NaN (Gaussian σ=1e-160, Rayleigh, LogNormal); large-shape cancellation (StudentT ν=1e15: 3 nats);
    getProbability 0/NaN where the log value is ordinary (Pareto, Beta). VERIFIED (Gaussian). [N5–N7,N9,M4]
14. Moments: Weibull variance cancels at large k (0.33 rel at 1e8); NegBin getMode INT_MIN; Beta/LogNormal
    variance underflows to 0 though representable; FullCov samples/pdf use Σ+1e-5·I but getCovariance returns Σ. [M5,M7–M9]
15. BW zero-transition test `a <= 0.0` lets NaN/subnormal A entries take the dense path and pick up the exp_pd
    floor (extends #99). [H8]

### C. Contract and consistency
16. Scalar from_json reads keys positionally: reordered keys silently swap parameters (Gamma, Gaussian, DiagGauss).
    Contradicts the documented top-level key policy. VERIFIED. [F5]
17. MV JSON loader accepts nan/inf and non-PD/zero cov (33/65 mutations); legacy operator>> and XML accept
    nan/inf in pi/trans (24/24); scalar JSON holds (0/651). [F3,F4]
18. setPi/setTrans accept NaN/±inf/negative/>1; +inf in A gives mode-dependent logP. [H7]
19. to_json writes subnormals that from_json rejects (ERANGE on macOS strtod) — all classes; confirm on Linux. [F6]
20. Batch vs scalar on IEEE edges: NaN/+inf observations give NaN in batch vs −inf scalar in 7/4 classes (Beta
    lane-dependent); x=0 shape<1 +inf vs −inf; exp_batch +inf in the vector body for x∈(709.436,709.783] while
    the scalar tail is finite. No lane poisoning. [N12,N13]
21. Non-finite observations: SegmentalKMeans and kmeans_init install NaN parameters (BW/Viterbi skip);
    NaN-data fit policy differs per class (throw/skip/reset; Pareto biases n). [H4,H5,F11]
22. Viterbi/SegKM trainers leave forbidden transitions at 1/N after train() (README recipe unusable; extends #95);
    Discrete JSON accepts short probs (sum 1.33); legacy VonMises stream is write-only; Rayleigh/StudentT/VonMises
    operator<< undeclared in headers; Pareto/LogNormal operator>> half-update; trailing JSON text accepted. [H6,F8,F9,S5,S7,F12]

### D. Test infrastructure
23. A dispatch mutant forcing the scalar table on AVX2 passes all 1025 tests — a tier fall-back is undetectable. [T2]
24. tests/distributions/test_distribution_traits.cpp (12 tests) never registered; 1 DISABLED_ test; no test carries
    any label so `-LE` excludes nothing; 7 wall-clock timing assertions live in correctness tests and failed under
    today's load (Beta, Weibull). [T1,T3]
25. Mutation of accuracy fixes: m73 killed; m72 untestable on libc++; remaining 13 — see T6 section (pending/updated below).

### Clean (recorded so nobody re-hunts)
FP-contraction A/B bit-identical on every SIMD kernel, batch path, FB/Viterbi, MV (1.59M values); concurrent training
of distinct instances clean under TSan; logP vs mpmath enumeration ≤1e-15 on 2000 tiny models in every mode incl. MV;
Viterbi exact; BW monotone (288 runs); relabel/time-reversal/T=1/AIC-BIC-AICc; stale-cache-after-mutator 0/3178;
checkBatchSpans in all 16 overrides; weighted-fit guard convention in all 16; scalar/MV twins share one template;
weight-scale/duplicate/permutation invariance of fits; 10,900-instance JSON round trip bit-identical (non-subnormal).

### Guard placement (consolidated from the per-agent lists; each agent's ranked list is in its section)
CI every push (seconds): weights-size check + post-fit validity loop; MAP-BW Poisson; SegKM empty/mixed; batch==scalar
IEEE-edge × size × lane table; identical-emissions γ=0.5 at T=1e5 + N=1 A'==1; long-double brute-force N≤3,T≤6;
key-name checks + reject short probs; setPi/setTrans reject what the loader rejects; von Mises sampler watchdog;
fit scale-equivariance ×1e±6/±30; ChiSquared/Beta fit vs stored MLE; logpdf(sample())>−inf; FullCov sample cov ==
getCovariance; dispatcher-picked-best-tier asserted both ways; register traits test; label timing assertions.
Per machine/release: ~50 mpmath values each for gammap/incompleteBeta/digamma/trigamma; CDF tail table; exp and
log1p_inplace per-tier ULP gates with lane sweep; large-shape and Bessel Tier-2 sweeps; binned chi-square GoF
harness; the io2/io3 JSON mutation tables; copy-during-cold-fill and moved-from loops in the TSan/ASan legs.
Not covered here: SSE2-only, AVX-512 and NEON tiers (other fleet machines); GCC/MSVC (m72, subnormal ERANGE).

---
# libhmm defect hunt — 2026-10-12 (HEAD 58695c3, v4.4.1 + deps)

Machine: Kaby Lake Mac, macOS 13.7.8, AppleClang, AVX2 tier only (SSE2/AVX-512/NEON not reachable here).
Skill: numerical-defect-hunt. Worktree: libhmm-hunt, build-hunt (Release), baseline 51/51.
Six agents: dist-numerics (T1/T2/T4), hmm-algos (T1–T4 at HMM level), io-fit-fuzz (T3),
sampler-moments-fit (T9–T11), builds-mutation-conc (T5–T7), structural-scan (T8).
"Verified" = re-run by the orchestrator from the agent's probe source.

## hmm-algos — 8 confirmed of ~370 raw hits (detail: hunt/hmm-algos/REPORT.md)
- H1 [accuracy] FB posteriors drift ~T²·ε, scalar+MV, both modes; feeds BW/MAP-BW M-step.
  N=2 identical Gaussian emissions, T=1e6: max|γ−0.5| = 1.23e-5 (1e5: 6.5e-8; 1e4: 6.5e-10).
  N=1: A'(0,0)−1 = −9.8e-12. VERIFIED (c1_posterior_drift). Pattern = accumulated running sum of
  log scale factors not cancelled between α and β; closed-form expectation.
- H2 [crash] SegmentalKMeans scalar+MV SEGV in learnPi on all-empty ObservationLists; BW throws
  runtime_error on the same input. VERIFIED (exit 139, c3_segkm_empty).
- H3 [contract] SegmentalKMeans on {valid, empty} installs π=(0.5,0), sum 0.5. VERIFIED.
- H4 [contract] SegmentalKMeans installs NaN emission params on NaN/±inf observations; BW/Viterbi
  skip the sequence. (c4_nonfinite_obs)
- H5 [contract] kmeans_init with one NaN/inf coordinate installs NaN mean/variance (Diag+FullCov).
- H6 [contract, low, extends #95] Viterbi/SegmentalKMeans trainers run to convergence inside train()
  and leave forbidden transitions at 1/N, so the README enforce_topology-after-each-train recipe
  cannot apply; MAP-BW fills forbidden entries too (doc gap).
- H7 [contract] setPi/setTrans accept NaN/±inf/negative/>1 which the JSON loader rejects; +inf in
  A gives mode-dependent logP (MaxReduce −23.91, Pairwise NaN, Viterbi +inf), BW trains on it.
- H8 [accuracy, low, extends #99] BW zero-transition test is `a <= 0.0`, so NaN/subnormal A entries
  take the dense SIMD path and pick up the exp_pd floor: A(0,1)=1e-310 → A'=1.0e-307.
Suspected: ViterbiTrainer hasConverged()==true at logP=−inf (a test asserts it); fit_best_of_n
keeps restart 0 when all restarts score −inf.
Clean: logP vs mpmath ≤1e-15 rel on 2000 tiny models all modes incl. MV; Viterbi exact; BW monotone
288 runs; relabel/time-reversal/T=1/AIC-BIC-AICc pass; #86 span check never triggered from HMM path.
Guards proposed (yield÷cost): 1 SegKM empty-list throw + π sums to 1 (CI); 2 identical-emissions
γ=0.5 within 1e-12 at T=1e5 + N=1 A'==1 (CI); 3 long-double brute-force N≤3,T≤6 test (CI);
4 non-finite obs never install NaN (CI); 5 setPi/setTrans reject what loader rejects (CI);
6 forbidden transitions exactly 0 after Viterbi/SegKM training (CI); 7 tail-lane sweep (per-machine).

## structural-scan (T8) — 10 confirmed of ~25 raw hits (detail: hunt/structural-scan/REPORT.md)
- S1 [crash] MapBaumWelchTrainer (basic_map_baum_welch_trainer.h:178,231) checks isDiscrete() — true for
  Discrete, Poisson, Binomial, NegBinomial — then static_casts to DiscreteDistribution. 2-state Poisson HMM,
  default pseudo_count, train() → SIGSEGV. VERIFIED (probe_map_poisson, exit 139). Highest-severity hit of the hunt
  so far: reachable from an ordinary public call with ordinary parameters.
- S2 [crash] Moved-from object + reset() + evaluate → SEGV for NegBinomial and Discrete; ASan null deref also in
  DiagonalGaussian, FullCov, IndependentComponents (stale size/cache survives the move). 14 other classes fine.
  VERIFIED (probe_nb_move nbmoved, exit 139). Low practical reach (moved-from reuse).
- S3 [concurrency] Copy ctor / clone() in every class except FullCov does not take the cache lock, so a copy racing a
  const evaluation's first cache fill is a heap-buffer-overflow (ASan) / abort (Release 2/2). Contradicts the
  ARCHITECTURE.md "const ops on a shared instance are safe" contract when the other party copies. NOT re-verified
  by orchestrator (needs ASan build; see probe_copy_race). Pending TSan confirmation.
- S4 [contract] Gaussian/Rayleigh/VonMises fit({NaN,NaN}) install NaN; Rayleigh fit on 1e300 → σ=inf, 1e-300 → σ=0;
  the model then cannot reload from JSON.
- S5 [API] Rayleigh, StudentT, VonMises operator<< defined in .cpp but not declared in the header: `os << d` fails to compile.
- S6 [cosmetic] FullCov toString prints log_det 0 while cache cold.
- S7 [contract, minor] Pareto/LogNormal operator>> half-update on a second invalid value (failbit set).
- T1 [test gap] tests/distributions/test_distribution_traits.cpp (12 tests) never registered in CMake; its traits header
  leaves VonMises neither discrete nor continuous.
- T2 [missing guard] Dispatch mutant forcing the scalar table on AVX2: all 51 binaries pass (1025 tests) — a
  silent fall-back-to-scalar regression is undetectable. SSE2-capped mutant: 27/51 done, 0 failures (rest pending).
- T3 [flaky] 7 wall-clock timing assertions in unlabelled correctness tests; Beta and Weibull failed under load.
  No test carries any label, so `-LE known_broken|benchmark` excludes nothing. 13 correctness cases never run.
Suspected: no weighted-fit size precondition check; StudentT ν declared [1e-10,1e6] but fit clamps [0.1,1000];
SIMD_ALIGNMENT defined twice (unused); 1e-4 tolerances on closed forms; tests swallowing exceptions (Pareto
accepts a NaN-parameter result).
Clean: stale-cache-after-mutator 0/3178; checkBatchSpans in all 16 overrides (#86 holds); weighted-fit guard
convention in all 16; scalar/MV twins share one template.
Guards proposed: 1 MAP-BW Poisson test (CI); 2 moved-from+reset loop over 19 classes (ASan leg); 3 dispatcher
picked best reported tier, asserted both ways (CI per runner); 4 post-fit JSON reload check on degenerate data
(CI); 5 register traits test + VonMises trait; 6 label timing assertions `timing`; 7 copy-during-cold-fill (TSan leg).
Note: `nice` is refused in the sandbox; orchestrator renices from outside.

## io-fit-fuzz (T3) — 12 confirmed of ~2,660 raw hits (detail: hunt/io-fit-fuzz/REPORT.md)
- F1 [crash] fit(data, weights) never checks weights.size()==data.size(): empty weights SEGV Exponential/
  ChiSquared/Poisson/VonMises; 1 weight vs 1e6 data SEGV Gaussian/Exponential/Gamma/ChiSquared/Weibull; short
  weights = silent heap OOB read in all 16; IndependentComponents also reads past rows shorter than dim.
  VERIFIED (wlen: signal 11 Poisson/VonMises/Weibull). Same class as #86, one layer up.
- F2 [contract] fit installs non-finite params without throwing on NaN/inf data or inf/overflowing weights
  (weight guard misses sumW=+inf): Gaussian μ=NaN; Rayleigh σ=inf for data ≥1e155, σ=0 for ≤1e-200; Poisson λ
  NaN/inf; Discrete NaN probs; VonMises μ,κ NaN; MV classes NaN mean/var/cov. Poisson(NaN).sample() and
  VonMises(NaN).sample() never return (extends #97). Overlaps S4.
- F3 [contract, extends #87/#91] MV JSON loader accepts nan/inf in DiagonalGaussian mean/var and FullCov
  reg/mean/cov (33/65 mutations), plus non-PD cov, cov=0, reg<0 that setCovariance rejects. Scalar path holds (0/651).
- F4 [contract] legacy operator>>(Hmm) and XMLFileReader accept nan/inf in pi/trans (24/24); JSON rejects.
- F5 [contract] scalar distribution from_json reads keys positionally and discards names: Gamma with theta
  before k loads k=2.5, theta=1.5; Gaussian sigma/mu swap; renamed keys accepted. Contradicts the documented
  top-level key-order policy. Same for DiagonalGaussian. VERIFIED (scratchpad keyswap.cpp).
- F6 [contract] to_json writes subnormal params, from_json rejects them ("number out of range": read_double
  treats ERANGE as error; macOS strtod sets ERANGE on subnormal results). All classes. Confirm on Linux first.
  Everything else round-trips bit-identically (10,900 instances, 15 classes + MV IC).
- F7 [accuracy] Weibull fit returns (1,1) silently for data scale ≥1e6 (λ cap 1e6); tight data overflows x^k
  (CV 0.01 about 10 → λ=1.4; Weibull(300,100) → k=9053, λ=1.08).
- F8 [contract] Discrete JSON with probs shorter than n accepted (unset slot keeps default 1/3, sum 1.33); sums 3 or 0 accepted.
- F9 [API] legacy VonMises stream format is write-only: reader throws "Unknown distribution type: Von" (304/304).
- F10 [DoS] Binomial per-evaluation cost O(n): n=1e7 3.6 s, n≥1e9 does not finish; JSON loader accepts n to INT_MAX.
- F11 [contract] NaN-data policy differs per class: throw (Beta, Weibull, ChiSquared, StudentT, Poisson, Uniform),
  skip (Gamma, LogNormal, NegBin, Binomial, Discrete), reset (Exponential; Pareto for inf); Pareto skips NaN in
  the sum but counts it in n; only ChiSquared throws on empty data.
- F12 [minor] trailing text / second document accepted; fractional counts truncated; hex floats accepted; legacy
  stream is 6-decimal fixed and zeroes <5e-7 (undocumented); operator>>(Hmm) has no exception guarantee;
  StudentT fit(data) is one ECME step not an MLE (df=1.15 from t(5.5)); base doc says "reset" on near-zero
  weight sum but every implementation keeps parameters.
Held: weighted-fit guard in all classes for zero/negative/subnormal/NaN sums; weight-scale invariance,
duplicate-vs-weight-2, permutation invariance, n=1..33 all pass.
Guards proposed: 1 weights-size check in a shared helper beside checkBatchSpans (CI); 2 weight guard
`!(sumW >= 0 && sumW < inf)` + reject non-finite data, post-fit validity loop over all classes (CI); 3 route
from_json_mv and legacy readers through setters/validateParameters (CI) + the io2/io3 mutation tables as
per-release T3; 4 key-name checks in every from_json, end-of-input after root, reject short probs (CI);
5 accept subnormals in the reader; 6 rescale Weibull fit by max(x), drop the cap; 7 cap Binomial n from JSON
or make the cache O(1); 8 VonMises stream parser or document unsupported.

## builds-mutation-conc (T5/T6/T7) — PARTIAL, paused for machine load (detail: hunt/builds-mutation-conc/REPORT.md)
T5 FP-contraction A/B: CLEAN. build-hunt-fpc with -ffp-contract=off on all 138 TUs, 51/51. 1.59M values over
46 instances × 16 classes, all DoubleVecOps/TranscendentalKernels kernels, FB 3 modes, Viterbi, all trainers, MV:
bit-identical on every SIMD kernel (incl. compensated sin/cos), batch path, FB/Viterbi likelihood, MV output.
Differences only in scalar paths of the -march=native TUs (#94's set): VonMises log 74k ULP (5.7e-14 abs),
a VonMises sample 9e6 ULP (4.9e-13 abs, acos near 1) — conditioning, not defects. No one-sided NaN/inf, no sign flips.
T6 mutation (final, chain completed 2026-10-12): 13 killed, 2 survived as predicted, 1 untestable here.
| mutant | reverted | suite | failing tests |
|---|---|---|---|
| m73 | 84bc997 von Mises hunk | killed | CircularVarianceFiniteWhereI0Overflows, FitOnConcentratedAnglesGivesFiniteVariance |
| m84 | e7631ee | killed | FitOnNearlyDegenerateAnglesGivesFiniteKappa |
| m85 | cf410f1 | killed | LogSubnormalPrescale.Sse2 (guard fires on AVX2 hosts: tests call SSE2 symbols directly) |
| m81 | 3e875ee | killed | TrigSignOfZero.Sse2 (only the SSE2 part is guarded; AVX2 part is a no-op here, AVX-512/NEON untestable) |
| m74 | kTrigSinC[0] mantissa bit 8 | killed | TrigUlpGates, TrigUlpSpecialsGates (Avx2, Sse2, dispatched, tail) |
| m86 | 5018c1b | killed | BatchLogProbabilitiesRejectsShortOutSpan; test_poisson_distribution aborts |
| m87 | 34a6948 | killed | ReadDoubleDoesNotReadPastViewBounds |
| m88 | f701c57 | SURVIVED | none — predicted by the commit on x86 (CVTTSD2SI → INT_MIN path); needs the AArch64 leg → #139 |
| m89 | 9de7d2b | killed | StatesCountAboveUint32RangeThrows (test_hmm_stream_io killed after 196 s) |
| m90 | b501a57 | killed | 3 StudentT location tests |
| m91 | 7e7d90a | killed | 7 JSON sanitisation tests (scalar + MV) |
| m83 | d1ce078 .cpp + cmake | SURVIVED | none — expected on an AVX2+FMA host without AVX-512; the mask test below is the guard |
| m83mask | one bit dropped from kAvx512RequiredMask | killed | Avx512RequiredMaskExactValue |
| mbestofn | 6608bab fit_best_of_n.h | killed | NaNLogLikelihoodRestartIsDiscarded |
| mmvsumw | 6608bab three MV sumW guards | killed | SubnormalTotalWeightPreservesParameters — Diagonal and FullCov ONLY |
| m72 | — | not run | the #72 fix lives in the C++17-Bessel branch libc++ never compiles; GCC/MSVC legs cover it |
Suspected gap from mmvsumw: no test exercises the IndependentComponents half of the MV sumW guard (noted on #124).
Also new: the Discrete clone()-vs-cold-fill crash (C1/#115) reproduces through the public HMM API — clone() on a shared
Discrete-emission HMM while other threads run forward-backward crashed Release 3/3 (t7/hmm_clone_race.cpp).
T7 concurrency: concurrent training of distinct instances CLEAN (Release + TSan). 3 confirmed:
- C1 [crash] DiscreteDistribution::clone() (const) racing the first const evaluation on a cold shared instance:
  Release SIGABRT/SIGSEGV 5/5 runs at 1000 rounds; TSan: copy ctor reads cache vectors while the fill resizes
  them under its lock. basic_hmm.h says concurrent const evaluation is safe. Same defect as S3.
- C2 [concurrency] same race in Binomial on logFactorialCache_ (TSan; no Release crash in 3×20k rounds).
- C3 [concurrency] DiagonalGaussian copy ctor reads cached values without the source lock (operator= and
  FullCov copy ctor do lock). TSan; results still bit-exact.

## dist-numerics (T1/T2/T4) — 30 confirmed in 14 groups of ~14.4k raw hits (detail: hunt/dist-numerics/REPORT.md)
Coverage: T1 16 classes × 120 instances, batch sizes 1–40, 64–263, 1023–1025, 4000 triples per identity, MV 3000
instances D=1–12; T2 2389 adversarial + 2400 random rows vs mpmath dps 50, special functions and log/exp/log1p
kernels called per tier (scalar, SSE2, AVX2 symbols directly), 47 oracle rows timed out (logged); T4 20 instances ×
every edge × 11 sizes × every lane = 439k neighbour-lane checks + every setter. Bessel = Tier 2 (libc++).
Note: distributions DO expose getCumulativeProbability (the common brief said otherwise); CDF findings below.
- N1 [accuracy] gammap (Gamma/ChiSquared CDF): Gamma(1,1) F(1)=0.63212054878 vs 0.63212055883 (1e-8 rel);
  Gamma(1e5,1) F(100001) off 6.7e-5. Region x≈a only. VERIFIED.
- N2 [accuracy] NegBin CDF: NegBin(1,1e-8) F(99999999) = 1.0e-5 vs 0.632.
- N3 [accuracy+doc] Poisson CDF switches to a normal approximation at k>100 ∧ λ>100 (header says pmf sum):
  F(101) at λ≈100 0.5596 vs 0.5660; Poisson(700) F(101) = 0.
- N4 [accuracy] CDF lower tails flush to 0 via 1−upper: Gaussian F(−10)=0 (true 7.6e-24), also LogNormal,
  Exponential, Weibull, Rayleigh. Gamma F(1e300) θ=1e-150 → 0 (should be 1); Pareto(1e-6,1e-300) F(1e300) → 1 (0.00138). VERIFIED (Gaussian).
- N5 [contract] accepted tiny σ gives NaN: Gaussian(0,1e-160) x=0 scalar logpdf 367.49 but batch and
  getProbability NaN; LogNormal(0,1e-160) logpdf NaN at x=1; Rayleigh(1e-300) batch/pdf NaN. VERIFIED.
- N6 [accuracy] scalar wrong, batch right at extremes: Weibull(1e-6,1e300) x=1e-300 scalar off 1.0 nat;
  StudentT(1e300,1.5,1) x=6.5 scalar −345.96 vs −358.46.
- N7 [accuracy] large-shape cancellation: StudentT logpdf at mode off 8e-7 at ν=1e9, 3.08 nats at ν=1e15;
  NegBin r≥1e8 same. Reachable via ctor/setter/JSON (fit caps ν at 1000).
- N8 [accuracy+resource] Binomial: running-sum log-factorial table → logpmf error 3.4e-9 at n=2e5, pmf sums
  to 1+6.8e-9; log(1−p) not log1p(−p) → pmf(0)=1 exactly at p=1e-17; cache O(n): ~24 s and 4 GB at n=5e8 inside
  a noexcept call (overlaps F10).
- N9 [contract] getProbability 0/NaN where log value is ordinary: Pareto(261.8,0.0017973) P=0 (pdf 1.45e5);
  Beta(994,4904) x=0.3246 NaN.
- N10 [accuracy, extends #73] Bessel Tier 2 (every macOS build): log I0 off 4.7e-7 at κ≈97 (header claims
  <1.6e-7); 1−I1/I0 for x<30 rel err 8.5e-6 — the 52-ULP #73 bound holds on Tier 1 only.
- N11 [accuracy] digamma/trigamma asymptotic series stops early at the x=6 switch: trigamma(6) rel 1e-9,
  digamma(6) 8.8e-12 (header claims ~2e-14).
- N12 [contract] batch vs scalar on edges: x=NaN batch NaN vs scalar −inf in 7 classes (Beta lane-dependent);
  x=+inf batch NaN vs scalar −inf in 4; x=0 shape<1 scalar +inf vs batch −inf; MV NaN component: Diag/FullCov NaN,
  IC −inf; Beta/StudentT/NegBin accept DBL_MAX then NaN at ordinary x. No lane poisoning.
- N13 [contract] exp_batch SSE2+AVX2 returns +inf for x∈(709.436,709.783] in the vector body while the scalar
  tail of the same call is finite; log1p_inplace SIMD up to 3742 ulp on (1e-4,0.02) (abs ≤1e-16, minor).
- N14 [minor] Uniform(−DBL_MAX,DBL_MAX) accepted, CDF NaN; discrete classes treat non-integer x three ways.
Overlap: erfinv up to 0.0093 near y=1 adds numbers to #103; #77/#99 discarded as documented.
Guards proposed: 1 batch==scalar IEEE-edge × sizes {1,5,8,9,17} × lane table (CI; N12); 2 ~50 checked-in mpmath
values each for gammap/incompleteBeta/digamma/trigamma incl. x=a+1±ulp, x=6±ulp (CI; N1,N11); 3 CDF tail/range
table (CI; N2–N4); 4 extreme-parameter sweep "accepted parameter never yields NaN" (CI; N5,N6,N9,N12); 5 exp and
log1p_inplace per-tier ULP gates with lane sweep (per-machine; N13); 6 large-shape and Bessel Tier-2 mpmath
sweeps (per release; N7,N8,N10).

## sampler-moments-fit (T9/T10/T11) — 18 confirmed of ~120 raw hits (agent could not write REPORT.md; summary there)
- M1 [hang] VonMises::sample for κ∈[1e-9,1e-8]: 1000 draws exceed 10 s (κ≥3e-8 is <1 ms). VERIFIED (gtimeout 124).
- M2 [accuracy] Beta::sample returns exactly 0.5 when both gamma variates underflow: 22.5% at (0.001,0.001).
  VERIFIED: 22,555/1e5. Also ~2× too many exact zeros at shapes 0.003–0.01.
- M3 [contract] Beta draws exactly 1.0 (1.3% at (0.1,0.1), 34% at (0.01,0.01)) while logpdf(0 or 1) = −inf, so
  sample() output scores −inf on the same object. VERIFIED: at (0.001,0.001) 62,406/1e5 draws score −inf.
- M4 [accuracy] Rayleigh logpdf wrong for σ outside ~[1e-154,1e154]: σ=1e-155 scalar −inf (should be 356.40);
  σ=1e200 missing −x²/2σ²; batch NaN/−inf; CDF 0 or 1. (Overlaps N5.)
- M5 [contract] FullCovariance sample_mv/pdf/getLogDet use Σ+1e-5·I but getCovariance() returns Σ:
  Σ=1e-6·[[1,.5],[.5,1]] gives sample variance 1.1e-5; diag(1e4,1e-4) → 1.104e-4. Scale-dependent design issue.
- M6 [resource] Binomial ctor O(n) table: 6.5 s at n=1e8, ~17 GB at INT_MAX; validator accepts (= N8/F10).
- M7 [accuracy] Weibull::getVariance cancels at large k: rel 1.3e-10 (k=1e3), 8.9e-8 (1e5), 0.33 (1e8); NaN at
  tiny k where +inf is correct.
- M8 [accuracy] NegativeBinomial::getMode returns INT_MIN for NB(1e8,1e-3) (should be 99,899,999,000).
- M9 [accuracy] variance/sd 0 or inf via intermediates though representable: Beta(1e-300,1e-100) var 0 (1e-200);
  Beta(1e100,1e200) var 0 (1e-300); LogNormal(−700,19.1) var 0 (1.1e-290); sd 0 for Rayleigh 1e-300,
  Pareto xm=1e-200, StudentT scale 1e-200.
- M10 [accuracy] Weibull fit silently resets to (1,1) for data scale ≳1e6 or ≲1e-40; k=10, λ=1e-60 → k=107. (= F7.)
- M11 [accuracy] ChiSquared fit returns ν≈1.048e-4 for every ν≲0.4 (MLE 0.05/0.1/0.2/0.3; Gamma fit recovers).
- M12 [contract] Rayleigh fit leaves σ=0 (true 1e-200) or σ=inf (true ≥1e160) without throwing; weighted path
  resets to 1. Same overflow shape as #92 in another class. (= F2 Rayleigh.)
- M13 [accuracy] Gaussian fit: Σ(x−μ)² overflow (σ≥1e154) → σ set to the 1e-6 floor.
- M14 [contract, extends #92] StudentT fit() is one ECME step, not an MLE: from default StudentT(5), scale 1e3 →
  ν=0.15, σ=2.45; scale ≤1e-3 → ν=5.48 whatever the truth; scale ≥1e30 resets. (= F12 item.)
- M15 [accuracy] Beta fit misses the MLE at small shapes: Beta(0.02,3) → (0.0426,5.62) vs MLE (0.0201,2.885),
  0.36 nat/obs shortfall; Beta(0.1,1) ~5 SE off.
- M16 [contract] weights of 1e300: Poisson λ=inf installed; Gaussian σ→1e-6; Weibull/Gamma/NegBin/VonMises reset.
Side CDF findings (T2 territory, add to N1–N3): NegBin CDF 6.8e-61 for (100,0.01) (true 0.999999), 0 for (1e6,0.5),
0.0037 for (5,0.001); Gamma err 3.6e-5 at k=1e6; Poisson err 2.1e-3 at λ=1000; VonMises CDF (512-step
trapezoid) 0.084 where 0.99 at κ=1e6.
Suspected: GOLD_STANDARD_CHECKLIST calls Binomial Tier A while n̂=max(x) is documented MOM-ish.
Guards proposed: 1 von Mises sampler under watchdog at κ∈{1e-9,2e-9,1e-8} (CI); 2 fit scale-equivariance for
Weibull/Rayleigh/Gaussian/StudentT/ChiSquared at ×1e±6, ×1e±30 (CI); 3 ChiSquared and Beta fits vs stored MLE at
small shapes (CI); 4 logpdf(sample()) > −inf for Beta small shapes and Rayleigh σ=1e±200 (CI); 5 Beta sampler
exact-0.5 fraction = 0 and zero fraction at shape 0.001 (CI); 6 FullCov sample covariance == getCovariance()
two-sided at Σ=1e-6 (CI); 7 accessor sweep vs mpmath tables (per release); 8 binned chi-square GoF harness
(per machine, per release).
