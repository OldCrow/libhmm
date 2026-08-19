#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "libhmm/distributions/diagonal_gaussian_distribution.h"
#include "libhmm/distributions/gaussian_distribution.h"
#include "libhmm/hmm.h"
#include "libhmm/training/baum_welch_trainer.h"

using namespace libhmm;

// ---------------------------------------------------------------------------
// Thread-safety contract (issue #48 decision, 2026-08-19): the library owns
// no threads. A shared instance supports concurrent const evaluation (the
// distribution caches use a mutex-serialised double-checked fill), but any
// mutation — setters, fit(), training — must not overlap with other access.
// What this test pins down under TSan in CI is the stronger everyday case,
// concurrent TRAINING of DISTINCT model instances:
//   - each thread owns its own BasicHmm (and therefore its own distributions
//     and their per-object caches);
//   - the observation data is shared read-only across threads (trainers hold
//     it by const reference and never write it);
//   - process-wide state is limited to function-local statics with
//     thread-safe C++11 initialisation (the DoubleVecOps dispatch table and
//     the CPUID feature flags), which threads may race to first-touch.
//
// This is the contract caller-level parallelism needs: concurrent restarts,
// model-selection sweeps, or cross-validation folds, each on its own model.
//
// gtest assertions are not thread-safe on all platforms, so worker threads
// only record results; all EXPECTs run on the main thread after join.
// ---------------------------------------------------------------------------

namespace {

struct WorkerResult {
    bool threw = false;
    std::string what;
    double firstLogL = 0.0;
    double lastLogL = 0.0;
};

// Deterministic bimodal scalar data, shared read-only by all scalar workers.
ObservationLists make_scalar_data() {
    constexpr double jitter[5] = {-0.4, -0.1, 0.15, 0.3, 0.05};
    ObservationLists lists;
    for (int s = 0; s < 4; ++s) {
        ObservationSet seq(40);
        for (std::size_t t = 0; t < 40; ++t) {
            const double centre = ((t + static_cast<std::size_t>(s)) % 2 == 0) ? 0.0 : 6.0;
            seq(t) = centre + jitter[(t * 2 + static_cast<std::size_t>(s)) % 5];
        }
        lists.push_back(seq);
    }
    return lists;
}

Hmm make_scalar_model(double meanOffset) {
    Hmm hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.6;
    trans(0, 1) = 0.4;
    trans(1, 0) = 0.4;
    trans(1, 1) = 0.6;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    // Per-thread offsets give each worker a distinct optimisation path.
    hmm.setDistribution(0, std::make_unique<GaussianDistribution>(1.0 + meanOffset, 2.0));
    hmm.setDistribution(1, std::make_unique<GaussianDistribution>(5.0 - meanOffset, 2.0));
    return hmm;
}

MultiObservationLists make_mv_data() {
    constexpr double jitter[5] = {-0.3, 0.0, 0.2, -0.15, 0.35};
    MultiObservationLists lists;
    for (int s = 0; s < 2; ++s) {
        ObservationMatrix m(30, 2);
        for (std::size_t t = 0; t < 30; ++t) {
            const double cx = (t % 2 == 0) ? -3.0 : 3.0;
            m(t, 0) = cx + jitter[(t + static_cast<std::size_t>(s)) % 5];
            m(t, 1) = -cx + jitter[(t * 3 + static_cast<std::size_t>(s)) % 5];
        }
        lists.push_back(m);
    }
    return lists;
}

HmmMV make_mv_model() {
    HmmMV hmm(2);
    Matrix trans(2, 2);
    trans(0, 0) = 0.5;
    trans(0, 1) = 0.5;
    trans(1, 0) = 0.5;
    trans(1, 1) = 0.5;
    hmm.setTrans(trans);
    Vector pi(2);
    pi(0) = 0.5;
    pi(1) = 0.5;
    hmm.setPi(pi);
    hmm.setDistribution(0, std::make_unique<DiagonalGaussianDistribution>(2, -1.0, 4.0));
    hmm.setDistribution(1, std::make_unique<DiagonalGaussianDistribution>(2, 1.0, 4.0));
    return hmm;
}

} // namespace

TEST(ConcurrentTrainingTest, DistinctInstancesTrainConcurrently) {
    constexpr int kScalarWorkers = 4;
    constexpr int kMvWorkers = 2;
    constexpr int kIterations = 8;

    const ObservationLists scalarData = make_scalar_data();
    const MultiObservationLists mvData = make_mv_data();

    std::vector<WorkerResult> results(kScalarWorkers + kMvWorkers);
    std::vector<std::thread> workers;
    workers.reserve(results.size());

    // Spin-latch so every worker enters train() as close to simultaneously
    // as possible — the first calls race to initialise the DoubleVecOps
    // dispatch table and the CPUID flags, which is exactly the window TSan
    // should inspect.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    const int total = kScalarWorkers + kMvWorkers;

    for (int w = 0; w < kScalarWorkers; ++w) {
        workers.emplace_back([&, w] {
            WorkerResult &r = results[static_cast<std::size_t>(w)];
            try {
                Hmm hmm = make_scalar_model(0.25 * w);
                BaumWelchTrainer trainer(hmm, scalarData);
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {
                }
                for (int it = 0; it < kIterations; ++it) {
                    trainer.train();
                    if (it == 0)
                        r.firstLogL = trainer.getLastLogProbability();
                }
                r.lastLogL = trainer.getLastLogProbability();
            } catch (const std::exception &e) {
                r.threw = true;
                r.what = e.what();
            }
        });
    }
    for (int w = 0; w < kMvWorkers; ++w) {
        workers.emplace_back([&, w] {
            WorkerResult &r = results[static_cast<std::size_t>(kScalarWorkers + w)];
            try {
                HmmMV hmm = make_mv_model();
                BasicBaumWelchTrainer<ObservationVectorView> trainer(hmm, mvData);
                ready.fetch_add(1);
                while (!go.load(std::memory_order_acquire)) {
                }
                for (int it = 0; it < kIterations; ++it) {
                    trainer.train();
                    if (it == 0)
                        r.firstLogL = trainer.getLastLogProbability();
                }
                r.lastLogL = trainer.getLastLogProbability();
            } catch (const std::exception &e) {
                r.threw = true;
                r.what = e.what();
            }
        });
    }

    while (ready.load() < total) {
    }
    go.store(true, std::memory_order_release);
    for (auto &t : workers)
        t.join();

    for (std::size_t i = 0; i < results.size(); ++i) {
        const WorkerResult &r = results[i];
        EXPECT_FALSE(r.threw) << "worker " << i << " threw: " << r.what;
        EXPECT_TRUE(std::isfinite(r.firstLogL)) << "worker " << i;
        EXPECT_TRUE(std::isfinite(r.lastLogL)) << "worker " << i;
        // EM monotonicity per worker; small tolerance for FP rounding.
        EXPECT_GE(r.lastLogL, r.firstLogL - 1e-6) << "worker " << i;
    }

    // Determinism cross-check: concurrent training must equal serial
    // training of the same model on the same data — threads share nothing
    // mutable, so results are not merely race-free but identical.
    for (int w = 0; w < kScalarWorkers; ++w) {
        Hmm hmm = make_scalar_model(0.25 * w);
        BaumWelchTrainer trainer(hmm, scalarData);
        for (int it = 0; it < kIterations; ++it)
            trainer.train();
        EXPECT_EQ(trainer.getLastLogProbability(), results[static_cast<std::size_t>(w)].lastLogL)
            << "worker " << w << " diverged from its serial twin";
    }
}
