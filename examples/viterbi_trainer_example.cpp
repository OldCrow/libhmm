/**
 * viterbi_trainer_example — ViterbiTrainer with TrainingConfig presets.
 *
 * ViterbiTrainer performs hard-assignment training: at each iteration it
 * runs the Viterbi algorithm to find the most likely state sequence, then
 * re-estimates parameters from those hard assignments.
 *
 * This example shows:
 *   1. The three built-in presets: fast, balanced, precise.
 *   2. Convergence flags reported after training.
 *   3. How to use a custom TrainingConfig for specific requirements.
 *   4. Segmental K-means as an alternative initialisation strategy.
 */
#include <iostream>
#include <iomanip>
#include <memory>
#include "libhmm/libhmm.h"

using namespace libhmm;

static std::unique_ptr<Hmm> make_3state_hmm() {
    auto hmm = std::make_unique<Hmm>(3);

    Matrix trans(3, 3);
    trans(0, 0) = 0.7;
    trans(0, 1) = 0.2;
    trans(0, 2) = 0.1;
    trans(1, 0) = 0.1;
    trans(1, 1) = 0.8;
    trans(1, 2) = 0.1;
    trans(2, 0) = 0.1;
    trans(2, 1) = 0.1;
    trans(2, 2) = 0.8;
    hmm->setTrans(trans);

    Vector pi(3);
    pi(0) = 0.5;
    pi(1) = 0.3;
    pi(2) = 0.2;
    hmm->setPi(pi);

    hmm->setDistribution(0, std::make_unique<GaussianDistribution>(0.0, 1.5));
    hmm->setDistribution(1, std::make_unique<GaussianDistribution>(5.0, 1.5));
    hmm->setDistribution(2, std::make_unique<GaussianDistribution>(10.0, 1.5));
    return hmm;
}

static ObservationLists make_obs() {
    ObservationLists obs;
    ObservationSet s(30);
    for (std::size_t i = 0; i < 10; ++i)
        s(i) = static_cast<double>(i % 3) * 0.5;
    for (std::size_t i = 10; i < 20; ++i)
        s(i) = 5.0 + static_cast<double>(i % 3) * 0.5;
    for (std::size_t i = 20; i < 30; ++i)
        s(i) = 10.0 + static_cast<double>(i % 3) * 0.5;
    obs.push_back(s);
    return obs;
}

static void run_and_report(const char *label, ViterbiTrainer &trainer) {
    trainer.train();
    std::cout << "  " << std::left << std::setw(14) << label
              << "converged=" << (trainer.hasConverged() ? "yes" : "no ")
              << "  max_iter=" << (trainer.reachedMaxIterations() ? "yes" : "no") << "\n";
}

int main() {
    std::cout << "ViterbiTrainer: TrainingConfig Presets\n";
    std::cout << "=======================================\n\n";

    const auto obs = make_obs();

    // -------------------------------------------------------------------------
    // 1. Built-in presets
    // -------------------------------------------------------------------------
    std::cout << "Built-in presets (same initial HMM, different tolerances):\n";

    {
        auto hmm = make_3state_hmm();
        ViterbiTrainer trainer(hmm.get(), obs, training_presets::fast());
        run_and_report("fast:", trainer);
    }
    {
        auto hmm = make_3state_hmm();
        ViterbiTrainer trainer(hmm.get(), obs, training_presets::balanced());
        run_and_report("balanced:", trainer);
    }
    {
        auto hmm = make_3state_hmm();
        ViterbiTrainer trainer(hmm.get(), obs, training_presets::precise());
        run_and_report("precise:", trainer);
    }

    // -------------------------------------------------------------------------
    // 2. Custom config
    // -------------------------------------------------------------------------
    std::cout << "\nCustom config (tight tolerance, large window):\n";
    {
        auto hmm = make_3state_hmm();
        TrainingConfig cfg;
        cfg.convergenceTolerance = 1e-9;
        cfg.maxIterations = 500;
        cfg.convergenceWindow = 5;
        ViterbiTrainer trainer(hmm.get(), obs, cfg);
        run_and_report("custom:", trainer);
    }

    // -------------------------------------------------------------------------
    // 3. Preset comparison: learned parameters
    // -------------------------------------------------------------------------
    std::cout << "\nLearned means after balanced training (target: 0, 5, 10):\n";
    {
        auto hmm = make_3state_hmm();
        ViterbiTrainer trainer(hmm.get(), obs, training_presets::balanced());
        trainer.train();

        for (int s = 0; s < 3; ++s) {
            const auto &d = static_cast<const GaussianDistribution &>(hmm->getDistribution(s));
            std::cout << "  State " << s << ": μ=" << std::fixed << std::setprecision(3)
                      << d.getMean() << "  σ=" << d.getStandardDeviation() << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // 4. SegmentalKMeansTrainer as an alternative initialiser
    // -------------------------------------------------------------------------
    std::cout << "\nSegmental K-means (discrete HMM, for comparison):\n";
    {
        Hmm hmm_disc(2);
        Matrix t2(2, 2);
        t2(0, 0) = 0.8;
        t2(0, 1) = 0.2;
        t2(1, 0) = 0.3;
        t2(1, 1) = 0.7;
        hmm_disc.setTrans(t2);
        Vector pi2(2);
        pi2(0) = 0.5;
        pi2(1) = 0.5;
        hmm_disc.setPi(pi2);

        auto d0 = std::make_unique<DiscreteDistribution>(6);
        for (int i = 0; i < 6; ++i)
            d0->setProbability(i, 1.0 / 6.0);
        hmm_disc.setDistribution(0, std::move(d0));

        auto d1 = std::make_unique<DiscreteDistribution>(6);
        for (int i = 0; i < 5; ++i)
            d1->setProbability(i, 0.1);
        d1->setProbability(5, 0.5);
        hmm_disc.setDistribution(1, std::move(d1));

        ObservationLists disc_obs;
        ObservationSet ds(12);
        for (std::size_t i = 0; i < 12; ++i)
            ds(i) = i % 6;
        disc_obs.push_back(ds);

        SegmentalKMeansTrainer km(&hmm_disc, disc_obs);
        std::cout << "  SegmentalKMeansTrainer constructed, isTerminated="
                  << (km.isTerminated() ? "yes" : "no") << "\n";
        std::cout << "  (Use for initialisation before Baum-Welch or Viterbi training)\n";
    }

    return 0;
}
