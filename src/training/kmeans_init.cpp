#include "libhmm/training/kmeans_init.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace libhmm {

namespace {

/// Squared Euclidean distance between a row-view and a centroid vector.
double sq_dist(ObservationVectorView x, const std::vector<double>& c) noexcept {
    double d = 0.0;
    const std::size_t D = x.size();
    for (std::size_t i = 0; i < D; ++i) {
        const double diff = x[i] - c[i];
        d += diff * diff;
    }
    return d;
}

/// Minimum squared distance from x to the nearest of the first @p n_chosen centroids.
double min_sq_dist(ObservationVectorView x,
                   const std::vector<std::vector<double>>& centroids,
                   std::size_t n_chosen) noexcept {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n_chosen; ++j) {
        const double d = sq_dist(x, centroids[j]);
        if (d < best) best = d;
    }
    return best;
}

} // namespace

void kmeans_init(HmmMV& hmm, const MultiObservationLists& data,
                 std::mt19937_64& rng, std::size_t max_iter)
{
    if (data.empty()) {
        throw std::invalid_argument("kmeans_init: data must be non-empty");
    }
    const std::size_t K = static_cast<std::size_t>(hmm.getNumStates());
    if (K == 0) {
        throw std::invalid_argument("kmeans_init: HMM must have at least one state");
    }

    // Collect all observation row-views into a flat vector.
    std::vector<ObservationVectorView> pts;
    for (const auto& seq : data) {
        const std::size_t T = seq.size1();
        for (std::size_t t = 0; t < T; ++t) {
            pts.push_back(row_view(seq, t));
        }
    }
    if (pts.empty()) {
        throw std::invalid_argument("kmeans_init: data contains no observations");
    }
    const std::size_t D = pts[0].size();
    if (D == 0) {
        throw std::invalid_argument("kmeans_init: observation dimensionality must be > 0");
    }
    const std::size_t M = pts.size();

    // -------------------------------------------------------------------------
    // k-means++ seeding
    // -------------------------------------------------------------------------
    std::vector<std::vector<double>> centroids(K, std::vector<double>(D, 0.0));

    // First centroid: uniform random.
    {
        std::uniform_int_distribution<std::size_t> uni(0, M - 1);
        const std::size_t idx = uni(rng);
        for (std::size_t d = 0; d < D; ++d) centroids[0][d] = pts[idx][d];
    }

    // Subsequent centroids: proportional to min-distance-squared.
    std::vector<double> dists(M);
    for (std::size_t k = 1; k < K; ++k) {
        double total = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            dists[i] = min_sq_dist(pts[i], centroids, k);
            total += dists[i];
        }
        if (total <= 0.0) {
            // All points coincide with existing centroids; just reuse any point.
            std::uniform_int_distribution<std::size_t> uni(0, M - 1);
            const std::size_t idx = uni(rng);
            for (std::size_t d = 0; d < D; ++d) centroids[k][d] = pts[idx][d];
        } else {
            std::uniform_real_distribution<double> u(0.0, total);
            const double threshold = u(rng);
            double cumulative = 0.0;
            std::size_t chosen = M - 1;
            for (std::size_t i = 0; i < M; ++i) {
                cumulative += dists[i];
                if (cumulative >= threshold) { chosen = i; break; }
            }
            for (std::size_t d = 0; d < D; ++d) centroids[k][d] = pts[chosen][d];
        }
    }

    // -------------------------------------------------------------------------
    // Lloyd's iterations
    // -------------------------------------------------------------------------
    std::vector<std::size_t> assign(M);
    std::vector<std::size_t> counts(K);

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Assignment step.
        bool changed = false;
        for (std::size_t i = 0; i < M; ++i) {
            std::size_t best = 0;
            double bestD = sq_dist(pts[i], centroids[0]);
            for (std::size_t k = 1; k < K; ++k) {
                const double d = sq_dist(pts[i], centroids[k]);
                if (d < bestD) { bestD = d; best = k; }
            }
            if (assign[i] != best) { assign[i] = best; changed = true; }
        }

        if (!changed && iter > 0) break;  // converged

        // Update step: recompute centroids from assigned points.
        std::fill(counts.begin(), counts.end(), 0);
        for (auto& c : centroids) std::fill(c.begin(), c.end(), 0.0);

        for (std::size_t i = 0; i < M; ++i) {
            const std::size_t k = assign[i];
            ++counts[k];
            for (std::size_t d = 0; d < D; ++d) centroids[k][d] += pts[i][d];
        }
        for (std::size_t k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                const double inv = 1.0 / static_cast<double>(counts[k]);
                for (std::size_t d = 0; d < D; ++d) centroids[k][d] *= inv;
            }
            // Empty cluster: centroid unchanged (retain previous value).
        }
    }

    // -------------------------------------------------------------------------
    // Initialise each state's emission distribution from its cluster members.
    // -------------------------------------------------------------------------
    std::vector<std::vector<ObservationVectorView>> cluster_pts(K);
    for (std::size_t i = 0; i < M; ++i) {
        cluster_pts[assign[i]].push_back(pts[i]);
    }

    for (std::size_t k = 0; k < K; ++k) {
        auto& dist = hmm.getDistribution(k);
        if (cluster_pts[k].empty()) {
            dist.reset();
        } else {
            dist.fit(std::span<const ObservationVectorView>(
                cluster_pts[k].data(), cluster_pts[k].size()));
        }
    }
}

} // namespace libhmm
