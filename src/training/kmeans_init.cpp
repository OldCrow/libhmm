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
double sq_dist(ObservationVectorView x, const std::vector<double> &c) noexcept {
    double d = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double diff = x[i] - c[i];
        d += diff * diff;
    }
    return d;
}

/// Minimum sq-distance from x to the first @p n_chosen already-chosen centroids.
double min_sq_dist(ObservationVectorView x, const std::vector<std::vector<double>> &centroids,
                   std::size_t n_chosen) noexcept {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n_chosen; ++j) {
        const double d = sq_dist(x, centroids[j]);
        if (d < best)
            best = d;
    }
    return best;
}

/// k-means++ seeding: choose K initial centroids from @p pts using @p rng.
/// @throws std::invalid_argument if pts.size() < K (cannot seed K distinct centroids).
std::vector<std::vector<double>> seed_kmeanspp(const std::vector<ObservationVectorView> &pts,
                                               std::size_t K, std::size_t D, std::mt19937_64 &rng) {
    const std::size_t M = pts.size();
    if (M < K)
        throw std::invalid_argument("kmeans++: cannot initialize " + std::to_string(K) +
                                    " centroids from " + std::to_string(M) +
                                    " observations (need at least K observations)");
    std::vector<std::vector<double>> centroids(K, std::vector<double>(D, 0.0));

    std::uniform_int_distribution<std::size_t> uni(0, M - 1);
    const std::size_t first = uni(rng);
    for (std::size_t d = 0; d < D; ++d)
        centroids[0][d] = pts[first][d];

    std::vector<double> dists(M);
    for (std::size_t k = 1; k < K; ++k) {
        double total = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            dists[i] = min_sq_dist(pts[i], centroids, k);
            total += dists[i];
        }
        std::size_t chosen = M - 1;
        if (total > 0.0) {
            std::uniform_real_distribution<double> u(0.0, total);
            const double thr = u(rng);
            double cumul = 0.0;
            for (std::size_t i = 0; i < M; ++i) {
                cumul += dists[i];
                if (cumul >= thr) {
                    chosen = i;
                    break;
                }
            }
        }
        for (std::size_t d = 0; d < D; ++d)
            centroids[k][d] = pts[chosen][d];
    }
    return centroids;
}

/// One Lloyd's assignment pass. Returns true if any assignment changed.
bool lloyd_assign(const std::vector<ObservationVectorView> &pts,
                  const std::vector<std::vector<double>> &centroids,
                  std::vector<std::size_t> &assign) {
    const std::size_t K = centroids.size();
    bool changed = false;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        std::size_t best = 0;
        double bestD = sq_dist(pts[i], centroids[0]);
        for (std::size_t k = 1; k < K; ++k) {
            const double d = sq_dist(pts[i], centroids[k]);
            if (d < bestD) {
                bestD = d;
                best = k;
            }
        }
        if (assign[i] != best) {
            assign[i] = best;
            changed = true;
        }
    }
    return changed;
}

/// One Lloyd's update pass: recompute centroids from current assignments.
/// Empty clusters retain their previous centroid rather than collapsing to zero.
void lloyd_update(const std::vector<ObservationVectorView> &pts,
                  const std::vector<std::size_t> &assign, std::size_t D,
                  std::vector<std::vector<double>> &centroids) {
    const std::size_t K = centroids.size();
    // Save centroids before zeroing so empty clusters can retain theirs.
    const auto prev_centroids = centroids;
    std::vector<std::size_t> counts(K, 0);
    for (auto &c : centroids)
        std::fill(c.begin(), c.end(), 0.0);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const std::size_t k = assign[i];
        ++counts[k];
        for (std::size_t d = 0; d < D; ++d)
            centroids[k][d] += pts[i][d];
    }
    for (std::size_t k = 0; k < K; ++k) {
        if (counts[k] > 0) {
            const double inv = 1.0 / static_cast<double>(counts[k]);
            for (std::size_t d = 0; d < D; ++d)
                centroids[k][d] *= inv;
        } else {
            // Empty cluster: restore the previous centroid so the cluster
            // is not collapsed to the zero vector.
            centroids[k] = prev_centroids[k];
        }
    }
}

/// Fit each HMM state's emission from its assigned cluster members.
void fit_clusters(HmmMV &hmm, const std::vector<ObservationVectorView> &pts,
                  const std::vector<std::size_t> &assign, std::size_t K) {
    std::vector<std::vector<ObservationVectorView>> cluster_pts(K);
    for (std::size_t i = 0; i < pts.size(); ++i)
        cluster_pts[assign[i]].push_back(pts[i]);
    for (std::size_t k = 0; k < K; ++k) {
        auto &dist = hmm.getDistribution(k);
        if (cluster_pts[k].empty()) {
            dist.reset();
        } else {
            dist.fit(std::span<const ObservationVectorView>(cluster_pts[k].data(),
                                                            cluster_pts[k].size()));
        }
    }
}

} // namespace

void kmeans_init(HmmMV &hmm, const MultiObservationLists &data, std::mt19937_64 &rng,
                 std::size_t max_iter) {
    if (data.empty())
        throw std::invalid_argument("kmeans_init: data must be non-empty");
    const std::size_t K = static_cast<std::size_t>(hmm.getNumStates());
    if (K == 0)
        throw std::invalid_argument("kmeans_init: HMM must have at least one state");

    // Flatten all sequences into a single view vector.
    std::vector<ObservationVectorView> pts;
    for (const auto &seq : data)
        for (std::size_t t = 0; t < seq.size1(); ++t)
            pts.push_back(row_view(seq, t));

    if (pts.empty())
        throw std::invalid_argument("kmeans_init: data contains no observations");
    const std::size_t D = pts[0].size();
    if (D == 0)
        throw std::invalid_argument("kmeans_init: observation dimensionality must be > 0");

    auto centroids = seed_kmeanspp(pts, K, D, rng);

    std::vector<std::size_t> assign(pts.size(), 0);
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        const bool changed = lloyd_assign(pts, centroids, assign);
        lloyd_update(pts, assign, D, centroids);
        if (!changed && iter > 0)
            break;
    }

    fit_clusters(hmm, pts, assign, K);
}

} // namespace libhmm
