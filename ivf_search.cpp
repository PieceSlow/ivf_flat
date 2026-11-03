#include "ivf_search.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ivf {

bool better(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

TopK::TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

void TopK::add(Neighbor value) {
    if (heap_.size() < k_) {
        heap_.push_back(value);
        std::push_heap(heap_.begin(), heap_.end(), better);
    } else if (better(value, heap_.front())) {
        std::pop_heap(heap_.begin(), heap_.end(), better);
        heap_.back() = value;
        std::push_heap(heap_.begin(), heap_.end(), better);
    }
}

void TopK::finish(Neighbor* output) {
    std::sort_heap(heap_.begin(), heap_.end(), better);
    std::copy(heap_.begin(), heap_.end(), output);
    std::fill(output + heap_.size(), output + k_, Neighbor{});
}

double l2_distance_scalar(const float* a, const float* b, std::size_t dim) {
    double sum = 0;
    for (std::size_t j = 0; j < dim; ++j) {
        const double delta = double(a[j]) - double(b[j]);
        sum += delta * delta;
    }
    return sum;
}

std::vector<std::uint32_t> select_lists(const IVFIndex& index, const float* query,
                                      std::size_t nprobe) {
    std::vector<Neighbor> centers(index.nlist);
    for (std::size_t l = 0; l < index.nlist; ++l) {
        centers[l] = {l2_distance_scalar(query, index.centroid(l), index.dim),
                      static_cast<std::int32_t>(l)};
    }
    std::partial_sort(centers.begin(), centers.begin() + nprobe, centers.end(), better);
    std::vector<std::uint32_t> lists(nprobe);
    for (std::size_t i = 0; i < nprobe; ++i) lists[i] = centers[i].id;
    return lists;
}

void search_ivf_serial(const IVFIndex& index, const float* queries, std::size_t count,
                       const SearchOptions& options, Neighbor* output) {
    for (std::size_t q = 0; q < count; ++q) {
        const float* query = queries + q * index.dim;
        const auto lists = select_lists(index, query, options.nprobe);
        TopK best(options.k);
        for (auto list : lists) {
            for (std::size_t p = index.list_ptr[list]; p < index.list_ptr[list + 1]; ++p) {
                best.add({l2_distance_scalar(query, index.vector(p), index.dim), index.ids[p]});
            }
        }
        best.finish(output + q * options.k);
    }
}

void search_exact_serial(const IVFIndex& index, const float* queries, std::size_t count,
                         const SearchOptions& options, Neighbor* output) {
    for (std::size_t q = 0; q < count; ++q) {
        TopK best(options.k);
        for (std::size_t p = 0; p < index.size; ++p) {
            best.add({l2_distance_scalar(queries + q * index.dim, index.vector(p), index.dim),
                      index.ids[p]});
        }
        best.finish(output + q * options.k);
    }
}

void check_result(const std::vector<Neighbor>& actual,
                  const std::vector<Neighbor>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("result size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto& a = actual[i];
        const auto& e = expected[i];
        const bool same_distance = e.id == -1
            ? a.distance == std::numeric_limits<double>::infinity()
            : std::isfinite(a.distance) &&
              std::abs(a.distance - e.distance) <= 1e-10 + 1e-10 * std::abs(e.distance);
        if (a.id != e.id || !same_distance) {
            throw std::runtime_error("correctness failure at output " + std::to_string(i) +
                                     ": expected ID " + std::to_string(e.id) +
                                     ", got " + std::to_string(a.id));
        }
    }
}

double recall_at_k(const std::vector<Neighbor>& result,
                   const IntMatrix& groundtruth, std::size_t count, std::size_t k) {
    std::size_t hits = 0;
    for (std::size_t q = 0; q < count; ++q) {
        const auto* truth = groundtruth.row(q);
        for (std::size_t j = 0; j < k; ++j) {
            const auto id = result[q * k + j].id;
            if (id != -1 && std::find(truth, truth + k, id) != truth + k) ++hits;
        }
    }
    return double(hits) / double(count * k);
}

}
