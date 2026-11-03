#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ivf {

template <typename T>
struct Matrix {
    std::size_t rows = 0;
    std::size_t dim = 0;
    std::vector<T> data;

    const T* row(std::size_t i) const { return data.data() + i * dim; }
};

using FloatMatrix = Matrix<float>;
using IntMatrix = Matrix<std::int32_t>;

FloatMatrix read_fbin(const std::string& path);
IntMatrix read_ibin(const std::string& path);
FloatMatrix read_fvecs(const std::string& path);
IntMatrix read_ivecs(const std::string& path);

struct IVFIndex {
    std::size_t size = 0;
    std::size_t dim = 0;
    std::size_t nlist = 0;
    std::vector<std::size_t> list_ptr;
    std::vector<std::int32_t> ids;
    std::vector<float> vectors;
    std::vector<float> centroids;

    const float* vector(std::size_t pos) const {
        return vectors.data() + pos * dim;
    }
    const float* centroid(std::size_t list) const {
        return centroids.data() + list * dim;
    }
};

IVFIndex read_ivf(const std::string& base_fbin,
                  const std::string& centroids_fvecs,
                  const std::string& cluster_ids_ivecs);

}
