#include "ivf_io.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ivf {

static_assert(sizeof(std::size_t) >= 8 && sizeof(float) == 4 &&
              std::numeric_limits<float>::is_iec559 &&
              __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

template <typename T>
Matrix<T> read_matrix(const std::string& path, bool vecs) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    const auto end = file.tellg();
    if (end < (vecs ? 4 : 8)) throw std::runtime_error(path + ": missing header");
    const auto bytes = static_cast<std::size_t>(end);
    file.seekg(0);
    auto read = [&](auto* data, std::size_t count) {
        if (!file.read(reinterpret_cast<char*>(data), count * sizeof(*data))) {
            throw std::runtime_error(path + ": incomplete read");
        }
    };

    std::uint32_t shape[2];
    read(shape, vecs ? 1 : 2);
    const std::size_t dim = vecs ? shape[0] : shape[1];
    if (dim == 0 || dim > INT_MAX) throw std::runtime_error(path + ": invalid dimension");
    const auto row_bytes = 4 + dim * sizeof(T);
    const std::size_t rows = vecs ? bytes / row_bytes : shape[0];
    const bool valid_size = vecs ? bytes % row_bytes == 0
        : (bytes - 8) % sizeof(T) == 0 && (bytes - 8) / sizeof(T) == rows * dim;
    if (rows == 0 || !valid_size) throw std::runtime_error(path + ": size mismatch");

    Matrix<T> result{rows, dim, std::vector<T>(rows * dim)};
    if (vecs) {
        file.seekg(0);
        for (std::size_t i = 0; i < rows; ++i) {
            std::uint32_t row_dim;
            read(&row_dim, 1);
            if (row_dim != dim) throw std::runtime_error(path + ": inconsistent dimension");
            read(result.data.data() + i * dim, dim);
        }
    } else {
        read(result.data.data(), result.data.size());
    }
    if constexpr (std::is_same_v<T, float>) {
        for (float value : result.data) {
            if (!std::isfinite(value)) throw std::runtime_error(path + ": non-finite coordinate");
        }
    }
    return result;
}

FloatMatrix read_fbin(const std::string& path) { return read_matrix<float>(path, false); }
IntMatrix read_ibin(const std::string& path) { return read_matrix<std::int32_t>(path, false); }
FloatMatrix read_fvecs(const std::string& path) { return read_matrix<float>(path, true); }
IntMatrix read_ivecs(const std::string& path) { return read_matrix<std::int32_t>(path, true); }

IVFIndex read_ivf(const std::string& base_path, const std::string& centers_path,
                  const std::string& clusters_path) {
    const auto base = read_fbin(base_path);
    auto centers = read_fvecs(centers_path);
    const auto clusters = read_ivecs(clusters_path);
    if (base.rows > INT_MAX || centers.rows > INT_MAX || centers.dim != base.dim) {
        throw std::runtime_error("incompatible IVF dimensions or ID range");
    }
    if (clusters.rows != base.rows || clusters.dim != 1) {
        throw std::runtime_error(clusters_path + ": expected N rows with one cluster ID each");
    }

    IVFIndex index;
    index.size = base.rows;
    index.dim = base.dim;
    index.nlist = centers.rows;
    index.list_ptr.resize(index.nlist + 1);
    for (auto list : clusters.data) {
        if (list < 0 || std::size_t(list) >= index.nlist) {
            throw std::runtime_error(clusters_path + ": cluster ID out of range");
        }
        ++index.list_ptr[list + 1];
    }
    std::partial_sum(index.list_ptr.begin(), index.list_ptr.end(), index.list_ptr.begin());
    auto next = index.list_ptr;
    index.ids.resize(index.size);
    index.vectors.resize(base.data.size());
    for (std::size_t id = 0; id < index.size; ++id) {
        const auto pos = next[clusters.data[id]]++;
        index.ids[pos] = static_cast<std::int32_t>(id);
        std::copy_n(base.row(id), index.dim, index.vectors.data() + pos * index.dim);
    }
    index.centroids = std::move(centers.data);
    return index;
}

}
