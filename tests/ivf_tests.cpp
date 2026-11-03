#include "ivf_search.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("accepted invalid input: " + message);
}

template <typename T>
void write_matrix(const fs::path& path, const std::vector<T>& data,
                  std::uint32_t rows, std::uint32_t dim, bool vecs) {
    std::ofstream out(path, std::ios::binary);
    if (vecs) {
        for (std::size_t i = 0; i < rows; ++i) {
            out.write(reinterpret_cast<const char*>(&dim), 4);
            out.write(reinterpret_cast<const char*>(data.data() + i * dim), dim * sizeof(T));
        }
    } else {
        const std::uint32_t shape[]{rows, dim};
        out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    }
    if (!out) throw std::runtime_error("failed writing test fixture");
}

void test(const fs::path& dir) {
    const std::vector<float> floats{1, 2, 3, 4, 5, 6};
    const std::vector<std::int32_t> ints{4, 2, 0, 3, 1, 2};
    write_matrix(dir / "x.fbin", floats, 2, 3, false);
    write_matrix(dir / "x.fvecs", floats, 2, 3, true);
    write_matrix(dir / "gt.ibin", ints, 2, 3, false);
    write_matrix(dir / "gt.ivecs", ints, 2, 3, true);
    const auto fbin = ivf::read_fbin((dir / "x.fbin").string());
    const auto fvecs = ivf::read_fvecs((dir / "x.fvecs").string());
    require(fbin.rows == 2 && fbin.dim == 3 && fbin.data == floats && fvecs.data == floats,
            "float matrix decoding");
    require(ivf::read_ibin((dir / "gt.ibin").string()).data == ints &&
            ivf::read_ivecs((dir / "gt.ivecs").string()).data == ints, "integer matrix decoding");

    fs::resize_file(dir / "x.fbin", fs::file_size(dir / "x.fbin") - 1);
    rejects([&] { ivf::read_fbin((dir / "x.fbin").string()); }, "truncated fbin");
    write_matrix(dir / "bad.fbin", std::vector<float>{1}, 2, 3, false);
    rejects([&] { ivf::read_fbin((dir / "bad.fbin").string()); }, "bin size mismatch");
    fs::resize_file(dir / "gt.ivecs", fs::file_size(dir / "gt.ivecs") - 1);
    rejects([&] { ivf::read_ivecs((dir / "gt.ivecs").string()); }, "truncated ivecs");
    {
        std::fstream file(dir / "x.fvecs", std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(16);
        const std::int32_t wrong_dim = 2;
        file.write(reinterpret_cast<const char*>(&wrong_dim), 4);
    }
    rejects([&] { ivf::read_fvecs((dir / "x.fvecs").string()); }, "inconsistent fvecs dimensions");
    write_matrix(dir / "nan.fbin", std::vector<float>{std::numeric_limits<float>::quiet_NaN()},
                 1, 1, false);
    rejects([&] { ivf::read_fbin((dir / "nan.fbin").string()); }, "non-finite input");

    const auto base_path = (dir / "base.fbin").string();
    const auto centers_path = (dir / "centers.fvecs").string();
    const auto clusters_path = (dir / "clusters.ivecs").string();
    const std::vector<float> base_data{0, 0, 10, 10, 1, 0, 10, 11, 1, 0};
    const std::vector<std::int32_t> cluster_ids{1, 2, 1, 2, 1};
    write_matrix(base_path, base_data, 5, 2, false);
    write_matrix(centers_path, std::vector<float>{99, 99, 0, 0, 10, 10}, 3, 2, true);
    write_matrix(clusters_path, cluster_ids, 5, 1, true);
    const auto index = ivf::read_ivf(base_path, centers_path, clusters_path);
    require(index.list_ptr == std::vector<std::size_t>{0, 0, 3, 5}, "empty list / offsets");
    auto sorted_ids = index.ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    require(sorted_ids == std::vector<std::int32_t>{0, 1, 2, 3, 4}, "all original IDs retained");
    for (std::size_t list = 0; list < index.nlist; ++list) {
        for (std::size_t p = index.list_ptr[list]; p < index.list_ptr[list + 1]; ++p) {
            const auto id = index.ids[p];
            require(std::size_t(cluster_ids[id]) == list &&
                    std::equal(index.vector(p), index.vector(p) + index.dim,
                               base_data.data() + id * index.dim), "list membership and original vector mapping");
        }
    }
    require(index.centroid(1)[0] == 0 && index.centroid(2)[1] == 10, "trained centers preserved");
    write_matrix(clusters_path, std::vector<std::int32_t>{1, 3, 1, 2, 1}, 5, 1, true);
    rejects([&] { ivf::read_ivf(base_path, centers_path, clusters_path); }, "cluster ID out of range");
    write_matrix(clusters_path, std::vector<std::int32_t>{1, -1, 1, 2, 1}, 5, 1, true);
    rejects([&] { ivf::read_ivf(base_path, centers_path, clusters_path); }, "negative cluster ID");
    write_matrix(clusters_path, std::vector<std::int32_t>{1, 2}, 2, 1, true);
    rejects([&] { ivf::read_ivf(base_path, centers_path, clusters_path); }, "cluster count mismatch");
    write_matrix(clusters_path, std::vector<std::int32_t>(10, 1), 5, 2, true);
    rejects([&] { ivf::read_ivf(base_path, centers_path, clusters_path); }, "cluster IDs must be N by 1");
    write_matrix(clusters_path, cluster_ids, 5, 1, true);
    write_matrix(dir / "wrong.fvecs", std::vector<float>{0, 0, 0}, 1, 3, true);
    rejects([&] { ivf::read_ivf(base_path, (dir / "wrong.fvecs").string(), clusters_path); },
            "centroid/base dimension mismatch");

    const std::vector<float> queries{0, 0, 10, 10, 99, 99};
    ivf::SearchOptions options;
    options.k = 4;
    options.nprobe = 1;
    std::vector<ivf::Neighbor> serial(3 * options.k);
    ivf::search_ivf_serial(index, queries.data(), 3, options, serial.data());
    require(serial[0].id == 0 && serial[1].id == 2 && serial[2].id == 4 &&
            serial[3].id == -1, "Top-k tie order and fewer than k candidates");
    require(serial[4].id == 1 && serial[5].id == 3 && serial[6].id == -1,
            "second list lookup");
    require(std::all_of(serial.begin() + 8, serial.end(),
                       [](auto v) { return v.id == -1; }), "empty candidate list");
    auto result = serial;
    result[1].id = 0;
    rejects([&] { ivf::check_result(result, serial); }, "wrong result IDs");
    result = serial;
    result[0].distance = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { ivf::check_result(result, serial); }, "non-finite result distance");

    options.nprobe = index.nlist;
    ivf::search_ivf_serial(index, queries.data(), 3, options, result.data());
    ivf::search_exact_serial(index, queries.data(), 3, options, serial.data());
    ivf::check_result(result, serial);
    options.k = 1;
    std::vector<ivf::Neighbor> nearest(1);
    ivf::search_ivf_serial(index, queries.data(), 1, options, nearest.data());
    require(nearest[0].id == 0 && nearest[0].distance == 0, "k=1 nearest neighbor");

    options.k = 4;
    options.nprobe = 1;
    options.threads = 4;
    options.grain = 1;
    std::vector<ivf::Neighbor> expected(queries.size() / index.dim * options.k);
    ivf::search_ivf_serial(index, queries.data(), queries.size() / index.dim,
                           options, expected.data());
    {
        std::vector<ivf::Neighbor> parallel(queries.size() / index.dim * options.k);
        ivf::search_std_thread(index, queries.data(), queries.size() / index.dim,
                               options, parallel.data());
        ivf::check_result(parallel, expected);
    }
    for (const bool simd : {false, true}) {
        options.simd = simd;
        for (const auto search : {ivf::search_omp_student_1, ivf::search_omp_student_2}) {
            std::vector<ivf::Neighbor> parallel(queries.size() / index.dim * options.k);
            search(index, queries.data(), queries.size() / index.dim, options, parallel.data());
            ivf::check_result(parallel, expected);
        }
        std::vector<ivf::Neighbor> single(options.k);
        ivf::search_omp_student_2(index, queries.data(), 1, options, single.data());
        ivf::check_result(single, std::vector<ivf::Neighbor>(expected.begin(),
                                                               expected.begin() + options.k));

        options.nprobe = index.nlist;
        ivf::search_exact_serial(index, queries.data(), queries.size() / index.dim,
                                 options, expected.data());
        for (const auto search : {ivf::search_omp_student_1, ivf::search_omp_student_2}) {
            std::vector<ivf::Neighbor> parallel(expected.size());
            search(index, queries.data(), queries.size() / index.dim, options, parallel.data());
            ivf::check_result(parallel, expected);
        }
        options.nprobe = 1;
        ivf::search_ivf_serial(index, queries.data(), queries.size() / index.dim,
                               options, expected.data());
    }

    ivf::IntMatrix gt{1, 3, {0, 2, 4}};
    std::vector<ivf::Neighbor> partial{{0, 0}, {1, 2}, {}};
    require(std::abs(ivf::recall_at_k(partial, gt, 1, 3) - 2.0 / 3) < 1e-12,
            "recall denominator remains k with missing results");
}

int main() {
    try {
        const char* home = std::getenv("HOME");
        if (!home) throw std::runtime_error("HOME must be set for test scratch files");
        const fs::path scratch = fs::path(home) / "tmp";
        fs::create_directories(scratch);
        std::string pattern = (scratch / "ivf_tests_XXXXXX").string();
        char* created = mkdtemp(pattern.data());
        if (!created) throw std::system_error(errno, std::generic_category(), "mkdtemp");
        const fs::path dir(created);
        test(dir);
        fs::remove_all(dir);
        std::cout << "PASS: matrix/index readers, malformed files, Top-k, "
                     "parallel SIMD on/off correctness, recall\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
