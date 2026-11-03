#include "ivf_search.h"

#include <omp.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

struct Arguments {
    std::string base = "/var/dataset/sift_base.fbin";
    std::string centroids = "/var/dataset/centroid_4096.fvecs";
    std::string cluster_ids = "/var/dataset/cluster_id_4096.ivecs";
    std::string queries_file = "/var/dataset/sift_query.fbin";
    std::string groundtruth = "/var/dataset/gt.ibin";
    std::string only = "serial", mode = "bench", csv;
    std::size_t queries = 128, batch = 32, repeats = 10, warmup = 1;
    ivf::SearchOptions search;
};

std::size_t number(const std::string& value) {
    std::size_t n;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), n);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid integer: " + value);
    }
    return n;
}

Arguments parse(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i + 1];
        if (key == "--base") args.base = value;
        else if (key == "--centroids") args.centroids = value;
        else if (key == "--cluster-ids") args.cluster_ids = value;
        else if (key == "--queries-file") args.queries_file = value;
        else if (key == "--gt") args.groundtruth = value;
        else if (key == "--only") args.only = value;
        else if (key == "--mode") args.mode = value;
        else if (key == "--csv") args.csv = value;
        else if (key == "--queries") args.queries = number(value);
        else if (key == "--batch") args.batch = number(value);
        else if (key == "--repeats") args.repeats = number(value);
        else if (key == "--warmup") args.warmup = number(value);
        else if (key == "--k") args.search.k = number(value);
        else if (key == "--nprobe") args.search.nprobe = number(value);
        else if (key == "--grain") args.search.grain = number(value);
        else if (key == "--threads") {
            const auto n = number(value);
            if (n == 0 || n > INT_MAX) throw std::invalid_argument("invalid thread count");
            args.search.threads = static_cast<int>(n);
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (!args.queries || !args.batch || !args.repeats || !args.search.k ||
        !args.search.nprobe || !args.search.grain) {
        throw std::invalid_argument("counts must be positive (except warmup)");
    }
    if (args.mode != "check" && args.mode != "bench") {
        throw std::invalid_argument("--mode expects check or bench");
    }
    return args;
}

ivf::SearchFunction select_strategy(const std::string& name) {
    if (name == "serial") return ivf::search_ivf_serial;
    if (name == "std") return ivf::search_std_thread;
    if (name == "exact") return ivf::search_exact_serial;
    if (name == "omp1") return ivf::search_omp_student_1;
    if (name == "omp2") return ivf::search_omp_student_2;
    throw std::invalid_argument("unknown strategy: " + name);
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "SIFT IVF-Flat starter\n"
            "  --only serial|std|exact|omp1|omp2   --mode check|bench\n"
            "  --queries 128   --batch 32   --k 10   --nprobe 8\n"
            "  --threads 4   --grain 1024   --repeats 10   --warmup 1\n"
            "  --base PATH (.fbin)   --centroids PATH (.fvecs)\n"
            "  --cluster-ids PATH (.ivecs)   --queries-file PATH (.fbin)\n"
            "  --gt PATH (.ibin)   --csv PATH\n";
        return 0;
    }
    auto args = parse(argc, argv);
    const auto search = select_strategy(args.only);
    const auto index = ivf::read_ivf(args.base, args.centroids, args.cluster_ids);
    const auto queries = ivf::read_fbin(args.queries_file);
    const auto truth = ivf::read_ibin(args.groundtruth);
    if (args.only == "exact") args.search.nprobe = index.nlist;
    if (args.only == "serial" || args.only == "exact") args.search.threads = 1;
    if (queries.dim != index.dim || args.queries > queries.rows || args.queries > truth.rows ||
        args.search.k > index.size || args.search.k > truth.dim || args.search.nprobe > index.nlist) {
        throw std::invalid_argument("query count, dimension, k or nprobe exceeds input bounds");
    }
    for (std::size_t q = 0; q < args.queries; ++q) {
        const auto* row = truth.row(q);
        for (std::size_t j = 0; j < args.search.k; ++j) {
            if (row[j] < 0 || std::size_t(row[j]) >= index.size ||
                std::find(row, row + j, row[j]) != row + j) {
                throw std::invalid_argument("invalid or duplicate ground truth ID");
            }
        }
    }
    omp_set_dynamic(0);
    omp_set_num_threads(args.search.threads);
    std::cout << "N=" << index.size << " dim=" << index.dim << " nlist=" << index.nlist
              << " strategy=" << args.only << " threads=" << args.search.threads << '\n';

    std::vector<ivf::Neighbor> reference(args.queries * args.search.k), output(reference.size());
    const auto reference_search = args.search.nprobe == index.nlist
        ? ivf::search_exact_serial : ivf::search_ivf_serial;
    reference_search(index, queries.data.data(), args.queries, args.search, reference.data());
    const double recall = ivf::recall_at_k(reference, truth, args.queries, args.search.k);
    auto run_once = [&] {
        std::fill(output.begin(), output.end(),
                  ivf::Neighbor{std::numeric_limits<double>::quiet_NaN(), -2});
        double ms = 0;
        for (std::size_t begin = 0; begin < args.queries; begin += args.batch) {
            const auto count = std::min(args.batch, args.queries - begin);
            const auto start = std::chrono::steady_clock::now();
            search(index, queries.row(begin), count, args.search, output.data() + begin * args.search.k);
            ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        }
        ivf::check_result(output, reference);
        return ms;
    };
    std::cout << std::fixed << std::setprecision(6);
    if (args.mode == "check") {
        run_once();
        std::cout << "correctness=PASS recall@" << args.search.k << '=' << recall << '\n';
        return 0;
    }

    std::ofstream csv;
    if (!args.csv.empty()) {
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(args.csv);
        csv << "strategy,threads,queries,batch,k,nprobe,grain,repeat,time_ms,qps,recall,correctness\n"
            << std::setprecision(12);
    }
    for (std::size_t r = 0; r < args.warmup; ++r) run_once();
    std::vector<double> times(args.repeats);
    for (std::size_t r = 0; r < args.repeats; ++r) {
        times[r] = run_once();
        if (csv.is_open()) {
            csv << args.only << ',' << args.search.threads << ',' << args.queries << ','
                << args.batch << ',' << args.search.k << ',' << args.search.nprobe << ','
                << args.search.grain << ',' << r + 1 << ',' << times[r] << ','
                << args.queries * 1000.0 / times[r] << ',' << recall << ",PASS\n";
        }
    }
    if (csv.is_open()) csv.close();
    std::sort(times.begin(), times.end());
    const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
    std::cout << "correctness=PASS recall@" << args.search.k << '=' << recall << '\n'
              << "time_ms[min/median/max]=" << times.front() << '/' << median << '/' << times.back()
              << " qps=" << args.queries * 1000.0 / median << '\n';
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
