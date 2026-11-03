#pragma once

#include "ivf_io.h"

#include <limits>
#include <vector>

namespace ivf {

struct Neighbor {
    double distance = std::numeric_limits<double>::infinity();
    std::int32_t id = -1;
};

bool better(const Neighbor& a, const Neighbor& b);

class TopK {
public:
    explicit TopK(std::size_t k);
    void add(Neighbor value);
    void finish(Neighbor* output);

private:
    std::size_t k_;
    std::vector<Neighbor> heap_;
};

struct SearchOptions {
    std::size_t k = 10;
    std::size_t nprobe = 8;
    int threads = 4;
    std::size_t grain = 1024;
};

double l2_distance_scalar(const float* a, const float* b, std::size_t dim);
std::vector<std::uint32_t> select_lists(const IVFIndex& index, const float* query,
                                      std::size_t nprobe);

using SearchFunction = void (*)(const IVFIndex&, const float*, std::size_t,
                               const SearchOptions&, Neighbor*);

void search_ivf_serial(const IVFIndex&, const float*, std::size_t,
                       const SearchOptions&, Neighbor*);
void search_exact_serial(const IVFIndex&, const float*, std::size_t,
                         const SearchOptions&, Neighbor*);
void search_std_thread(const IVFIndex&, const float*, std::size_t,
                       const SearchOptions&, Neighbor*);
void search_omp_student_1(const IVFIndex&, const float*, std::size_t,
                          const SearchOptions&, Neighbor*);
void search_omp_student_2(const IVFIndex&, const float*, std::size_t,
                          const SearchOptions&, Neighbor*);

void check_result(const std::vector<Neighbor>& actual,
                  const std::vector<Neighbor>& expected);
double recall_at_k(const std::vector<Neighbor>& result,
                   const IntMatrix& groundtruth, std::size_t count, std::size_t k);

}
