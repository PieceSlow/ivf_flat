#include "ivf_search.h"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ivf {

namespace {

// 只加速候选向量的距离计算；中心选择仍调用公共的标量函数，保证选中相同列表。
double l2_distance_simd(const float* a, const float* b, std::size_t dim) {
    double sum = 0.0;
    // 各 SIMD lane 先累加自己的部分，循环结束后再归约到 sum。
#pragma omp simd reduction(+:sum)
    for (std::size_t j = 0; j < dim; ++j) {
        const double delta = double(a[j]) - double(b[j]);
        sum += delta * delta;
    }
    return sum;
}

// std::thread 版本复用串行检索过程：一个查询由一个线程完整处理。
void search_one_scalar(const IVFIndex& index, const float* query,
                       const SearchOptions& options, Neighbor* output) {
    const auto lists = select_lists(index, query, options.nprobe);
    TopK best(options.k);
    for (const auto list : lists) {
        // list_ptr 给出该倒排列表在重排后的向量数组中的半开区间。
        for (std::size_t p = index.list_ptr[list]; p < index.list_ptr[list + 1]; ++p) {
            best.add({l2_distance_scalar(query, index.vector(p), index.dim), index.ids[p]});
        }
    }
    best.finish(output);
}

std::size_t worker_count(int requested, std::size_t count) {
    if (count == 0) return 0;
    return std::min<std::size_t>(requested > 0 ? std::size_t(requested) : 1, count);
}

} // namespace

void search_std_thread(const IVFIndex& index, const float* queries, std::size_t count,
                       const SearchOptions& options, Neighbor* output) {
    // 查询不足线程数时，不创建没有工作的线程。
    const std::size_t workers = worker_count(options.threads, count);
    if (workers == 0) return;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    // 按连续查询区间分配；向上取整以覆盖最后不足一块的部分。
    const std::size_t block = (count + workers - 1) / workers;
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t begin = t * block;
        const std::size_t end = std::min(count, begin + block);
        if (begin >= end) break;
        threads.emplace_back([&, begin, end] {
            for (std::size_t q = begin; q < end; ++q) {
                // 每个 q 只由一个线程处理，写入的 output 区间互不重叠。
                search_one_scalar(index, queries + q * index.dim, options,
                                  output + q * options.k);
            }
        });
    }
    // 函数返回前必须等待所有线程完成；创建和等待都计入调用耗时。
    for (auto& thread : threads) thread.join();
}

void search_omp_student_1(const IVFIndex& index, const float* queries, std::size_t count,
                          const SearchOptions& options, Neighbor* output) {
    const auto distance = options.simd ? l2_distance_simd : l2_distance_scalar;
    // 一个循环迭代就是一个完整查询；dynamic 让空列表/长列表造成的工作量差异
    // 不必固定落在某个线程上。batch=1 时仍只有一个查询能参与此策略。
#pragma omp parallel for schedule(dynamic) num_threads(options.threads)
    for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(count); ++q) {
        const float* query = queries + std::size_t(q) * index.dim;
        const auto lists = select_lists(index, query, options.nprobe);
        // TopK 是本次迭代的局部对象，线程之间不共享。
        TopK best(options.k);
        for (const auto list : lists) {
            for (std::size_t p = index.list_ptr[list]; p < index.list_ptr[list + 1]; ++p) {
                best.add({distance(query, index.vector(p), index.dim), index.ids[p]});
            }
        }
        best.finish(output + std::size_t(q) * options.k);
    }
}

void search_omp_student_2(const IVFIndex& index, const float* queries, std::size_t count,
                          const SearchOptions& options, Neighbor* output) {
    if (count == 0) return;
    const auto distance = options.simd ? l2_distance_simd : l2_distance_scalar;
    // 整个 batch 只建立一次线程团队；single 线程负责创建查询任务。
#pragma omp parallel num_threads(options.threads)
    {
#pragma omp single
        {
            for (std::size_t q = 0; q < count; ++q) {
                // q 按值复制，避免循环前进后任务读到别的查询编号。
#pragma omp task firstprivate(q)
                {
                    const float* query = queries + q * index.dim;
                    const auto lists = select_lists(index, query, options.nprobe);
                    const std::size_t grain = options.grain;
                    std::size_t blocks = 0;
                    // 每个选中列表各自切块，块长最多 grain；空列表产生 0 块。
                    for (const auto list : lists) {
                        const std::size_t length = index.list_ptr[list + 1] - index.list_ptr[list];
                        blocks += (length + grain - 1) / grain;
                    }
                    // 每块预留 k 个输出位置。只有所属块任务写自己的区间。
                    std::vector<Neighbor> partials(blocks * options.k);
                    std::size_t block_id = 0;
                    for (const auto list : lists) {
                        const std::size_t begin = index.list_ptr[list];
                        const std::size_t end = index.list_ptr[list + 1];
                        for (std::size_t first = begin; first < end; first += grain) {
                            const std::size_t last = std::min(end, first + grain);
                            const std::size_t id = block_id++;
                            // 边界和块编号按值复制；partials/query 由查询任务持有，
                            // 在下面的 taskwait 之前始终有效。
#pragma omp task firstprivate(first, last, id, distance) shared(partials, query)
                            {
                                TopK best(options.k);
                                for (std::size_t p = first; p < last; ++p) {
                                    best.add({distance(query, index.vector(p), index.dim),
                                              index.ids[p]});
                                }
                                best.finish(partials.data() + id * options.k);
                            }
                        }
                    }
                    // 等待本查询创建的所有块任务，才能读取 partials 或释放它。
#pragma omp taskwait
                    TopK best(options.k);
                    // 全局前 k 名一定在某一块的前 k 名中；合并这些局部结果即可。
                    for (std::size_t b = 0; b < blocks; ++b) {
                        for (std::size_t r = 0; r < options.k; ++r) {
                            best.add(partials[b * options.k + r]);
                        }
                    }
                    best.finish(output + q * options.k);
                }
            }
        }
    }
}

}
