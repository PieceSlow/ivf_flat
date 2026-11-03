# OpenMP IVF-Flat 实验代码

这是 [向量检索作业](openmp_ivf_topk_lab.pdf) 的配套代码，使用 SIFT1M 数据和已经训练好的 IVF 索引。数据读取、串行检索、结果检查和计时已经写好。OpenMP、`std::thread` 和 SIMD 都需要自己实现，多线程函数留在 `ivf_student.cpp` 中。

## 编译和运行

进入解压后的代码目录，执行：

```bash
make
make test
```

需要支持 C++17 和 OpenMP 的 g++，运行环境为 64 位 little-endian Linux。索引训练结果已经导出，编译时不需要链接 Faiss。默认编译选项是 `-O3 -std=c++17 -fopenmp -pthread`，完整选项见 `Makefile`。改过编译选项后，用 `make clean && make` 重新编译。

先运行串行版本：

```bash
./ivf_lab --only serial --queries 128 --batch 32 --csv serial.csv
```

这条命令处理前 128 个查询，每批 32 个，预热一次后重复 10 次。结果会与串行参考逐项比较，终端显示时间和 Recall，CSV 保存每次重复的数据。

只想检查结果时，可以使用 `--mode check`。下面两条命令分别检查扫描全部列表的 IVF 和全库检索：

```bash
./ivf_lab --only serial --queries 4 --batch 1 --nprobe 4096 --mode check
./ivf_lab --only exact --queries 4 --batch 1 --mode check
```

## 数据文件

实验机器统一在 `/var/dataset/` 提供数据，程序默认读取下面五个文件。压缩包包含源码、测试和文档，数据文件不用另行复制。

| 文件 | 内容 |
| --- | --- |
| `sift_base.fbin` | 1,000,000 个 128 维数据库向量 |
| `sift_query.fbin` | 10,000 个 128 维查询向量 |
| `gt.ibin` | 每个查询的 1000 个近邻编号，共 10,000 行 |
| `centroid_4096.fvecs` | Faiss 训练得到的 4096 个中心，每个中心 128 维 |
| `cluster_id_4096.ivecs` | 每个数据库向量所属的列表编号，共 1,000,000 行，每行一个编号 |

中心和列表编号是 Faiss 训练后的导出结果。`cluster_id_4096.ivecs` 的第 i 行对应数据库中的第 i 个向量，因此数据库、中心和列表编号这三份文件要配套使用。加载时，程序把同一列表的向量排到一起，并保留原始编号；这个过程约需 1 GiB 内存，不计入检索时间。

读取函数声明在 `ivf_io.h` 中：

```cpp
auto index = ivf::read_ivf("/var/dataset/sift_base.fbin",
                          "/var/dataset/centroid_4096.fvecs",
                          "/var/dataset/cluster_id_4096.ivecs");
auto queries = ivf::read_fbin("/var/dataset/sift_query.fbin");
auto gt = ivf::read_ibin("/var/dataset/gt.ibin");
```

`.fbin` 和 `.ibin` 的开头是两个 `uint32`，依次表示行数、维数，后面是连续的 `float32` 或 `int32` 数据。`.fvecs` 和 `.ivecs` 则在每一行前放一个 `int32` 维数，再存这一行的数据。四种格式都有对应的读取函数；主程序的查询文件固定使用 `.fbin`，ground truth 固定使用 `.ibin`。

`queries.row(i)` 返回第 i 个查询。列表 l 中的向量位于 `[index.list_ptr[l], index.list_ptr[l+1])`，用 `index.vector(p)` 访问向量，用 `index.ids[p]` 取得它的原始编号。`index.centroid(l)` 返回列表中心。

## 需要补全的代码

`ivf_student.cpp` 留了三个函数：

- `search_omp_student_1`：把不同查询分给不同线程。
- `search_omp_student_2`：把一个查询的候选分成多个任务，分别计算后合并 Top-k。
- `search_std_thread`：用 `std::thread` 实现按查询分块的版本，与 OpenMP 对比。

两个 OpenMP 实现还需要接入自己编写的 SIMD 距离函数，并添加开关做对照实验。

三个函数的参数相同，声明在 `ivf_search.h` 的 `ivf` 命名空间中：

```cpp
void search_omp_student_1(const IVFIndex& index, const float* queries,
                         std::size_t count, const SearchOptions& options,
                         Neighbor* output);
```

一次调用处理一个 batch。输入数据只读，`queries` 中有 `count * index.dim` 个元素，`output` 已分配 `count * options.k` 个位置。第 q 个查询的第 rank 个结果写到 `output[q * options.k + rank]`。`SearchOptions` 中的 `k`、`nprobe`、`threads`、`grain` 分别控制结果数、搜索列表数、线程数和候选块大小。

可以直接使用已有的 `select_lists`、`l2_distance_scalar`、`TopK` 和 `better`。调用 `TopK::finish` 后，结果已经排好序并写入输出，不要再往这个 TopK 对象里添加候选。

三个空函数目前都会报 TODO 错误。实现后，分别用 `--only omp1`、`--only omp2`、`--only std` 运行。`make test` 只检查已提供的公共代码，自己的实现还要另行验证。

## 参数和输出

运行 `./ivf_lab --help` 可查看参数。

| 参数 | 默认值或用途 |
| --- | --- |
| `--only` | 默认 `serial`，也可选 `std`、`exact`、`omp1`、`omp2` |
| `--mode` | 默认 `bench`；`check` 只运行并检查一次 |
| `--queries` / `--batch` | 前 128 个查询，每批最多 32 个 |
| `--k` / `--nprobe` | 10 / 8 |
| `--threads` / `--grain` | 4 / 1024 |
| `--repeats` / `--warmup` | 10 / 1；warmup 可以为 0 |
| `--base` / `--centroids` / `--cluster-ids` | 指定数据库、中心、列表分配文件 |
| `--queries-file` / `--gt` | 指定查询 `.fbin` 和 ground truth `.ibin` |
| `--csv` | 保存每次正式重复的数据；文件已存在时会覆盖 |

计时从调用检索函数开始，到这一批的结果全部算完为止，包括中心选择、候选扫描、Top-k、线程管理和同步。数据加载、参考结果计算、检查和打印都在计时之外。程序汇总各批时间，再输出多次重复的最小值、中位数、最大值和 QPS。

每次运行都会重置并检查全部输出。结果按距离、编号升序排列，编号必须与参考一致，距离容差为 `atol=rtol=1e-10`。候选不足 k 个时，余下位置填 `id=-1, distance=+inf`。当 `nprobe=nlist` 时，参考结果改用全库扫描。

CSV 记录策略、线程数、查询数、batch、k、nprobe、grain、重复编号、时间、QPS、Recall 和检查结果。作业要求的候选数、线程工作量和延迟分位数需要自行采集。数据路径、编译选项和运行环境请记在报告中。

`--only` 只选择要测的检索函数。直接对整个程序执行 `perf stat`，仍会把加载和串行参考算进去；如何单独测检索部分，见作业的性能工具要求。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `ivf_io.h/.cpp` | 数据和索引读取 |
| `ivf_search.h/.cpp` | 距离、Top-k、串行检索、结果检查和 Recall |
| `ivf_student.cpp` | 待完成的 OpenMP 和 `std::thread` 函数 |
| `ivf_lab.cpp` | 参数解析、运行、计时和 CSV 输出 |
| `tests/ivf_tests.cpp` | 小型数据、文件读取和结果检查的测试 |
