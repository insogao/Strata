# Backend 选择：官方原生引擎 vs `qwen35_llama`

截至 2026-10-14（第一阶段收尾），本仓内存在两种可用的执行体：

| | ① 官方原生引擎 | ② `qwen35_llama` |
|---|---|---|
| 形态 | 官方 exe ＋ 官方 server（server 以 `exe+args` 配置驱动引擎） | **进程级独立可执行**：`backends/qwen35_llama/build/strata.exe`（自带 argv，不经 server） |
| 构建 | 官方顶层 CMake / `setup.py` | 独立 CMake 工程（`backends/qwen35_llama/build.ps1`），**不入官方构建**（目标名冲突 + vendor 为打补丁 llama.cpp） |
| 服务能力 | HTTP API、Batching、Session、prompt cache 等官方能力 | **无** serve/API；**无**取消/STOP 语义；**无**并发 Batching；**无**多序列状态；**无**磁盘 KV；**无**通用 Logits/Decisions 接口（未测/未接，见下） |
| 每进程请求数 | 官方引擎按 server 调度 | 生产协议口径实测 **1 请求/进程**（359：6/6 native）；实验线 txn 门开时每进程 2 请求（355/358；研究配置） |
| 模型线 | 官方发布面（本仓 base v0.1.44.1 的 setup） | 已实测两条线：生产 IQ3_S `66a3ca88…`、MTP 实验 IQ3_S `ab639a7f…`（+`pack-main40`） |
| 研究开关 | 无 | MTP K1 / COMPACT / profile 等（**EXPERIMENTAL、default off**，非产品能力） |
| 产出物 | 官方部署产物（engine/server） | 研究后端 exe（139,208,192 B，`6062a714…`） |

## 选择判据

1. **需要官方服务能力**（HTTP API、Batching、Session、prompt cache、多序列、取消/STOP 语义）→ 选
   官方原生引擎。`qwen35_llama` 没有这些能力，且不是 server 集成件。
2. **需要跑/复现该研究线**（MTP K1、COMPACT 槽位、profile 变体、以及本阶段验收过的 IQ3_S 双线
   对照口径）→ 选 `qwen35_llama`，并在报告中始终标注 EXPERIMENTAL。
3. **量化面**：本后端**实测覆盖**＝IQ3_S 两条线（构建开启 `STRATA_MMQ_KQUANTS`＋`STRATA_Q6K_EXPERTS`）；
   不要写成"IQ3_S 仅本后端支持"——官方 v0.1.44.1 的 setup 推荐表同样含 IQ3_S
   （`setup.py:2237` `RECOMMENDED_SIZES`）；两引擎对同一 GGUF 文件的互跑**未测**。除已实测者外，
   不得把任何其他量化/格式写成"已支持"。
4. **隔离要求**：两后端目标名冲突、vendor ABI/CUDA-DLL 期望不同，必须各自独立构建目录、独立 exe；
   不要加进同一棵 CMake 树（`docs/backends/qwen35_llama.md` "Build isolation"）。

## 措辞纪律（350/351 已固定）

- **不得**写成"`qwen35_llama` 已接入官方 Session/Batching/prompt cache"。它是**进程级独立可执行**；
  由官方 server 驱动它属于**单请求**能力（350/351 口径），且**当前未实现**（本阶段无 serve/API 证据）。
- **不得**把 MTP K1 / COMPACT 写成已认证：K1 的 T3/质量门未闭合（2-true-vs-fixed-label-3 风险未闭合），
  本阶段 355/358 是实验门开启下的研究配置，不构成产品能力。
- 选择依据只引用已被本阶段实测覆盖的数字（`docs/backends/qwen35_llama_test_evidence.md`），
  不引用官方/他机/他量化的换算值。
