# qwen35_llama：启动与模型/数据放置

运行体：`backends/qwen35_llama/build/strata.exe`（进程级独立可执行；构建见
`docs/backends/install_build.md`）。本文启动命令以 **359 生产线对照实测的 argv 为模板**；
355/358 的实验线 argv/env 另列，避免混淆两套口径。

## 1. 启动命令（359 实测 argv 模板）

359 两臂逐字 argv（cwd = `D:/work/Strata魔改`，全部相对路径）：

```
--qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models --tokens-file results/20261010-jev-moderation-3k4k/han3k-risky/prompt.tokens --max-new 16 --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096
```

参数（按 359 用例）：

| 参数 | 意义 / 实测事实 |
|---|---|
| `--qwen35-model <gguf>` | 模型文件（相对路径，如生产线 `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`）。 |
| `--qwen35-pack <dir>` | **必需项**。359 实测：空 pack 两臂均拒跑（N 源码 `qwen35_model.cpp:1697` 空 `pack_dir` 即退出 2；P 谱系同源要求）；MILESTONE-105 生产协议命令本就为 `--qwen35-pack models`，359 已按"与该 exe 兼容的最小写法"补回并登记。 |
| `--tokens-file <path>` | 预分词 prompt token id 文件（359 用例：3869 ids）。 |
| `--max-new <n>` / `--max-context <n>` | 生成长度 / 上下文上限（359：16 / 4096；355/358：256 / 8192）。 |
| `--qwen35-threads <n>` | CPU 线程（两轮实测均 8）。 |
| `--qwen35-cache-slots <n>` | 专家缓存槽（359 生产口径 3200；355/358 实验线 2700）。 |
| `--qwen35-ubatch <n>` | 物理微批（实测 4096）。 |
| `--qwen35-profile <path>` | 可选。359 生产协议**不传**；355/358 实验线传 `results/20261009-codex-b6d0bdef-capacity3700/calibration-expanded.profile`。 |

环境（359 生产三项，逐项设置前先清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING`）：

```
STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1
STRATA_QWEN35_PACKED_IDS=1
STRATA_QWEN35_EXPERT_ORDER=count_desc
```

- PATH 前置 CUDA 运行库目录（POSIX 写法）：`/d/work/Strata魔改/runtime/cuda-12.8.1/bin`。
- 运行期 stdout 为 JSON 事件行（`{"event":"summary"}`、`{"event":"first_token"}` 等）；
  359 生产协议口径下实测**每进程 1 请求**（6/6 native；未开 txn 引擎）。实验线（355/358）在
  `MTP_TXN=1`＋`MTP_TXN_REQUESTS=2` 下每进程 2 请求——那是研究配置，不是生产启动方式。
- 355/358 实验线另加 12 项研究门（`MTP=1`、`MTP_TXN=1`、`COMPACT_SLOTS=1`、`STOP_EOG=1`、
  `DECODE_TIMESTAMPS=1` 等，`SLICE_GU_MULTI=0` 保持关）；逐项见
  `docs/backends/qwen35_llama_smoke.md` / `qwen35_llama_regression.md`。
- 本机实践：engine 文件参数与导出目录一律 ASCII 相对路径，中文路径只用于 CreateProcessW 的宽 cwd；
  不要给 engine 传中文绝对 argv。

## 2. 模型 / 数据放置

**产品仓库不含模型权重**（`git ls-files` 不含 `models/`；本仓只放源码与文档）。模型沿用既有部署路径、
由用户自行放置。第一阶段的部署实测（355/358/359）是**把模型放在研究树 `D:/work/Strata魔改/models/`
下、以该目录为 cwd 运行**——如实记录；**"产品树自包含（模型随产品目录走）"列为待办**，当前未实现。

实测使用的两条线（研究树布局）：

| 线 | 模型路径（相对 cwd） | 大小 | SHA256 | pack |
|---|---|---|---|---|
| 生产（IQ3_S，冻结） | `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` | 13,676,723,168 B | `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66` | `--qwen35-pack models` |
| MTP 实验（新验证） | `models/experimental/mtp-qwen36-5bc3e238/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` | 15,346,432,288 B | `ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f` | `--qwen35-pack models/experimental/mtp-qwen36-5bc3e238/pack-main40` |

研究树其余相关目录（只读素材，非本后端产物）：`models/experts-requant/`（requant 素材）、
`models/native_experts.txt`、`results/20261009-codex-b6d0bdef-capacity3700/`（profile 标定）等。

## 3. 一次完整运行的落地做法（355/358/359 实际做法）

1. 在 ASCII 相对路径下组织模型/pack/tokens-file；`cwd` 指向其根（本机为中文路径 `D:/work/Strata魔改`，
   经 CreateProcessW 宽 cwd 传入）。
2. 清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING`；PATH 前置 CUDA 运行库目录；
   按用例设置 env。
3. 多 native 严格串行、每 native 单独取 GPU 租约（生产协议口径每 native 1 请求；实验线 txn 门开时每 native 2 请求）。
4. 收集 stdout 的 `summary`/`token`/`first_token` 事件与 exit code；输出文本从事件流读取。

未测/未支持：serve/HTTP API、取消/STOP 语义、并发 Batching、多序列状态、磁盘 KV、通用
Logits/Decisions 接口均**未实现/未接**（汇总见 `docs/backends/qwen35_llama_test_evidence.md`）。
