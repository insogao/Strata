# PROJECT_STATUS — Strata Unified（持续自主开发 + 异步外部审查）

> 维护者：本仓主管线（Codex）。**每次完成里程碑即更新，不等待反馈。**
> 各任务的逐条证据在旧研究仓 `D:/work/Strata魔改/results/opencode-execution/` 与 `docs/`；本文件只做索引与状态。

更新时间：2026-10-14（M1 完成，M2/M3 进行中）

## 0. 一句话状态

**第一阶段（核心迁移）进行中**：M1（源码导入＋独立构建＋保真/配对回归＋文档集）已完成并通过自测；M2（官方 Server 集成）与 M3（统一部署＋产品回归）正在做。第二阶段（统一 Backend/Logits Decisions）按路线图排在阶段一自测达标后**自主启动**；第三阶段（Autotuner）未开始。

## 1. 三大阶段进度

| 阶段 | 状态 | 关键门槛（自测） | 证据 |
|---|---|---|---|
| 阶段 1 核心迁移 | **进行中** | ①本仓独立构建 exit 0；②经**官方服务入口**完成普通聊天/连续请求/EOS/取消/状态隔离；③构建与运行不依赖旧研究工作树；④与旧冻结基线配对回归不劣化 | M1 证据见 §3；M2/M3 证据待补 |
| 阶段 2 统一 Backend + Logits Decisions | 未开始 | 双 Backend 能力表可查询；`/v1/chat/completions` 回归；`/v1/score`、`/v1/decisions` 文档化子集；状态语义与质量回归 | — |
| 阶段 3 Hardware/Workload Autotuner | 未开始 | 安全预算内搜索；4060 上识别 2700/3200 实测差异；第二张卡验证（若不可得则如实标注待办） | — |

## 2. 当前分支与提交

- 分支：`integration/qwen35-llama`（`origin` = `insogao/Strata`，`upstream` = `Niko1221/Strata`）
- PR：#1（自动随分支更新；**标题/正文编辑受 PAT 权限限制，见 §5 阻断项**）
- 提交链（自 base `296b59d2` 起）：
  - `27c987ca` 导入 `backends/qwen35_llama`（源自只读实验树 `phase2-mtp232`：HEAD `9bb9b6a`＋17 dirty＋2 untracked；vendor/llama.cpp live 态）
  - `e0fd2dea` 冒烟＋保真报告（task 355）
  - `c653bef1` ABAB 配对回归报告（task 358）
  - `b5e43917` 生产线同口径对照报告（task 359）
  - `0c118899` 第一阶段文档集（install/run/selection/versions/test-evidence）
  - `7386180d` PROJECT_STATUS（本文件首次建立）
  - `6d9543a8` **精简发布源码树**（task 361）：删 2,019 文件/29,888,754 B（旧项目历史材料），树 6,146→**4,128** 文件（vendor 3,697 占大头）；`docs/backends/qwen35_llama_excluded_material.md`（2,161 行逐文件 SHA 索引，可 `git show 27c987ca:<path>` 取回）；CMake 仅加 SYCL 守卫；**clean 全量重建 749 s / 383 目标 / exit 0** → 新 exe `8c2df0a7040224ccd3c7e960d84efcf9d741a6eb57bb93bf54bd07d71772cc07`（139,195,904 B）
  - （进行中）M2：官方 Server 行协议（task 362）
- 开发产物（未入库，`.gitignore`）：`backends/qwen35_llama/build/strata.exe`（当前 `8c2df0a7…`）
- PR #1 规模：变更文件 6,158 → **4,140**（其中 vendor/llama.cpp 3,697 为必需依赖）

## 3. 已完成的里程碑 M1（阶段 1 内部里程碑）

**M1＝源码导入＋独立构建＋保真/配对回归＋文档集**，自测通过。

| 项 | 结果 | 证据 |
|---|---|---|
| 导入 | 6,146 文件 / 217,233,773 B，19 个改动文件对归档＋live 双验 SHA 全等 | `strata-unified-import-354/` |
| 构建 | clean 全量 666 s、383 目标、exit 0 → `build/strata.exe` 139,208,192 B，sha256 `6062a714…` | 同上 |
| 导入保真 | vs 旧实验 exe `03bd7068`：4 请求可见 token id 序列同 SHA `f876b40d…`；finish=stop/eos=1；hits 3654 / misses 6586 / VRAM 7577.3 MiB 全等 | `strata-unified-smoke-355/`、`docs/backends/qwen35_llama_smoke.md` |
| ABAB 配对回归 | warm TPS A 33.062 vs B 34.470，Δ +4.26%（漂移带 4.248 内）→ PASS | `strata-unified-regression-358/`、`docs/backends/qwen35_llama_regression.md` |
| 生产线同口径 | vs 生产冻结 exe `0e00af8a`：6/6 native exit 0；6/6 输出 16-token 序列逐字节相同（`795a93ac…`）；prefill Δ +1.19%、decode Δ −1.87% → PASS | `strata-unified-prod-regression-359/`、`docs/backends/qwen35_llama_prod_regression.md` |

## 4. 进行中 / 下一步

### 4.1 夜间任务队列（每轮跟进按此顺序取用；一次只跑一个）

| 序 | 任务 | 入口条件 | 交付与自测判据 |
|---|---|---|---|
| 1 | **362 M2 serve 协议外壳**（在飞） | — | `--serve`；READY 时机；GEN 解析（penalty 非默认→ERR）；PP/T/DONE 字段；EOS 计入 produced；STOP→cancel 且进程续；**同进程 >2 连续请求**；**每请求状态重置 reuse=0**；未知输入 ERR 后继续；EOF 退出 0；serve 未默认开实验门；clean 重建 exit 0；`docs/backends/qwen35_llama_serve_protocol.md` |
| 2 | **363 直连协议冒烟**（GPU 租约） | 362 交付且构建产物可用 | 用 `--serve` 直连喂 GEN：READY→PP→T→DONE 逐字段核对；连续 3+ 请求；中途 `STOP` 取消后进程继续；**状态隔离**（两个不同 prompt 输出互不污染）；EOS 命中；资源门与回落 |
| 3 | **364 官方 server E2E**（GPU 租约） | 363 通过 | 官方 `serve/server.py` 以 `exe+args+cwd+env` 配置驱动本后端：HTTP 普通聊天请求成功；连续请求；取消；状态隔离；**不打印 `INFO batch_slots`** 保持单请求且如实声明 |
| 4 | **365 中立放置**（0 GPU） | 364 通过 | 模型/pack/profile 放到**中立方**（`D:/work/StrataUnified/models*` 硬链接或复制＋文档），运行不再以旧研究树为 cwd；写 `versions_and_hashes`/`run_and_model_layout` 更新 |
| 5 | **366 最终配对回归**（GPU 租约） | 365 完成 | 最终 exe 对生产冻结 exe `0e00af8a` 同口径配对回归（prefill/decode 两指标 PASS 判据同 359）；输出序列一致性如实；结果写入 `docs/backends/qwen35_llama_prod_regression.md`（追加或新章） |
| 6 | **367 阶段交付汇总**（0 GPU） | 366 通过 | 更新本文件＋PR #1 标题/正文（迁移模块/构建依赖/验证范围/性能对照/已知限制/验收结果）；列明未测项；**不合并 main** |

> 阶段 1 门槛（全部满足才进入阶段 2 分支）：①本仓独立构建 exit 0；②经**官方服务入口**完成普通聊天＋连续请求＋EOS＋取消＋状态隔离；③构建/运行不依赖旧研究工作树；④与旧冻结基线配对回归不劣化。当前：①④已达成；②③由 362–366 补齐。
> 每完成上表一项：更新本文件 §1–§3、把证据写进 `SUPERVISION.md` 与 `construction-state.json`，然后取下一项；**不等待用户在线验收**。

### 4.2 原计划（已被 4.1 取代，保留作背景）

- **M2（官方 Server 集成）**：在 `qwen35_llama` 内实现官方 server 行协议（`READY`/`GEN`/`PP`/`T`/`DONE`/`STOP`），支持**连续请求、取消、状态隔离**；提供 unified 配置/启动入口（用户不必分别管理两个项目）。
- **M3（统一部署＋产品回归）**：经官方服务入口的 E2E（普通聊天、连续请求、EOS、取消、状态隔离）；构建/运行不依赖旧研究工作树；最终产物重跑与旧冻结基线的配对回归。
- 完成后按阶段门槛自测 → 更新本文件 → 自主创建第二阶段开发分支（**不合并 main**）。

## 5. 阻断与待决项

| # | 项 | 状态 |
|---|---|---|
| B1 | **PR 标题/正文自动更新**：fine-grained PAT 缺 `Pull requests: Read and write`（`gh pr edit` 报 `Resource not accessible by personal access token (updatePullRequest)`；创建 PR 时同因被拒） | 待用户授权该权限；**不影响**开发与分支推送（PR 自动更新） |
| B2 | 生产口径 `TPS_true_flush` 不可得（无逐 token 时间戳门） | 已登记；decode 可比量取 `decode_evaluations_per_s` |
| B3 | MTP K1/T3 正确性门未闭合、标注 2-vs-3 质量争议 | 与迁移无关的既有项；**不**作为已验证前提 |
| B4 | 产品树自包含：模型/pack/profile 放置 | M3 处理（中立路径＋文档） |
| B5 | 历史实验数据未入库 | 由 `docs/backends/qwen35_llama_excluded_material.md`（SHA 索引）＋旧研究仓追溯 |

## 6. 纪律（持续自主开发下仍然有效）

- 不合并稳定 `main`；不替换生产冻结件（GGUF `66a3ca88…`、exe `0e00af8a…`、M0573、PR#3、3200 默认、ACTIVE25）。
- 单 GPU owner；GPU 实验走租约；诊断数字不作产品速度；无跨量化比较。
- 破坏性操作、改变已批准的根本架构、权限扩大、生产切换 —— 停下等用户决策。
- 阻断性正确性问题（MTP 状态错误、生成结果不一致、数据损坏、CUDA 内存安全）不得以"自测通过"掩盖。
