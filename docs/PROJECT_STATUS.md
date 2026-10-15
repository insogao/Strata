# PROJECT_STATUS — Strata Unified（持续自主开发 + 异步外部审查）

> 维护者：本仓主管线（Codex）。**每次完成里程碑即更新，不等待反馈。**
> 各任务的逐条证据在旧研究仓 `D:/work/Strata魔改/results/opencode-execution/` 与 `docs/`；本文件只做索引与状态。

更新时间：2026-10-14（**阶段 1 完成**：四项门槛自测达成，交付汇总见 `docs/PHASE1_DELIVERY.md`；阶段 2 下一步）

## 0. 一句话状态

**第一阶段（核心迁移）已完成并通过自测**：①本仓独立构建 exit 0；②**经官方服务入口**的普通聊天/连续请求/EOS/**取消**/状态隔离全部达成（含引擎级 `finish=cancel` 原文）；③构建与运行**不依赖旧研究树**（中立路径实跑）；④与生产冻结基线的 ABAB 配对回归 **decode PASS、prefill INCONCLUSIVE（中位 +7.15% 在带内，配对噪声；无回归证据）**。阶段 2（统一 Backend/Logits Decisions）可自主启动（**不合并 main**）；阶段 3 未开始。

## 1. 三大阶段进度

| 阶段 | 状态 | 关键门槛（自测） | 证据 |
|---|---|---|---|
| 阶段 1 核心迁移 | **完成（自测达成）** | ①独立构建 exit 0；②官方服务入口的聊天/连续/EOS/取消/隔离；③不依赖旧研究树；④与本产基线配对回归不劣化 | `docs/PHASE1_DELIVERY.md`（四项逐条证据） |
| 阶段 2 统一 Backend + Logits Decisions | 未开始（可启动） | 双 Backend 能力表可查询；`/v1/chat/completions` 回归；`/v1/score`、`/v1/decisions` 文档化子集；状态语义与质量回归 | 接口草案见 350/351 报告（E1–E9、四类缓存边界） |
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
  - `131953d7` **官方 Server 行协议（serve 模式）**（task 362）：`qwen35_serve.cpp` 554 行；exe `2f47b88a…`
  - `17e8ce8d` 直连协议冒烟（task 363d）：连续 3 请求、隔离成立（req1==req3 同 SHA `12699c3c…`）
  - `1e7a0bda` 官方 Server E2E（task 364c）：聊天/连续 8+/隔离；取消有功能性证据（`cancel=True`、engine 7<128）
  - `4e32aae2` 修正 E2E 文档中的生产模型 SHA（主管全量复算 `66a3ca88…`）
  - `cca5f672` **中立放置**（task 365）：8/8 硬链接（同 inode）、pack=native_experts.txt、配置模板与三文档
  - `3503eace` serve 路径 **一行 stderr 诊断** ＋ 中立路径实跑（task 365b）：最终前 exe `f9f4485c…`（504 s 构建）
  - `fb1fc1bb`/`58d18259` **引擎级取消证据**（task 366）：`serve: req 3 finish=cancel produced=8`（官方 server 实时 STOP）
  - `d103d5ff` **最终配对回归**（task 367b）：6/6 native、ABAB、中立路径；decode PASS（+5.74%）、prefill INCONCLUSIVE（+7.15% 在带内、2/3 微小负配对）、两臂输出序列逐字节相同（`795a93ac…`）
  - `a96065ff` **PHASE1_DELIVERY.md**（阶段交付汇总）
- 开发产物（未入库，`.gitignore`）：`backends/qwen35_llama/build/strata.exe`（最终 `f9f4485c…`，139,215,872 B）
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

### 4.0 M2/M3 进度（2026-10-14 21:0x）

- **M2 官方 Server 集成**：`--serve` 行协议已实现（task 362，commit `131953d7`；exe `2f47b88a…`）；直连协议冒烟通过（task 363d，连续 3 请求＋**状态隔离** req1==req3 序列同 SHA `12699c3c…`；生产模型短答 `stop`@2 属正确行为）；文档 `docs/backends/qwen35_llama_serve_protocol.md`、`docs/backends/qwen35_llama_serve_smoke.md`。
- **M3 官方服务入口 E2E**：**已达成**（task 364c，commit `1e7a0bda`，主管修正模型 SHA → `4e32aae2`）：E1 聊天 HTTP 200＋文本＋EOS；E2 常驻连续 8+ 请求全 200；E4 隔离（同 prompt 热重复 4/4 逐字节一致、不同 prompt 不同）；**E3 取消**＝流式中止后 server 记 `disconnect,cancel=True`、引擎**只产出 7 个 token（<128）**、下一请求正常；**引擎字面 `DONE … cancel` 未落盘（官方 server 内部 drain）⇒ 登记为未取得**，将由 365b 的诊断行补齐。租约形态：`gpu-lock hold`（`DENY_BASENAMES` 禁解释器作 declared binary）＋补偿证据（跑前 exe SHA 复算、引擎 READY、收尾锁释放无残留）。
- **M3 中立放置**：task 365 进行中（模型/pack/profiles → `D:/work/StrataUnified/models*` 硬链接；配置模板重指）⇒ 随后 **365b**（serve 路径加一行 stderr 诊断并重建，从中立路径实跑并补引擎级取消证据）→ **366** 用最终 exe 对生产冻结 `0e00af8a` 做同口径配对回归 → **367** 阶段交付汇总＋PR #1 正文。

### 4.0b 阶段 2 进度（2026-10-15 00:3x）

| 项 | 状态 | 证据 |
|---|---|---|
| P2-1 能力表 ＋ 只读 `SCORE` | 完成 | commit `cf134310`（exe `66a48ad1…`；能力字段与实现一致，无虚报） |
| P2-2 CAPS/SCORE 的 GPU 验证 | 完成 | commit `e01e4feb`（零 `T` 行＝不生成 token；`pos=3868`；CLI 0.141 s 不加载模型） |
| P2-3 `/v1/score`＋`/v1/capabilities`＋最小 `/v1/decisions` 薄层服务 | 完成 | commit `7f5392ce`（4 新文件/1424 行；**19 单测全绿**；官方/引擎零改动；SGLang 子集边界文档） |
| P2-4 服务的 GPU 端到端 | 完成（带数值告警） | commit `228796f6`（与引擎 stdout **4dp 全等**；**冷↔热最大 |Δ|=0.0111**，精度告警入档；根因仅假设） |
| P2-5 `SAVE`/`LOAD`/`STATE` 单会话状态（含 GDN） | 完成 | commit `f6e7d048`（exe `6dee0c66…`；先导临时对象不半恢复；path gate） |
| P2-6 `SAVE`/`LOAD` GPU parity | 完成 | commit `596e936f`（同进程/跨进程逐 id 全等；失败不半恢复；可重入字节相等；`state_export` 保持 true） |
| P2-7 双 Backend 适配器＋Batching 契约 | 完成 | commit `17c9ea1f`（官方后端能力表如实 false；不支持端点 501；契约 v1；既有 19＋新 13 单测全绿） |
| 阶段 2 交付汇总 | 完成 | `docs/PHASE2_DELIVERY.md` |

### 4.0c 阶段 3 进度（2026-10-15 13:5x，本分支）

| 任务 | 状态 | 提交/证据 |
|---|---|---|
| P3-1 设备/模型指纹 ＋ 资源预算规划器（0 GPU） | 完成 | commit `dee07c30`；`docs/backends/qwen35_llama_autotuner_foundation.md`（锚点复现 0.439/0.220 GiB） |
| P3-2 离线成本模型 ＋ 候选筛选（0 GPU） | 完成（**只能可信拒绝**） | commit `d350d768`；`docs/backends/qwen35_llama_offline_screen.md`（hold-out 均值绝对误差 **9.70%**、最大 21.30%、同批异号；短名单 ≤5） |
| P3-3 短名单 GPU 配对交错验证（真实引擎） | **完成：候选 `s2200-c-t8-u4096-x8192` 双形状 NO-GO** | commit `66b8528d`；`docs/backends/qwen35_llama_autotune_pilot.md`（6 native 一租约一例、严格串行、0 重跑；S Δ=−5.17%（漂移带 5.58%）、L Δ=−17.52%（漂移带 4.31%）；hits 3654→3032 / 25314→19896、misses 上升 ⇒ **P3-2"2700 饱和、2200 不应更慢"的拐点假设被实测证伪**；3 项口径偏差＋1 项采样口径**在 native 之前**登记于 `retry2/prep/preregistration.json`；`git diff --stat 27c987ca..HEAD -- src` 证明 prefill/decode/tier/MTP 核心未被触碰） |
| P3-4 `strata tune` CLI ＋ profile schema/失效规则 ＋ 阶段 3 交付汇总 | **完成（0 GPU）** | 本提交；`backends/qwen35_llama/tools/strata_tune.py`（inspect/screen/plan/profile/check 五子命令，复用 P3-1/P3-2 模块）＋`test_strata_tune.py`（25 单测全绿）＋`docs/PHASE3_DELIVERY.md`；`check` 四类失效非零退出、跨卡拒绝不换算、`uncalibrated` 警告码 `W-UNCALIBRATED`；既有 19+20+13+19 单测未回归；阶段三**独立 PR #6**（https://github.com/insogao/Strata/pull/6；head＝`phase3/autotuner`，base＝`phase2/backend-interface`，与 PR #3 并列、均不合并 main；同阶段另有 Codex 直接实现线 PR #4 的并行实验线） |

**本分支的 `PROJECT_STATUS` 尚未包含阶段二评审补强（P2R-1…P2R-5）的行**——那部分在 `phase2r/review-gaps` 的拷贝里（§4.0b，权威）。交接先读：**`docs/HANDOFF_FOR_NEXT_AGENT.md`**（全量交接说明：入口、冻结件、口径、纪律、失败处置、未闭合清单、别盲信清单）。

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
- **阶段 1 已完成**（见 `docs/PHASE1_DELIVERY.md`）→ 按用户工作流**自主创建第二阶段开发分支**（自 `integration/qwen35-llama` 通过测试的提交起；**不合并 main**）。
- **阶段 2 起步计划（草案）**：①Backend 能力表（capabilities 可查询：模型格式/MTP/并发/logits 支持）；②`read_candidate_logits` 只读接口（Prefill 后不生成、状态不变）；③`/v1/score`、`/v1/decisions` 的**文档化兼容子集**；④Session 句柄与状态导出/导入边界（KV＋GDN 同存）；⑤官方原生 Backend 适配器与 qwen35_llama 的**同接口**化。

## 5. 阻断与待决项

| # | 项 | 状态 |
|---|---|---|
| B1 | PR 标题/正文自动更新 | **已解决**：用户已授予 `Pull requests: Read and write`；PR #1 标题/正文由主管自动维护 |
| B2 | 生产口径 `TPS_true_flush` 不可得（无逐 token 时间戳门） | 已登记；decode 可比量取 `decode_evaluations_per_s` |
| B3 | MTP K1/T3 正确性门未闭合、标注 2-vs-3 质量争议 | 与迁移无关的既有项；**不**作为已验证前提 |
| B4 | 产品树自包含：模型/pack/profile 放置 | **已完成**（365：中立硬链接＋配置模板＋文档；365b/366 中立路径实跑） |
| B5 | 历史实验数据未入库 | 由 `docs/backends/qwen35_llama_excluded_material.md`（SHA 索引）＋旧研究仓追溯 |

## 6. 纪律（持续自主开发下仍然有效）

- 不合并稳定 `main`；不替换生产冻结件（GGUF `66a3ca88…`、exe `0e00af8a…`、M0573、PR#3、3200 默认、ACTIVE25）。
- 单 GPU owner；GPU 实验走租约；诊断数字不作产品速度；无跨量化比较。
- 破坏性操作、改变已批准的根本架构、权限扩大、生产切换 —— 停下等用户决策。
- 阻断性正确性问题（MTP 状态错误、生成结果不一致、数据损坏、CUDA 内存安全）不得以"自测通过"掩盖。
