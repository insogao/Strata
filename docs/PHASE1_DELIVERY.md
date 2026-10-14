# Phase 1 delivery — Strata Unified（核心迁移）

> 阶段 1（Strata Unified 路线图"第一大阶段"）的交付汇总。**未合并 main、未替换生产冻结件。**
> 逐项证据在本文件与 `docs/PROJECT_STATUS.md` 索引的任务目录中；本文件只做汇总与状态判定。
> 生成时间：2026-10-14 22:1x +08:00（主管线）。

## 1. 阶段门槛（自测）

| # | 门槛（用户 2026-10-14 定义） | 状态 | 证据 |
|---|---|---|---|
| ① | 新仓库内可独立构建 | **达成** | 三次 clean 全量构建 exit 0：`6062a714`（354，666 s，导入首版）、`8c2df0a7`（361，749 s，精简后）、`2f47b88a`（362，384 目标）、`f9f4485c`（365b，504 s，最终，含诊断行） |
| ② | **经官方服务入口**完成普通聊天、连续请求、EOS、取消、状态隔离 | **达成** | 364c（hold 租约，官方 `serve/server.py`）：E1 聊天 HTTP 200＋文本＋EOS；E2 常驻连续 8+ 请求；E4 隔离（同 prompt 热重复 4/4 逐字节一致）；**366**：引擎自身日志 `serve: req 3 finish=cancel produced=8`（流式 abort 后），req 4 后续正常 |
| ③ | 构建与运行**不依赖旧研究工作树** | **达成** | 365：模型/pack/profiles 以硬链接落 `D:/work/StrataUnified/{models,profiles}`（`nlink=2`＋`fsutil` 证同 inode；pack 仅需 `native_experts.txt`）；365b/366 从 `cwd=D:/work/StrataUnified`＋中立路径实跑成功。剩余运行时依赖仅 **CUDA DLL 目录**（PATH 前置 `runtime/cuda-12.8.1/bin`） |
| ④ | 与旧冻结基线**配对回归不劣化** | **实质达成（prefill 判定 INCONCLUSIVE，非劣化）** | 367b：最终 exe `f9f4485c` vs 生产冻结 `0e00af8a`，ABAB 6 native、每例一租约、SHA 6/6 匹配；**decode Δ +5.74% = PASS**；**prefill Δ +7.15% 在阈值内**但 2/3 时间序配对为微小负（−0.52%/−0.12%）⇒ 按预登记规则记 INCONCLUSIVE（**无回归证据**）；两臂 16-token 输出序列逐字节相同（`795a93ac…`） |

> 口径纪律：单次/小样本数字仅作对照；无跨量化比较；未写产品速度声明。367b 的 P1 为冷启动（prefill 6.60 s、`tier_ms.misses` 417.5 ms）已如实登记、未补跑。

## 2. 已迁移 / 未迁移 / 实验性 清单

**已迁入并可用（CERTIFIED 面）**
- Qwen3.6 MMQ Prefill（`moe_mmq.hpp/.cu`、`cold_stager.*`、`qwen35_prefill.*`）
- CPU/GPU 混合专家服务（`expert_source/cache/pool` ＋ AVX2 IQ 内核 ＋ CUDA iq kernels）
- COMPACT 槽位、Prefill loan/restore、路由 Profile 与标定数据
- vendor/llama.cpp（pin `98819068…`/b11476 的 **live 补丁态**）
- 文档：`install_build` / `run_and_model_layout` / `backend_selection` / `versions_and_hashes` / `qwen35_llama_test_evidence` / `qwen35_llama_serve_protocol` / `qwen35_llama_serve_smoke` / `qwen35_llama_server_e2e` / `qwen35_llama_neutral_run` / `qwen35_llama_excluded_material`

**随迁但默认关（EXPERIMENTAL）**：MTP K1 事务链（含两个原未跟踪 txn 文件）、`STRATA_QWEN35_COMPACT_SLOTS`、profile 变体、decode 工作树增量。
**明确不带（REJECTED）**：CPU GU 复用、norms 融合、CUDA Graph 开关（按用户决定不重启）；一次性探针（DIAGNOSTIC-ONLY）不入产品。

## 3. 用户验收清单（第六节）对照

| 要求 | 交付 |
|---|---|
| 开发分支与 PR | `integration/qwen35-llama` → **PR #1**（本文件所在分支；自动更新） |
| 已迁移/未迁移清单 | 见 §2；被移除的旧项目材料见 `qwen35_llama_excluded_material.md`（2,019 文件逐项 SHA，可 `git show 27c987ca:<path>` 取回） |
| 编译结果、安装与启动 | `install_build.md`（工具链、`build.ps1`、实测构建耗时/目标数/产物 SHA）、`run_and_model_layout.md`（启动命令、`--qwen35-pack` 必需、中立布局） |
| Backend 选择方法 | `backend_selection.md`（官方原生引擎 vs `qwen35_llama` 独立可执行；**不声明**已接入 Session/Batching） |
| 普通聊天/EOS/连续/取消/状态隔离 | `qwen35_llama_server_e2e.md`（364c＋366；含引擎级 `finish=cancel` 原文） |
| 同模型同配置性能回归 | `qwen35_llama_prod_regression.md`（359 首版 ＋ 367 最终 exe，两轮 ABAB）＋ `qwen35_llama_regression.md`（导入保真） |
| 模型/程序/依赖 SHA | `versions_and_hashes.md`；模型＝生产 `66a3ca88…`（13,676,723,168 B，主管全量复算）/ 实验 MTP `ab639a7f…`（15,346,432,288 B）；最终 exe `f9f4485c…`；vendor pin `98819068…` |
| 下一阶段接口建议 | 350/351 报告的 E1–E9 职责草案 ＋ 351 的 Logits/Session/KV 边界（四类缓存严格分列） |

## 4. 未测 / 未支持（如实，随阶段推进逐项收敛）

- 并发 Batching、多序列状态、磁盘 KV 缓存、通用 Logits/Decisions API：**未实现**（第二/三阶段）。
- `TPS_true_flush` 在生产口径两臂**均不可得**（无逐 token 时间戳门）⇒ decode 可比量取 `decode_evaluations_per_s`。
- L 案 128/256 长输出**未在最终 exe 上复测**（347/348 为旧实验 exe）。
- MTP K1/T3 正确性门与 risk 2-vs-3 质量争议：**既有未闭合项**，不作为已验证前提。
- prefill 配对判定 INCONCLUSIVE（噪声）；如需正式 PASS 需更大样本（**不追值**，列为可选项）。
- 官方 server 驱动本后端时为单请求（不打印 `INFO batch_slots`，不虚报并发）。

## 5. 纪律与边界

- 生产冻结件（GGUF `66a3ca88…`、exe `0e00af8a…`、M0573、PR#3、3200 默认、ACTIVE25）全程**只读未动**。
- 旧工作树未提交成果已归档（`unified-assets-20261014/`，61/61 SHA 校验）。
- 单 GPU owner；GPU 实验走 `tools/gpu-lock.py`；服务器驱动用例用其**自带 `hold` 模式**并配补偿证据（`DENY_BASENAMES` 禁解释器作 declared binary，**未做改名绕过**）。
- 诊断数字不作产品速度；禁用数字（4656.79／1476.42／−1.12s）未引用。
