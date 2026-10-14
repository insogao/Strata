# Phase 2 delivery — Strata Unified（统一 Backend ＋ Logits/Decisions）

> 阶段 2（路线图"第二大阶段"）交付汇总。**未合并 main、未替换生产冻结件。**
> 分支 `phase2/backend-interface`（自阶段 1 通过提交 `e46956bd`）；逐项证据见 `docs/PROJECT_STATUS.md` 与旧研究仓任务目录。
> 生成时间：2026-10-15 01:4x +08:00（主管线）。

## 1. 阶段目标与验收对照

| 路线图验收项 | 状态 | 证据 |
|---|---|---|
| 两个 Backend 从框架层提供**统一能力表**；支持能力真正可用、未支持能力**正确拒绝** | **达成** | P2-1/P2-2（本地后端 CAPS；`SCORE` 零 `T` 行、`pos=3868`）；P2-7（`--backend qwen35_llama\|official_q36`；官方后端能力表合成 `logits_read/state_export/batching=false`；`/v1/score`、`/v1/decisions` 在不支持后端返回 **501 不降级**） |
| 普通 Chat 回归（EOS、连续会话、取消语义） | **达成（贪婪/流式中断路径）** | 364c（官方 server 驱动本后端：聊天 HTTP 200＋文本＋EOS；常驻连续 8+；流式 abort 后引擎自身 `serve: req 3 finish=cancel produced=8`（366））；**采样（temperature>0）与超长生成未系统复测**（见 §4） |
| 关闭 Decisions 后无审核专用行为或性能污染 | **成立（结构上）** | 本阶段新增均为**独立薄层服务**（`tools/score_api.py`）；官方 chat 路径未改动、不含任何业务分支 |
| `/v1/score` / `/v1/decisions` 的 **0 completion-token** 评分数值可复算 | **达成** | P2-3（19 项 stub 单测）＋P2-4（真实引擎：`generated_tokens=0`、与引擎 stdout **4dp 全等**、`pos=3868`） |
| 评分后原状态可继续生成 / 独立续生成可按误差门对拍 | **达成** | P2-2（SCORE 后 `GEN` 正常，语义 (b)）；P2-5/P2-6（`SAVE`/`LOAD` parity：同进程、跨进程、失败不半恢复、可重入字节相等） |
| 多问题是否共享 Prefill **如实计量** | **如实声明：不共享** | 本服务单请求（并发 503 不排队），**无**跨请求 Prefill/前缀复用；文档写明 |
| Session/GDN 隔离正确；前缀复用、状态保存、批处理的**已实现范围**明确 | **达成** | P2-5/P2-6（状态含 **GDN conv+recurrent** 与 Attention KV＋游标；`covers` 与实现一致）；`qwen35_llama_batching_contract.md`（**契约 v1**，`batching=false`；8 GiB 保守准入判据） |
| 真实标注集验证风险漏报/阈值；争议未裁定前不得默认放行 | **不适用/不越权** | 服务**不内置**任何业务词或阈值（risk0–3 留业务层）；历史"标签 3、两引擎均输出 2"争议**仍未裁定**（B3），本阶段**未**启用任何自动放行 |

## 2. 交付物

**代码（仅 `backends/qwen35_llama/**`，官方/引擎源码零改动）**
- `src/program/qwen35_serve.cpp`：serve 协议扩展——`CAPS`、`SCORE`（只读、全词表 log-softmax、不生成 token）、`SAVE`/`LOAD`/`STATE`（含 GDN）、一行 stderr 诊断（`serve: req … finish=…`）。
- `tools/score_api.py`（薄层服务 `/v1/score`、`/v1/capabilities`、最小 `/v1/decisions`）、`tools/backend_adapters.py`（双后端选择＋能力合成）、两份 stub 引擎、`tools/test_*.py`。
**文档**：`qwen35_llama_capabilities_and_score.md`、`qwen35_llama_score_api_design.md`（**SGLang 差异清单/子集边界**）、`qwen35_llama_score_api_e2e.md`、`qwen35_llama_session_state.md`、`qwen35_llama_backend_adapters.md`、`qwen35_llama_batching_contract.md`。

## 3. 关键实测结论（带口径）

| 项 | 结果 |
|---|---|
| CAPS 与实现一致性 | GPU 实测**逐字段 0 差异**（false 项全 false；`mtp.available` 由 GGUF 键推导；`expert_formats` 编译期白名单）；CLI `--capabilities` 0.141 s 且**不加载模型** |
| SCORE 只读性 | `PP` 与 `DONE` 之间**零 `T` 行**、`generated_tokens=0`、`pos=3868`（prompt 末位） |
| 服务透传保真 | 与引擎 **stdout** `LOGPROBS` 原行 **4 位小数全等（Δ=0）** |
| **数值稳定性告警** | **冷态首调 vs 热态重复最大 \\|Δ\\|=0.0111**；重复 2/3 次完全一致；**不得在 <1e-2 精度上比较冷/热评分**；根因未知（仅假设） |
| SAVE/LOAD parity | 同进程、跨进程逐 id 全等；失败（缺失/截断文件）**不半恢复**（STATE 字符级不变）；`SAVE→LOAD→SAVE` **字节全等**（145,215,172 B） |
| 双后端 | 官方 q36 后端：能力表如实 `logits_read/state_export/batching=false`；`/v1/score`、`/v1/decisions` 返回 **501**（无静默降级） |

## 4. 未测 / 未实现（如实；随阶段 3 与后续收敛）

- **采样（temperature>0）与超长生成**经官方服务入口未系统复测（364c 为贪婪＋流式中断；贪婪输出在阶段 1 已证逐字节一致）。
- **并发/批处理**：`batching=false`（契约 v1 只固化边界与准入，**本阶段不实现**）；服务并发为 503 不排队。
- **跨请求前缀复用/磁盘 KV**：未实现（`state_export` 仅覆盖**单会话**导出/导入；跨机器/跨构建**无保证**）。
- 官方 q36 后端**无** SCORE/SAVE/LOAD ⇒ 其在框架层的评分/状态能力为 false（**不得**视为与本地后端等价）。
- 业务层（risk 阈值、标注集）**不在本层**；标签争议未裁定。

## 5. 纪律

生产冻结件只读未动；单 GPU owner＋租约（服务器驱动用例用工具自带 `hold`＋补偿证据，未做改名绕过）；诊断数字不作产品速度；无跨量化比较；禁用数字未引用。
