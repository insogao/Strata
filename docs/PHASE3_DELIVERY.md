# Phase 3 delivery — Strata Unified（Hardware/Workload Autotuner 阶段收尾）

> 阶段 3（路线图"第三大阶段：硬件/负载自调优"）交付汇总。**未合并 main、未替换生产冻结件、未改引擎/vendor 源码。**
> 分支 `phase3/autotuner`（自阶段二 `954a88e4`）；阶段三**独立 PR**（head＝`phase3/autotuner`，base＝`phase2/backend-interface`，与补强 PR #3 并列、互不包含）。
> 逐项证据在旧研究仓 `D:/work/Strata魔改/results/opencode-execution/strata-unified-p3-*`；本文档为索引与结论。

## 1. 阶段目标与验收对照

路线图三门槛（`docs/PROJECT_STATUS.md` §1）：①安全预算内搜索；②4060 上识别 2700/3200 实测差异；③第二张卡验证（不可得则如实标注待办）。

| 门槛 | 状态 | 证据/边界 |
|---|---|---|
| ① 安全预算内搜索 | **工具链交付；实测搜索仅 1 个候选** | P3-1 预算器 + P3-2 筛选 + P3-4 验证计划构成闭环工具；P3-3 实际验证了短名单 #2 一个候选（NO-GO），其余候选未上机；**不宣称搜索已完成** |
| ② 识别 2700/3200 差异 | **部分：2700 vs 2200 有实测；3200 未测** | P3-3：2200 双形状 NO-GO（下表）；3200 仅存在于短名单 #3，**未上机**；拐点假设（"2700 饱和"）**被实测证伪** |
| ③ 第二张卡验证 | **待办（无第二张卡可用）** | 跨卡/跨量化一律拒绝且**不做任何换算**（P3-4 `check` 固化该纪律）；如实标注 |

P3-1…P3-4 逐项映射（提交号、证据目录、关键数字）：

| 项 | 提交 | 交付 | 关键数字 | 证据目录 |
|---|---|---|---|---|
| P3-1 指纹＋预算器 | `dee07c30` | `strata_fingerprint.py`＋`strata_plan.py`；19 单测 | 锚点复现 0.439/0.220 GiB（静态链 vs 运行期复现）；0 GPU | `strata-unified-p3-1-fingerprint/` |
| P3-2 离线模型＋筛选 | `d350d768` | `strata_cost_model.py`＋`strata_screen.py`；20 单测 | 72 候选；hold-out 均值绝对误差 **9.70%**、最大 **21.30%**、覆盖率 2/4；**只能可信拒绝** | `strata-unified-p3-2-offline-screen/` |
| P3-3 GPU 配对验证 | `78fa1f3c`＋`66b8528d` | 6 native、6 租约、0 重跑；预登记 | S Δ=−5.171%（漂移带 5.578%）/ L Δ=−17.524%（漂移带 4.311%）→ 候选 `s2200-c-t8-u4096-x8192` **双形状 NO-GO** | `strata-unified-p3-3-autotune-pilot/retry2/` |
| P3-4 `strata tune` CLI＋汇总＋PR | 本提交 | `strata_tune.py`（5 子命令）＋`test_strata_tune.py`（25 单测）＋本文档 | 0 GPU；字段/失效规则逐条测试；`plan` 不执行；`check` 四类失效非零退出 | `strata-unified-p3-4-tune-cli/` |

## 2. P3-4 交付物：`strata tune` CLI

`python backends/qwen35_llama/tools/strata_tune.py <COMMAND> …`（复用 P3-1/P3-2 模块，全部 0 GPU：不启动引擎、不申请租约、不构建；默认只做 P3-1 同款只读 nvidia-smi/GGUF 头探测，`--fingerprint` 提供指纹文档时可完全离线运行）。

| 子命令 | 做什么 | 关键约束 |
|---|---|---|
| `inspect` | 设备/模型指纹＋两池（VRAM/host）预算；`--json`/人类可读 | 复用 `strata_fingerprint.py`/`strata_plan.py`，不重实现 |
| `screen` | P3-2 候选筛选：ok/wide/uncalibrated/rejected 分类＋≤5 短名单 | **强制打印"只能可信拒绝、不足以选优"＋hold-out 均值 9.70%/最大 21.30%（覆盖率 2/4）**；同一输入同一输出（确定性） |
| `plan` | 对短名单输出预登记 GPU 验证计划：形状（S 3896→16 ctx4096 / L 3817→128 ctx8192）、臂（A 基线 2700 COMPACT / B 候选）、交错 A B A（每候选 6 native）、判定规则、**窗口与租约要求** | **只规划不执行**（`executed=false`；无 native/租约/引擎） |
| `profile` | 把**已被证据支持**的配置写成 profile JSON | 无证据只能标 `confidence=uncalibrated` 且**不得**写为推荐；`--recommend` 需 `confidence=ok`＋证据，否则拒绝 |
| `check <profile>` | 与当前环境逐项对比 | 四类失效任一即失败（非 0）；跨卡明确拒绝并打印两侧指纹、**不做任何换算**；`uncalibrated` 给警告码 `W-UNCALIBRATED` |

**退出码语义（稳定，`test_strata_tune.py` 逐条断言）**：`0`＝成功/`check` 通过（身份匹配且 confidence∈{ok,wide}）；`2`＝用法错误、输入无效、操作被拒或 `check` 失效失败；`3`＝`check` 身份匹配但 `confidence=uncalibrated`（显式警告码 `W-UNCALIBRATED`，不得当作推荐）。

### profile schema（`schema_version = "strata-tune-profile/1"`）

| 字段 | 含义 |
|---|---|
| `schema_version` | `strata-tune-profile/1` |
| `model_sha256_prefix` | 注册模型 SHA 前缀（8–64 位小写 hex；**从不重算全文件 hash**） |
| `quant` | 量化标签（如 `IQ3_S`；**禁止跨量化换算**） |
| `device_fingerprint` | GPU 名/显存/驱动/计算能力四元组（任一变即失效） |
| `argv` | `slots/ubatch/threads/ctx/compact`（与运行配置逐项比对） |
| `evidence` | 证据引用（目录/提交；无证据只能 `uncalibrated`） |
| `confidence` | `ok` / `wide` / `uncalibrated` |
| `recommendation` | 仅当 `confidence=ok`＋有证据＋`--recommend` 才为 true；`uncalibrated` 恒 false |
| `generated_at` | 生成时间（UTC ISO） |
| `invalidators` | 失效规则表（随 profile 固化，与 §3 一致） |

## 3. 失效规则（`check` 任一不满足即失败、退出码非 0、不静默沿用）

| 码 | 条件 | 行为 |
|---|---|---|
| `MODEL_SHA_MISMATCH` | `model_sha256_prefix` 与当前注册 SHA 不匹配；当前 SHA 未知时 **fail closed** | 打印 stored/current，退出 2 |
| `QUANT_MISMATCH` | `quant` 不一致 | 打印两侧；**no cross-quantization conversion**，退出 2 |
| `DEVICE_MISMATCH` | `device_fingerprint` 任一字段变化 | **明确拒绝跨卡/跨驱动套用，打印两侧指纹，不做任何换算**，退出 2 |
| `ARGV_MISMATCH` | `slots/ubatch/threads/ctx/COMPACT` 任一项与运行配置不一致 | 打印两侧，退出 2 |
| `W-UNCALIBRATED` | `confidence=uncalibrated` | 警告码；该 profile **不是推荐**；退出 3 |

规则表以 fixture 在 `test_strata_tune.py` 逐条正/负向断言（25 单测全绿）；四类失效各至少 1 条负向、跨卡拒绝 1 条、警告码 1 条、profile 往返 1 条、退出码与未知命令 2 条。

## 4. 关键实测结论（带口径，如实）

- **P3-3 候选 `s2200-c-t8-u4096-x8192`＝双形状 NO-GO**（真实引擎，6 native，一租约一例，0 重跑）：
  - S（fixture-benign 3896→16，ctx4096）：Δ(warm)=**−5.171%**（漂移带 5.578%）→ NO-GO；
  - L（l-risk-explain 3817→128，ctx8192）：Δ(warm)=**−17.524%**（漂移带 4.311%）→ **决定性 NO-GO**；
  - 计数：hits 3654→3032 / 25314→19896，misses 6586→7208 / 77086→81224（低槽数命中率下降是机制线索，非结论）。
- **P3-2 的拐点假设被实测证伪**："2700 已饱和、2200 不应更慢"不成立——2200 在两形状都更慢；P3-2 事后回放：L 臂权重误差 −3.3%/−0.3% 且在区间内，S 形状 +23%…+40% 出模型基域（该形状不在标定范围）。
- **离线模型只能可信拒绝、不足以选优**：hold-out 均值 9.70%/最大 21.30%、覆盖率 2/4、同批误差异号；`screen` 每次输出强制携带该定位句与误差。
- **P3-3 预登记口径偏差（native 之前冻结于 `retry2/prep/preregistration.json`）**：
  1. 模型路径：task 字面"中立路径"文件无 `nextn` 层、与标定量化不同 → 按意图使用注册实验模型 `ab639a7f…`；
  2. 测量模式（核心）：本 exe 的 `--serve` 故意不实现 MTP 门且 DONE 计数硬编码 0 → 改用 CLI 一次性路径（=347 同管线，天然 2 请求/进程，warm=rq2 语义等价）；
  3. env：三项生产门外另有 9 项 347 批次必需的测量/配置门（逐 native 相同并登记）；
  4. 采样口径：无 10 s 外部 GPU sampler（native 仅 19–25 s），峰值以引擎 `summary.peak_device_vram_mib`＋6×pre/6×post＋批前批末快照为准。
- **exe 沿革（核心未动的源码 diff）**：347 标定 exe `03bd7068…`（139,107,840 B）→ 本任务 exe `c3780a6a…`（139,267,072 B）；`git diff --stat 27c987ca..HEAD -- backends/qwen35_llama/src` 仅 `qwen35_serve.cpp`（新增 serve）、`generate.cpp`（+29 行路由）、`iq_kernels.cu`（+2 行能力查询）——`qwen35_model.cpp`/`expert_source.cpp`/`qwen35_mtp_txn.cpp` 等 prefill/decode/tier/MTP 核心**未被触碰**；**不声称代码生成层面等价**。

## 5. 未闭合项（如实；不含任何"已通过"暗示）

1. **跨卡验证**：无第二张卡 → 待办；跨卡/跨量化在工具层直接拒绝、不换算（P3-4 `check`、P3-2 文档）。
2. **16-token S 形状在模型标定范围外**：P3-2 预测对 S 形状 +23%…+40% 不覆盖；S 的 `finish=length/eos=0` 与 346 记录（EOS=1）语义差异未定位。
3. **冷/热根因未定位**（P2 遗留：冷态首评 vs 热态 |Δ|≤0.0111 仅入档；P3-3 会话级整体偏慢 9–14% 未分离）。
4. **批次契约 v1 未实现**：`batching=false`（P2-7 只固化了边界与准入判据）。
5. **两监听器并存**：thin service（`score_api.py`）与官方 server 驱动路径的监听入口未统一（接口未统一项见 `docs/HANDOFF_FOR_NEXT_AGENT.md` §6、`qwen35_llama_backend_adapters.md` §6–9；无合并服务声明）。
6. 其余：tier `admit` 全程 0 未逐项审计；B 臂设备峰值低 550–570 MiB 来源未分解；P2R 补强线（PR #3）未并入本分支；风险 label3 质量争议未裁定。

## 6. 纪律与 0 GPU 证据

- P3-4 全程 **0 GPU**：无 `strata.exe` 调用、无 `gpu-lock` 调用、无 fence、无构建、无下载、无全文件 hash；唯一外部读取是 `--fingerprint` 缺席时 P3-1 同款只读 nvidia-smi/GGUF 头。
- 测试：`python backends/qwen35_llama/tools/test_strata_tune.py -v` → **25/25 OK**；既有模块未改：`test_strata_autotune.py` 19/19、`test_strata_cost_screen.py` 20/20、`test_backend_adapters.py` 13/13、`test_score_api.py` 19/19（原文落在 `strata-unified-p3-4-tune-cli/`）。
- 阶段三全部提交（`954a88e4..phase3/autotuner`）**只动 `backends/qwen35_llama/tools/` 与 `docs/`**；`git diff --stat 954a88e4..HEAD -- backends/qwen35_llama/src backends/qwen35_llama/vendor` 为空。
- 不合并 main；不替换生产冻结件；未以 `D:/work` 为根 glob；无 `/tmp`/`%TEMP%` 写入。

## 7. 给评审的检查清单

1. 本文档 §1 映射表 → 对照 `docs/PROJECT_STATUS.md` §4.0c（P3-1…P3-3 行）与三个原始证据目录 `strata-unified-p3-1-fingerprint/`、`strata-unified-p3-2-offline-screen/`、`strata-unified-p3-3-autotune-pilot/retry2/`。
2. P3-3 结论复核：`docs/backends/qwen35_llama_autotune_pilot.md` §0/§3（Δ 与漂移带）、§5（exe 沿革）、§7（采样/复现）；重算脚本 `retry2/p33.py stats`。
3. P3-2 误差复核：`docs/backends/qwen35_llama_offline_screen.md` §1.4 与 `holdout.json`；或直接 `python backends/qwen35_llama/tools/strata_cost_model.py holdout`。
4. P3-4 复核：`python backends/qwen35_llama/tools/strata_tune.py --help`；`test_strata_tune.py -v`（25/25）；`strata-unified-p3-4-tune-cli/` 内各子命令真实输出样例与单测原文。
5. 引擎未动证明：`git diff --stat 954a88e4..phase3/autotuner -- backends/qwen35_llama/src`（空）＋ `docs/backends/qwen35_llama_autotune_pilot.md` §5 的 27c987ca..HEAD 版本。
6. 纪律复核：`git show --stat <P3-4 commit>`（只含本任务 5 个文件）；PR 与 PR #3 并列、均不合并 main。
