# 短名单 GPU 配对交错验证（P3-3，重派·修订）

阶段 3 第三项 **P3-3**（分支 `phase3/autotuner`，自 P3-2 提交 `d350d768`）。本任务是**真实 GPU 验证**：把 P3-2 短名单前 2 项带进真实引擎，在**留出形状** S/L 上与当前默认基线做**配对交错**（每形状 A B A），按预登记规则逐候选判定，并把 P3-2 的离线预测与实测对照。**≤6 native（实际 6）、每例一租约、严格串行、每例一次（0 重跑、0 围栏、6/6 exit 0）。**

证据（研究仓，首派死档沿用未删，全部新产物在 `retry2/`）：
`results/opencode-execution/strata-unified-p3-3-autotune-pilot/retry2/`
（`prep/` `pre-env/` `post-env/` `job/` `native-*/` `parse/` `stats/` `p33.py`）。

---

## 0. 结论（预登记规则下的判定）

- **候选 `s2200-c-t8-u4096-x8192`（短名单 #2）＝全局 NO-GO**：两个留出形状都实测更慢，且不在任何"加分"方向上。
  - S（fixture-benign 3896→16，ctx4096）：Δ(warm) = **−5.17%**（|Δ| 同时 ≤ 漂移带 5.58%）→ **NO-GO**；
  - L（l-risk-explain 3817→128，ctx8192）：Δ(warm) = **−17.52%**（|Δ| 4.89 TPS > 漂移带 1.20）→ **NO-GO**。
- **P3-2 的拐点假设（"≤2700 已饱和，2200 不应更慢"）在这两个留出形状上被实测证伪**：2200 槽的命中率确低（hits 3654→3032 / 25314→19896）、miss 确多（6586→7208 / 77086→81224），L 形状 warm TPS 从 27.9 跌到 23.0。**s2200 不进入 profile 推荐；不产生任何产品速度声明。**
- P3-2 离线模型在 L 形状的 A 臂上误差小（加权 −3.3% / −0.3%，区间覆盖），在 16-token 的 S 形状上误差大（+23%…+40%，多不覆盖）——**16-token 形状不在模型标的范围内**，如实分列，不跨形状相减。

---

## 1. 身份、口径与模式登记

### 1.1 固定身份（跑前复核）

| 项 | 值 |
|---|---|
| exe | `D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe`；sha256 `c3780a6a02be44a739a272c8f1cb39b76d9d92bb4f15d18b12cf026145812b0b`（139,267,072 B；prep 复算 + **gpu-lock 每次 launch 复算** `binary_sha256_actual` 6/6 一致） |
| GPU 锁 | `tools/gpu-lock.py` sha256 `337426d63d5e36f199b8bca3e46bea041987a5d68272f426a5976f50da2a7111`（即 345-amended `--stdin-file` 版；合同里登记的 `9382fb9f…` 是 345 前旧值，历次验收均按现值） |
| 模型 | `models/experimental/mtp-qwen36-5bc3e238/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`；15,346,432,288 B；mtime 2026-10-13T10:13:16.525120+08:00；注册 SHA `ab639a7f…`（**未重跑全量 hash**，按合同）；loader 头 `55 kv / 753 tensors`、`qwen35moe.nextn_predict_layers=1` |
| pack | `models/experimental/mtp-qwen36-5bc3e238/pack-main40`（`native_experts.txt`） |
| profile | `profiles/calibration-expanded.profile`（中立 cwd 内），sha256 `fab192ff…`（与 347 原 profile 同文件） |
| fixture S | `results/20261012-qwen35-decode-quality157/fixtures/fixture-benign/prompt.tokens`；21,747 B；sha256 `4e6b2529…`；3896 ids；只读副本→`D:/work/StrataUnified/fixtures-p33/s-benign-3896.tokens`（逐字节一致） |
| fixture L | `results/20261012-qwen35-decode-m0-repair156/fixtures/l-risk-explain/prompt.tokens`；21,389 B；sha256 `a920a130…`；3817 ids；副本→`l-risk-3817.tokens` |
| cwd / argv | `D:/work/StrataUnified`；全部为 ASCII 相对路径；PATH 前置 `D:/work/Strata魔改/runtime/cuda-12.8.1/bin`；先清 `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING` |

**模型路径偏差登记（1/3）**：task.md §1 写"中立模型 `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`＋`models/pack`"。该中立路径是**另一个文件**（13,676,723,168 B；loader 54 kv/733 tensors，**无 `nextn_predict_layers`**），MTP K1 事务引擎在其上不可运行，且与 347/P3-2 标定（ab639a7f、15.3 GB）不同量化文件——按"同量化（ab639a7f…）"的判定前提与 REQUIREMENTS"新模型已验证 SHAab639a7f…pack-main40 已就绪"，本任务使用**实验目录的注册模型**。此为按意图执行、按字面偏差的登记项。

### 1.2 测量模式登记（2/3，核心）

task.md §0/§2 写的"`--stdin-file` 含 2 行同 prompt 的 `GEN`（warm 复测）；每请求记录 `TPS_true_flush`（由 flush 数组复算）、`tier_ms`、hits/misses、peak VRAM"在本 exe 上**无法按 serve 字面执行**：

- 本 exe 的 `--serve` 路径**故意不实现 MTP draft/verify 家族**、并在模型加载前**拒绝** `STRATA_QWEN35_MTP*` 门（`qwen35_serve.cpp` 头注释 §"Nothing of the native engine's --serve is reused…" 与 `qwen35_serve(q)` 的 fail-closed 分支）；
- serve 的 `DONE` 行里 hits/lookups/tier 计数**硬编码 0**（"there is no cache tier behind them"），且 serve 解码循环**不打印任何计时事件**；
- `emit_flush_*` 数组只存在于 **CLI 一次性路径**的 txn 引擎输出（`qwen35_model.cpp`，`STRATA_QWEN35_DECODE_TIMESTAMPS=1` 时打印 `qwen35_mtp_txn_request_flush_arrays`）。

因此本任务采用 **CLI 一次性路径（`--tokens-file`，即 347 标定同管线）**，它天然每进程跑 **2 个同 prompt 请求**（`STRATA_QWEN35_MTP_TXN_REQUESTS=2`：rq1 冷、anchor=prefill_begin；rq2 热、anchor=request_reset）——即 task.md"2 行同 prompt 的 GEN（warm 复测）"的语义等价物（346/347/358 的既有口径：warm=req2）。判定取样一律用 **rq2（warm）**。此登记写在任何 native 之前（`retry2/prep/preregistration.json`）。

### 1.3 env 登记（3/3）

task.md 写"env 仅三项生产门"。三项生产门（`ROUTER_WEIGHTS_DEVICE=1`/`PACKED_IDS=1`/`EXPERT_ORDER=count_desc`）全部在场；**另 9 项是 347 批次必需的测量/配置门**，缺一则 task.md 要求的字段根本不会产生：`COMPACT_SLOTS=1`（配置）、`STOP_EOG=1`、`DECODE_TIMESTAMPS=1`（flush 数组/计时行）、`MTP_TXN=1`/`MTP_TXN_K=1`/`MTP_TXN_REQUESTS=2`（K1 事务轮）、`SMALL_T_NATIVE=1`/`SMALL_T_MULTI=1`/`SLICE_GU_MULTI=0`（346/347 公共口径）。完整 env 逐 native 相同，登记于 `preregistration.json` 与每例 `argv-plan.json`。

### 1.4 矩阵与租约纪律

- 短名单 **#1 `s2700-c-t8-u4096-x8192` 与基线完全等价**（slots2700/COMPACT/threads8/ubatch4096）→ 按 task.md"若某项与基线等价则跳过并说明"**跳过**；有效候选＝**#2 `s2200-c-t8-u4096-x8192`**。故每形状 **A B A**（基线括号候选；≤6 native 硬帽下不能再铺 ABBA），总 **6 native**。
- 顺序（严格串行，每例一个 `gpu-lock run` 租约，declared binary＝exe、`--expect-job`、`--source opencode-p33`）：

| # | case | 形状 | 臂 | slots | 租约获取时刻 | child s | exit | 围栏保留 | 等待 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S-A1 | S 3896→16 ctx4096 | A | 2700 | 13:42:30 | 18.953 | 0 | false | 1（0.0 s） |
| 2 | S-B1 | S | B | 2200 | 13:42:49 | 22.578 | 0 | false | 1 |
| 3 | S-A2 | S | A | 2700 | 13:43:13 | 21.219 | 0 | false | 1 |
| 4 | L-A1 | L 3817→128 ctx8192 | A | 2700 | 13:43:35 | 23.875 | 0 | false | 1 |
| 5 | L-B1 | L | B | 2200 | 13:44:00 | 24.328 | 0 | false | 1 |
| 6 | L-A2 | L | A | 2700 | 13:44:25 | 23.875 | 0 | false | 1 |

- 每个 native：`audit/result/ready/gpu-lock-run.log/raw/child_00.log/argv-plan.json` 全落盘；`child_cleanup` 全 `all_exited/roots_exited/active_processes_zero=true`；marker 6/6 `retained=false`，锁批后为空；批前/批后 RAM/GPU 全录（批前 16.79 GiB / 1037 MiB；批后 17.24 GiB / 988 MiB；无 strata 进程残留）。
- **窗口纪律**：以 `task.md` mtime（13:31:01）为锚，每 native 启动前程序化检查；六次 pre-launch 剩余 512/492/468/446/422/396 s，**全部 > 本 native 预估 300 s**，故六次都无需重锚（未 touch；批末 219 s）。检查值与余额逐例在 `argv-plan.json` 的 `window_check` 与 `pre-env/*`、`post-env/*` 中。
- 每例一次，0 工具故障、0 重跑；串行无并行；无 `/tmp`/`%TEMP%`/白名单外写；未以 `D:/work` 为根 glob；未改源码/冻结件；未合并 main。

---

## 2. 逐 native 事实（原文行为准）

每请求 `TPS_true_flush = (n−1)/(末次可见 flush − 首次可见 flush)`（由 `qwen35_mtp_txn_request_flush_arrays` 的 `emit_flush_request_seconds` 复算；逐条数组在 `retry2/parse/parsed-*.json` 与 raw 日志）。

| case | rq1 TPS（冷） | rq2 TPS（warm） | rq2 finish | eos | emitted | 可见文本 |
|---|---|---|---|---|---|---|
| S-A1 | 21.275494 | **28.834293** | length | 0 | 16+16 | 6 个 S 请求文本逐字节相同：`{\n"risk": 0,\n"violation": false\n}`（16 token 处 `json_closed=true`，未达 EOG） |
| S-B1 | 22.926115 | **28.127739** | length | 0 | 16+16 | 同上 |
| S-A2 | 25.558278 | **30.488871** | length | 0 | 16+16 | 同上 |
| L-A1 | 27.069589 | **28.501461** | length | 0 | 128+128 | 解释文本，128 token 顶满（未达 EOG；与 347 同位事实） |
| L-B1 | 25.153942 | **23.010806** | length | 0 | 128+128 | 解释文本 |
| L-A2 | 26.601555 | **27.298571** | length | 0 | 128+128 | 解释文本 |

每进程 summary 原文（节选，逐字）：

```text
S-A1 {"exit":0,"cache_slots":2700,"decode_hits":3654,"decode_misses":6586,"tier_ms":{"ids":160.0,"launch":90.7,"misses":483.3,"combine":18.4,"admit":0.0},"peak_device_vram_mib":7649.0,"cache_resident":2700,"prefill_tokens_per_s":702.535}
S-B1 {"exit":0,"cache_slots":2200,"decode_hits":3032,"decode_misses":7208,"tier_ms":{"ids":185.7,"launch":44.8,"misses":505.3,"combine":9.8,"admit":0.0},"peak_device_vram_mib":7041.8,"cache_resident":2200,"prefill_tokens_per_s":370.5}
S-A2 {"exit":0,"cache_slots":2700,"decode_hits":3654,"decode_misses":6586,"tier_ms":{"ids":183.0,"launch":47.1,"misses":428.4,"combine":10.2,"admit":0.0},"peak_device_vram_mib":7629.4,"cache_resident":2700,"prefill_tokens_per_s":406.41}
L-A1 {"exit":0,"cache_slots":2700,"decode_hits":25314,"decode_misses":77086,"tier_ms":{"ids":1747.6,"launch":482.3,"misses":3967.0,"combine":110.9,"admit":0.0},"peak_device_vram_mib":7599.8,"cache_resident":2700,"prefill_tokens_per_s":843.836}
L-B1 {"exit":0,"cache_slots":2200,"decode_hits":19896,"decode_misses":81224,"tier_ms":{"ids":1763.3,"launch":507.9,"misses":4480.1,"combine":121.9,"admit":0.0},"peak_device_vram_mib":7081.1,"cache_resident":2200,"prefill_tokens_per_s":876.31}
L-A2 {"exit":0,"cache_slots":2700,"decode_hits":25314,"decode_misses":77086,"tier_ms":{"ids":1738.0,"launch":517.7,"misses":4104.0,"combine":122.0,"admit":0.0},"peak_device_vram_mib":7597.9,"cache_resident":2700,"prefill_tokens_per_s":876.303}
```

计数复现事实：**L 臂 A 的 hits/misses=25314/77086 与 347 N1 逐字相同；S 臂 A=3654/6586 与 346 A 臂逐字相同**——路由/槽计数跨构建可复现（但见 §5 沿革与未解释项，不对"代码生成级等价"作任何声称）。

---

## 3. 配对判定（预登记规则，冻结于 `retry2/prep/preregistration.json`）

规则照录：对每条候选与基线的 **warm** 请求取中位；Δ＝候选−基线；漂移带＝臂内极差最大值；PASS：Δ≥+5% 且 >漂移带 且配对方向一致；NO-GO：Δ≤0 或 |Δ|≤漂移带；其余 INCONCLUSIVE。

| 形状 | A warm 样本 | A 中位 | B warm | Δ 绝对值 | Δ % | 漂移带 | 方向 | 判定 |
|---|---|---|---|---|---|---|---|---|
| S | 28.834293 / 30.488871 | 29.661582 | 28.127739 | −1.533843 | **−5.171%** | 1.654578（5.578%） | B 输两侧 | **NO-GO**（Δ≤0 且 \|Δ\|≤漂移带，两分支同真） |
| L | 28.501461 / 27.298571 | 27.900016 | 23.010806 | −4.889210 | **−17.524%** | 1.202890（4.311%） | B 输两侧 | **NO-GO**（Δ≤0） |

**总体：候选 `s2200-c-t8-u4096-x8192` = NO-GO**（双形状 NO-GO）。冷 sample（rq1）按 prereg 只记录不判定：A 21.28/25.56（S）、27.07/26.60（L）；B 22.93（S）、25.15（L）。

**与 P3-2 预登记的拐点假设/失败判据对照**：假设"若命中率损失换算 TPS 降幅 < 漂移带 ⇒ 2200 不应更慢"——实测 S 降 5.17%（≈漂移带上沿内），**L 降 17.52% 远超漂移带**；辅证 B 命中率下降、misses/token 升高（L：misses 77086→81224，+5.4%）。**拐点假设在 L 形状上被证伪（2200 明显未饱和、确实更慢）**，S 形状同向但幅度与漂移带同量级、单 B 样本不放大。

**"越界应被拒"项实证（task.md 额外问题）**：短名单 #1/#2 无应拒项（状态 ok/wide；P3-2 的拒绝空间由 probe 与单测覆盖）——本矩阵范围内 **N/A**；未把 P3-2 对拒斥的预测当结论使用。

---

## 4. 与 P3-2 预测的对照（预测误差的实测样本）

把 P3-2 成本模型（`calibration.json` 五参数，均值中心 + min/max 包络）**直接回放到每次运行的自身计数**（tokens=两请求 emitted 和、rounds=两请求 rounds 和、misses=summary decode_misses），与实测 ms/token 对照（weighted＝按 (n−1) 加权；warm＝rq2）：

| case | 实测 weighted | 实测 warm | 预测中心 | 预测区间 | 误差%（weighted / warm） | 覆盖（weighted / warm） |
|---|---|---|---|---|---|---|
| S-A1 | 40.842 | 34.681 | 29.205 | [25.106, 32.975] | +39.84 / +18.75 | 否/否 |
| S-B1 | 39.585 | 35.552 | 30.249 | [26.007, 34.176] | +30.86 / +17.53 | 否/否 |
| S-A2 | 35.963 | 32.799 | 29.205 | [25.106, 32.975] | +23.14 / +12.31 | 否/是 |
| L-A1 | 36.014 | 35.086 | 37.228 | [32.101, 42.192] | −3.26 / −5.76 | 是/是 |
| L-B1 | 41.607 | 43.458 | 37.975 | [32.742, 43.052] | +9.56 / +14.44 | 是/否 |
| L-A2 | 37.112 | 36.632 | 37.228 | [32.101, 42.192] | −0.31 / −1.60 | 是/是 |

读法（防混读）：

- **L 形状（与标定同族）**：A 臂误差 −3.3%/−0.3%（warm −5.8%/−1.6%），区间覆盖——模型在本形状族上延续 P3-2 记录的精度量级；**B 臂 +9.6%（warm +14.4%）**——模型对"低槽数 miss 上升"的系统性预测不足，正是本任务用实测补上的那部分。
- **S 形状（16 token）**：误差 +23%…+40%，大多数不覆盖——**该形状不在模型标定/形状范围内**（模型不含首请求/冷效应/prefill），如实分列，不与 L 数字相减。
- P3-2 发表值（256 形状）仅并排参考、**不跨形状换算**：anchor 中心 27.54 TPS、区间 [24.31, 31.95]；s2200 中心 32.36、物理界 [21.40, 56.27]（下界口径使其本可容纳"未知槽数"任何慢值——实测落在界内不构成支持）。

---

## 5. exe 沿革备注（未测残余差异）

- 347 标定批次 exe＝`results/phase2_mtp/consistentSlice300/strata.exe`，sha `03bd7068…`，139,107,840 B；本任务 exe＝`c3780a6a…`，139,267,072 B。
- 两构建之间引擎源码差异（`git diff --stat 27c987ca..HEAD -- backends/qwen35_llama/src`，27c987ca＝MTP 研究后端导入提交）：**仅** `program/qwen35_serve.cpp`（新增 serve 文件）、`program/generate.cpp`（+29 行：`--capabilities`/`--serve` 路由）、`kernels/cuda/iq_kernels.cu`（+2 行：`iq_expert_role_format` 能力查询辅助）。**`qwen35_model.cpp`/`expert_source.cpp`/`qwen35_mtp_txn.cpp` 等 prefill/decode/tier/MTP 核心自导入提交未被触碰。**
- **不声称代码生成层面的等价**：编译器/构建时间差异、会话级速度差异（见 §6）都算未测残余；本批一切数字只描述本批（同 exe/模型/profile/fixture 的配对交错内比较成立）。

---

## 6. 未解释项（如实记录，不作解释性结论）

1. **会话级整体偏慢**：同形状同配置的 L 臂 A（rq2 28.50/27.30、rq1 27.07/26.60）低于 347 N1 记录（rq2 31.41、rq1 30.90）约 9–14%；S 臂 A warm（28.83/30.49）低于 346 记录带（31.35–34.98）。原因未分离（会话/热漂移/构建残余混合），跨批差值**不用于任何换算**。
2. **S 形状 finish 语义与 346 记录不同**：本批 6/6 S 请求 `finish=length`、`eos=0`（16 token 处 JSON 已闭合、未出现 EOG；文本为多行 `{\n"risk": 0,\n"violation": false\n}`）；346 记录为"16 可见 token、EOS=1、单行文本"。是构建沿革还是会话采样差异**未测**（须专项复现才可判断）。
3. **B 臂设备峰值 VRAM 低约 550–570 MiB**（7629–7649 → 7042–7081 MiB）：静态分析只预期 host 侧省 0.59 GiB；设备侧差额的来源未分解。
4. **B 臂单样本**：漂移带按 pre-reg 只由 A 臂极差给出（B 的臂内极差无样本），已在 preregistration 登记为限制；不据此放宽任何判定。
5. tier `admit` 全程 0、`SMALL_T` 等门是否全程生效未逐项审计（本任务非其审计范围）。

---

## 7. 复现与边界

- 复现（研究仓）：`python retry2/p33.py stats`；原始逐请求数组在 `retry2/parse/parsed-<case>.json`，raw 日志 `retry2/native-<case>/raw/child_00.log`，租约原件 `retry2/native-<case>/{audit,result}.json`，预登记 `retry2/prep/preregistration.json`。
- 本任务：**0 构建、0 源码改动**；未动冻结件/生产/公开 PR；单一 GPU owner（gpu-lock 6 租约，串行）；≤6 native（用满 6）；每例一次；无自释放、无绕过、无围栏。
- **采样与回落口径（如实登记）**：本批未部署 10 s 外部 GPU sampler（native 仅 19–25 s，10 s 粒度只 2–3 样本且 347 已记录混叠问题）；每 native 峰值以引擎自身 `summary.peak_device_vram_mib` 为准（§2 原文行），并配 6×pre + 6×post + 批前/批末快照（相邻快照间隔 ~20–25 s）。"回落"证据：每例结束后 GPU 用量回到 ~1000 MiB、RAM ≥17 GiB、strata/子进程 0、锁空、无 marker（`retry2/post-env/*`、`retry2/pre-env/batch-end.json`）。
- **判定边界**：均在 4060 Laptop 8 GiB、真实 4K 语境下的**诊断数字**，非产品速度、非 p95、非稳态；30–60 档只是研究目标；本文件不含任何提速声明。
