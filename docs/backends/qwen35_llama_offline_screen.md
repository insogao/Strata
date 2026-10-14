# 离线成本模型与候选筛选（P3-2）

阶段 3 第二项 **P3-2**（分支 `phase3/autotuner`，自 P3-1 提交 `dee07c30`）。两个工具与本文档；
**0 GPU、不取租约、不运行引擎、不做基准、不改官方/引擎文件**，不产生任何产品速度声明。

| 文件 | 是什么 |
|---|---|
| `backends/qwen35_llama/tools/strata_cost_model.py` | 可复算的每 token 时间/TPS 区间模型（`cases` / `params` / `predict` / `holdout` 子命令） |
| `backends/qwen35_llama/tools/strata_screen.py` | 候选筛选器（调用 P3-1 预算规划器 + 成本模型，输出排序与 ≤5 短名单） |
| `backends/qwen35_llama/tools/test_strata_cost_screen.py` | CPU-only 单测 20 项（合成指纹/内存数据，无外部文件） |

一句话结论：把 347 单批的 `tier_ms` 分解变成**参数全部为两条记录字段之比**的区间模型（无自由
拟合参数），hold-out 误差如实报告（均值 9.7%、最大 21.3%、覆盖率 2/4）；筛选器对 72 个候选
逐一做**两池**容量检查（默认全过）并给出每项一个问题、一条失败判据的 5 项短名单。**预测不是结论。**

---

## 1. 成本模型

### 1.1 公式（每 token，MTP K1 解码事务）

```
ms_per_token = rounds_per_token × (id_readback_ms_per_round
                                   + gpu_prefix_ms_per_round
                                   + combine_ms_per_round)
             + misses_per_token × cpu_miss_us_per_expert / 1000
             + non_tier_ms_per_token
TPS = 1000 / ms_per_token
```

- 每项都有 **min/max 区间与均值**；区间端点用各参数的 min/max 组合（包络，不是联合分布）；
- `rounds_per_token`、`misses_per_token` 由调用者给（实测计数，或候选配置声明的命中率）；
- **miss fraction 缺失不猜**：miss 项加宽到物理界 `[0, 每次路由评估都 miss]`，状态标 `wide`；
- **threads ≠ 8 不给数**：347 全部参数只在 threads=8 标定，状态标 `uncalibrated`，不得外推；
- `ctx`/`ubatch` 不进 decode 模型（347 N3/N4 的 ctx 对照非单调、n=2，未建模）；prefill 不在模型内。

### 1.2 参数表（4 例 N1–N4 的逐例值 = 每条参数的标定来源）

| 参数 | N1 | N2 | N3 | N4 | min–max | 均值 | 单位 |
|---|---|---|---|---|---|---|---|
| `id_readback_ms_per_round` | 10.677 | 12.787 | 10.930 | 11.851 | 10.68–12.79 | 11.56 | ms/round |
| `gpu_prefix_ms_per_round` | 2.489 | 4.027 | 3.083 | 3.101 | 2.49–4.03 | 3.17 | ms/round |
| `combine_ms_per_round` | 0.586 | 0.942 | 0.757 | 0.763 | 0.59–0.94 | 0.76 | ms/round |
| `cpu_miss_us_per_expert` | 46.34 | 61.78 | 52.00 | 54.66 | 46.3–61.8 | 53.69 | µs/次 miss |
| `non_tier_ms_per_token` | 9.553 | 12.493 | 12.158 | 11.292 | 9.55–12.49 | 11.37 | ms/token |

来源（同一批，全部来自研究仓原件；工具输出里逐参数携带）：

- `tier_ms.{ids,launch,misses,combine}` ← `results/opencode-execution/phase3-60tps-local-long-baseline-347/raw-N*/child_00.log`
  的 `"event":"summary"` 行（人读表见 `docs/phase3_60tps/LOCAL_LONG_BASELINE_347.md` §2）；
- 每例 `rounds`（80/80、153/153）与 `tokens`（256、512）← 同文件的
  `"event":"qwen35_mtp_txn_request_timing"`（每进程 2 请求）；
- `misses`/`hits` ← 同 summary 的 `decode_misses`/`decode_hits`（25314/77086、46340/149500）；
- `non_tier` = `flush_ms_per_token − tier_ms_per_token`（残差）；`flush_ms_per_token` 由
  `metrics-347.jsonl` 的 `TPOT_true_flush` 按 (n−1) 加权复算（TPS_true_flush 公式见 347 文档 §1）；
- 批身份（exe `03bd7068…`、模型 `ab639a7f…` 15,346,432,288 B、profile `fab192ff…`、
  2700 COMPACT、threads 8、ubatch 4096、MTP K1、GU0、2 请求/进程）← `job-N*.json`/`manifest347.json`。

**没有任何未测自由参数**：五个参数都是两条已记录字段之比；`ctx`、`ubatch`、`COMPACT` 等未标定
维度没有被塞进模型去凑任何目标 TPS。

### 1.3 标定残差（同批 in-sample）

| 例 | 实测 ms/token | 中心预测 | 残差 | 区间覆盖 |
|---|---|---|---|---|
| N1 | 32.101 | 37.228 | +5.128 | 是（构造性） |
| N2 | 41.143 | 36.315 | −4.828 | 是（构造性） |
| N3 | 37.047 | 37.228 | +0.182 | 是（构造性） |
| N4 | 36.646 | 36.315 | −0.331 | 是（构造性） |

注意：全区间端点是四例 min/max 的组合，实测落在区间内是**构造性恒等**，不是独立预测能力。

### 1.4 hold-out（同批次留出，唯一的独立检验）

协议：留出 1 例、用其余 3 例重新拟合、再用留出例自己的计数预测（`holdout.json`）：

| 留出 | 实测 ms/token | 中心预测 | 预测区间 | 误差 | 区间覆盖 |
|---|---|---|---|---|---|
| N1 | 32.101 | 38.938 | [36.18, 42.19] | +21.30% | 否 |
| N2 | 41.143 | 34.705 | [31.30, 37.51] | −15.65% | 否 |
| N3 | 37.047 | 37.289 | [32.10, 42.19] | +0.65% | 是 |
| N4 | 36.646 | 36.204 | [31.30, 41.14] | −1.21% | 是 |

**均值绝对误差 9.70%、最大 21.30%、覆盖率 2/4。** N1/N2 同批但误差符号相反——这是会话/形状
方差没有被 3 例拟合吸收的直接证据（348 已记录 347/348 会话间 prefill/tier 的摆动），如实保留，
不用跨批次数字去抹平。

### 1.5 敏感性（参数 +10% 时中心预测 TPS 变化；只作排序参考，不是承诺）

256 形状（512 tokens / 306 rounds / misses=149500）：

| 参数 +10% | ΔTPS |
|---|---|
| `cpu_miss_us_per_expert` | −4.14% |
| `non_tier_ms_per_token` | −3.04% |
| `id_readback_ms_per_round` | −1.87% |
| `gpu_prefix_ms_per_round` | −0.52% |
| `combine_ms_per_round` | −0.13% |

即：miss CPU 成本与残差项是一阶杠杆；ids/launch/combine 是二阶。排序用，非假设检验。

---

## 2. 筛选器

### 2.1 候选空间（粗→细）与三阶段检查

1. **专家槽预算/层配额**：`--slots {2200,2700,3200}` × `COMPACT {on,off}`；
   host 侧槽缓存成本＝实测点线性缩放：COMPACT `2700→3199.2 MiB`（347 原文
   `native expert tier (compact): 2700 slots, 3.12 GiB` + summary `cache_bytes 3354632192`），
   uniform `2700→4.43 GiB`（同一原文 `uniform would be 4.43 GiB, saved 1336 MiB`）；
   非 2700 点按每槽线性缩放（**政策假设，输出内已标注**）；
2. **线程/CPU pool**：`threads {8,16}`（347 只标定 8）；
3. **prefill ubatch/context**：`ubatch {1024,2048,4096}` × `ctx {4096,8192}`。

每个候选依次做：

- **P3-1 预算规划器**（直接 `import strata_plan`）：VRAM/host **两池分离**；`status=rejected` 的
  原因逐条照录；`status=ok` 时规划器 `rejected` 里的“全 256 专家装不下”等属**告知性子组合**，
  不作为候选拒绝（例：anchor 候选输出 `all 256 experts … need 12590.0 MiB; the effective expert
  budget is 2185.2 MiB`——这正是“本地引擎专家层在 host 侧、规划器残余额度是官方引擎语义”的差异记录）；
- **本地守卫 1**：`ubatch > ctx` 直接拒绝；
- **本地守卫 2**：候选 KV/GDN 相对 ctx8192 的增长不得超过 **347 最高设备 VRAM 峰值
  （7618.4 MiB，N3）到卡总量（8188 MiB）的余量 569.6 MiB**（静态链 2185.2 MiB 中段残余远小于
  实际峰值，说明静态链不描述本地引擎的真实 VRAM 去向——见 §4）；
- **host 池**：`strata_plan` 的 host `requirement ≤ commit limit − OS reserve` 判定，不过即拒。

实测拒绝示例（均在真实指纹上跑出，原件在证据目录）：

- `screen-ctx-probe.json`（故意加入 ctx32768 探针）：
  `s2700-c-t8-u4096-x32768 rejected — local-vram-guard: KV/GDN growth 1056.0 MiB vs ctx8192
  exceeds the headroom 569.6 MiB to the highest recorded 347 device VRAM peak (7618.4 MiB)`；
- `screen-planner-reject-probe.json`（ctx524288）：
  `planner: exceeds the model's declared context_length 262144`（同列专家预算转负等派生条目）；
- `screen-host-reject-probe.json`（slots40000 演示）：
  `host-pool: host requirement 64181.1 MiB exceeds the pageable bound (commit limit minus OS reserve)`。

三条路径（规划器合法性、本地 VRAM 守卫、host 池）都有实测拒绝样例，且拒绝全部显式列出原因，
从不静默丢弃。

### 2.2 默认空间结果（`screen-default.json`，72 候选）

- 状态分布：`ok` 6（2700 COMPACT threads8 全部 ubatch/ctx 组合）、`wide` 30、`uncalibrated` 36、
  `rejected` **0**（默认空间无越界；拒绝机制由探针与单测覆盖）；
- anchor 预测（256 形状）：中心 27.54 TPS，区间 **[24.31, 31.95]**（端点构造性复现 N2/N4）；
- wide 候选（2200/3200/uniform 等）：TPS 物理界 **[21.40, 56.27]**——下界低于 anchor 区间下界，
  所以**未知槽数不可能靠"更多槽=更快"排到实测点前面**；
- 容量：anchor host 需求 19,984.7 MiB ≤ 上限 42,762.0 MiB（commit 44,810 − 保留 2,048）；
  3200 COMPACT 槽缓存 3,791.6 MiB（2200 为 2,606.7 MiB @1.185 MiB/slot）。

### 2.3 排序键（明确不含槽数先验）

`status 类 (ok<wide<uncalibrated<rejected) → 最坏情形 TPS（区间下界）→ 未验证维度数 →
|slots−2700| → ctx/ubatch（记录值优先）→ id`。未知候选以其**物理最坏情形**参排。

### 2.4 短名单（≤5，`screen-default.json`.`shortlist`；每项一个问题/一条失败判据）

| # | 候选 | 状态 | 要验证的问题 | 预期拐点假设 | 失败判据 |
|---|---|---|---|---|---|
| 1 | `s2700-c-t8-u4096-x8192` | ok | anchor 重测（后续臂共享其漂移） | 落在 347 记录的 256 形状 ms/token {36.65, 41.14}（四例区间 [32.10, 41.14]） | 超同形状漂移带（346 2.35–5.29% / 358 4.248% 参照）或 EOS/JSON 守卫失败 |
| 2 | `s2200-c-t8-u4096-x8192` | wide | 槽预算低于 2700 是否损失可测命中率 | 若损失 < 漂移带 ⇒ ≤2700 已饱和，2200 省 0.59 GiB host 且不应更慢 | TPS 降幅超漂移带；或 AvailPhys 触底/页文件代价可见 |
| 3 | `s3200-c-t8-u4096-x8192` | wide | 高于 2700 是否买到任何东西 | 若增益 < 漂移带 ⇒ “更多槽=更快”被证伪，2700 是拐点 | 无漂移带之外的增益；或 host 余量恶化 |
| 4 | `s2700-c-t16-u4096-x8192` | uncalibrated | miss 相是否受 CPU 池大小限制 | 若 `tier_ms.misses`/round 显著下降且 TPS 上移 ⇒ CPU 池是瓶颈 | TPS 因争用下降超漂移带；或 misses 相下降而 TPS 不动（证伪） |
| 5 | `s2700-u-t8-u4096-x8192` | wide | uniform 布局除足迹外是否改变语义 | 预期 hit/miss 计数与 COMPACT=on 相同（host +1337 MiB，原文“saved 1336 MiB”口径） | 同输入 hits/misses 与 compact 臂不同（布局正确性）；或 host 不适配 |

备选第 6 项（本轮因上限 5 落选）：`s2700-c-t8-u1024-x8192`（prefill ubatch 问题；decode 预测
不变，需 prefill t/s 实测）。每项都附带 `minimal_verification`：同二进制 sha（`03bd7068…`）、
同量化（`ab639a7f…`）、同输入（L 案 prompt sha）、配对 ABBA 交错、留出样本纪律、原始 tier/hits/
misses/峰值 VRAM 先落盘。短名单按“每维度一个问题”选取，**不按预期速度排序**。

---

## 3. 证据、来源与口径

- 标定批次（唯一进模型的批次）：**347**（4 native / 8 请求，同 exe/GGUF/profile，MTP K1、2700
  COMPACT、GU0；`docs/phase3_60tps/LOCAL_LONG_BASELINE_347.md`）。
- 只作参照、**不进拟合**：346（短形状 ABBA，A 31.13 / B 31.12，漂移带 2.35–5.29，
  `results/phase3_baselines/GU_RETEST.md`）；358（漂移带 4.248，`stats-358.json`）；359/367
  （生产线配对回归 PASS/INCONCLUSIVE 口径）；341/342/343（官方引擎 6/256·0.44 GiB、3/256·0.22 GiB
  夹紧与 RAM 边缘，P3-1 已锚定）。
- 路由 profile `results/20261009-codex-b6d0bdef-capacity3700/calibration-expanded.profile`
  （sha `fab192ff…`，10240 优先级对，short32 校准扩展）与可用轨迹（detail28 的 2240 行
  ×top-8×7 步、p512 两请求 top-5×32 步）**不足以做长输出、按槽数的命中率模拟**——因此
  2700 之外的 miss fraction 一律 `unknown`，不做假模拟（§4）。
- **跨批次数字不相减相加**；预测一律附来源路径+字段；“诊断数字不是产品速度”在工具与本文档
  各出现一次。

## 4. 未解释项（工具输出同列）

- 2700 COMPACT 之外的 miss fraction（无长输出路由轨迹）；
- ctx 对 decode TPS 的影响（347 N3/N4 非单调、n=2、单批）；
- ubatch≠4096 对 prefill/TTFT 的影响（无 L 案记录）；
- 本地引擎 7.6 GiB 设备 VRAM 峰值的去向（静态链只解释约 3.5 GiB 量级；未分解）；
- AvailPhys 触底（14–367 MiB 极小值）时的页文件代价（未计量）；
- `non_tier_ms_per_token` 残差的构成（draft/verify GPU 尾/emit/sampler/重叠误差）；
- 348 的 `COST_SPLIT/COST_SYNC` 两视图未纳入标定（非任务书列出的输入，且两视图不可相加）。

## 5. 为什么这仍是预测而非结论

1. 模型来自 **n=4、单批、单会话族**，跨会话方差未建模（hold-out 最大 21.3%、覆盖率 2/4）；
2. miss fraction 只在 2700 COMPACT 有测量；其余槽/线程点的区间是**物理界**，不是期望值；
3. prefill/TTFT/首请求、EOS/数值/品质、页文件代价都不在模型内；
4. 短名单是**待验证问题清单**，不是“更快配置列表”；任何 P3-3 结果都须按同二进制/同量化/
   同输入/配对交错/留出纪律复测后才可讨论，且 30–60 档只是研究目标。

## 6. 复现（0 GPU）

```text
python backends/qwen35_llama/tools/test_strata_cost_screen.py -v        # 20/20 OK
python backends/qwen35_llama/tools/strata_cost_model.py params|holdout|predict --tokens 512 --rounds 306 --misses 149500
python backends/qwen35_llama/tools/strata_screen.py --fingerprint <fp-mtp-iq3s.json> --out screen-default.json
```

证据（研究仓 `results/opencode-execution/strata-unified-p3-2-offline-screen/`）：
`calibration.json`、`holdout.json`、`recorded-347-cases.json`、`predict-256-{anchor,unknown,threads16}.json`、
`screen-default.json`、`screen-ctx-probe.json`、`screen-planner-reject-probe.json`、
`screen-host-reject-probe.json`、`test-strata-cost-screen.txt`（20/20 OK）。
指纹原件复用 P3-1 的 `fp-mtp-iq3s.json`（只读，未重取 nvidia-smi）。

## 7. 边界

- 只新增/修改 `backends/qwen35_llama/tools/**` 与 `docs/backends/**`；官方/引擎/冻结件零改动；
- 0 GPU、无租约、无引擎运行、无基准；无 `/tmp`、`%TEMP%`、白名单外路径；
- 与官方 `--expert-cache auto`/动态 swap 的区别：那是运行时容量响应，这里是离线容量与成本的
  **预测**；容量自适应 ≠ 性能自调优，性能结论必须来自 P3-3 的真实配对测量。
