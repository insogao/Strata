# qwen35_llama：测试证据汇总（354 / 355 / 358 / 359 已验收实测）

口径：全部为本机（RTX 4060 Laptop 8 GiB、Windows）单机小样本实测；n 小、非稳态，不构成统计显著性
推断，不给 p95。数字只在同一次实测内可比较，不跨 GPU/量化换算。**MTP K1 / COMPACT 仍为 EXPERIMENTAL，
不是已认证能力**；本页只汇总做过的事与数字，不做产品承诺。

## 实测配置对照（先读，避免混淆两套口径）

| | 355 / 358（导入保真 + 计时回归） | 359（生产线同口径对照） |
|---|---|---|
| exe | A＝`03bd7068…`（MTP 实验冻结） vs B＝`6062a714…`（本仓新构建） | P＝`0e00af8a…`（生产冻结） vs N＝`6062a714…` |
| 模型 | `models/experimental/mtp-qwen36-5bc3e238/...IQ3_S.gguf`（`ab639a7f…`） | `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`（`66a3ca88…`） |
| prompt | fixture-benign `prompt.tokens`（3896 ids） | han3k-risky `prompt.tokens`（3869 ids） |
| 形状 | max-new 256（实际 16-token EOS）、cache 2700 槽 | max-new 16（EOS）、cache 3200 槽 |
| env | 12 项研究门全开（MTP/K1/COMPACT/STOP_EOG/DECODE_TIMESTAMPS…） | 生产三项（`ROUTER_WEIGHTS_DEVICE/PACKED_IDS/EXPERT_ORDER`） |
| 性质 | 实验线诊断/回归（默认关闭的门被显式打开） | 生产协议口径 |

## 1. 354 构建（2026-10-14）

- clean 全量构建 **exit 0，383 ninja 目标，wall 666 s（8 路）**；产物
  `backends/qwen35_llama/build/strata.exe`，139,208,192 B，
  SHA256 `6062a714c38744a6adc1cb052f3242c304a3ff32d158545933834455544dc9b4`。
- 来源：`results/opencode-execution/strata-unified-import-354/build-354-clean.log`、`evidence-354.md`。

## 2. 355 GPU 冒烟 + A/B 保真

- native 3/3 全 exit 0（A 旧 exe ×2＝1 次采样器故障（我方采样器 bug，native 本体 exit 0）+1 次完整；
  B 新 exe ×1）；两臂唯一差异＝exe 路径，argv/env 逐项一致；无 fence、无残留。
- **可见 token id 序列 SHA256 4/4 相同**（A req1/req2、B req1/req2）＝
  `f876b40d97dd5fa276309562faacac640257edb01e38dc65d8cf8d37395f39a8`；可见文本相同；跨请求亦同
  （确定性重放）。
- **finish=stop，eos=1**（真实 EOG，`eog_id=248046`，end_pos 3912；max_new=256 未被顶到——非 length）。
- 资源/命中 4/4 全等：hits **3654**、misses **6586**、cache_resident 2700（cache 3199.2 MiB）、
  峰值 VRAM **7577.3 MiB**。
- 计时小样本偏差：TPS_true_flush A∈[33.493701, 35.331270]，B∈[25.348123, 32.457156]；req1 −28.3%
  （超 B 自身两请求离散 21.9%）**不能由本次样本判为构建回归**（n=2 请求/例，非稳态）。
- 来源：`docs/backends/qwen35_llama_smoke.md`（sha256 `892b35f2…`）、
  `results/opencode-execution/strata-unified-smoke-355/`。
- 登记偏差：355 预登记的 347 基线口径不适用（fixture 3896≠3791，且本轮为 16-token EOS 而非 256
  length）；fixture tokens 现盘 SHA 与其 manifest 不一致（事实记录，不影响 A/B）。

## 3. 358 ABAB 计时回归（旧 exe vs 新 exe，warm）

- 4 native（A1→B1→B2→A2）严格串行、全 exit 0、无 fence/无重跑；守卫 **8/8**＝emitted 16 /
  finish stop / eos 1。
- warm（req2）TPS_true_flush（口径：`emit_flush_request_seconds` 数组，(n−1)/(末−首)，n=16＝15 个 flush 区间）：
  A_med＝**33.061975** vs B_med＝**34.469589**，Δ＝**+1.407614（+4.2575%）**；
  阈值 max(5%×A_med, 漂移带 4.248466)＝4.248466 → **PASS**；成对 +3.23/−0.41 非同向负。
- 限制：每 exe 仅 n=2 warm；A 组内部漂移（4.248466）大于 Δ 本身，阈值实际由漂移带主导——PASS 强度
  受样本量限制（已登记待重复）。
- 来源：`docs/backends/qwen35_llama_regression.md`（sha256 `52269c6e…`）、
  `results/opencode-execution/strata-unified-regression-358/stats-358.json`。

## 4. 359 生产线同口径对照（P `0e00af8a` vs N `6062a714`）

- 6/6 native（P,N,N,P,P,N）exit 0、每 native 1 请求（该谱系无 txn 引擎）、严格串行、每例 1 租约、
  无 fence/残留。
- **6/6 可见 16-token 序列逐字节相同**（`index:id` 连接序列 SHA256＝
  `795a93accf98b2f398a9037fa13e7a41363a369c59756d626329101f6439b165`；可见文本 6/6 全等）。
- prefill（tok/s，= prefill_tokens_per_s）：P 1044.33 / 1085.74 / 1058.81（中位 1058.81） vs
  N 1071.39 / 1092.86 / 940.54（中位 1071.39）；**Δ +1.1881%（PASS**；阈值 152.319＝漂移带主导）。
- decode（可比量＝**`decode_evaluations_per_s`**，= 15 evals / decode_seconds）：P 15.90 / 19.74 / 19.10
  （中位 19.1047） vs N 17.79 / 19.77 / 18.75（中位 18.7467）；**Δ −1.8739%（PASS**；阈值 3.8348）。
  均无系统性单边（各仅 1 个配对为负）。
- `TPS_true_flush` **两臂均不可得**（生产协议无逐 token 时间戳门）。
- 引擎静态量 6/6 一致：prompt_tokens=3869、decode_hits=3020、decode_misses=1788、cache_resident=1788、
  cache_mib=3750.0、cache_slots=3200、strata_jobs=1211936。
- 登记偏差：argv 按"与该 exe 兼容的最小写法"补回 `--qwen35-pack models`（两臂空 pack 均拒跑；生产协议
  本就含此参数），两臂一致。
- 来源：`docs/backends/qwen35_llama_prod_regression.md`（sha256 `27cdfea0…`）、
  `results/opencode-execution/strata-unified-prod-regression-359/stats-359.json`、`compare-359.json`。

## 5. 未测 / 未支持清单（如实）

以下均为**未测或未实现**，不得对外写成"已支持"：

1. serve / HTTP API（OpenAI/Anthropic 兼容层、server 驱动）；
2. 取消 / STOP 语义（生成中途取消、停止事件）；
3. 并发 Batching；
4. 多序列状态；
5. 磁盘 KV（持久化 KV / 磁盘缓存）；
6. 通用 Logits / Decisions 接口；
7. `TPS_true_flush` 口径：生产协议口径（359）两臂均不可得（无逐 token 时间戳门）；只有实验门
   （`STRATA_QWEN35_DECODE_TIMESTAMPS`）打开时（355/358）才能复算，且那是研究配置；
8. L 案 128/256 长输出在**新 exe** 上的复测（347/348 测的是旧 exe，本新构建尚未复测）。

另：359 的 decode ~15–20 tok/s 是**生产协议口径**（无 MTP/SMALL_T/COMPACT 等实验门）的实测值，
不是产品 TPS，也未引用 30–60 档目标数字；MTP/COMPACT 研究线在本阶段**未做质量认证**。
