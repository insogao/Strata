# qwen35_llama：实测 profile（长输出 4096 槽）

本页说明 `profiles/long-output-4060-4096.json` 这一份**研究用实测 profile**：它是什么、什么时候能用、
实测数字是多少、什么情况下立刻失效、怎么套到引擎 argv 上，以及哪些结论还没闭合。
配套工具：`tools/check_profile.py`（纯 stdlib、CPU、零依赖、**不碰 GPU**）。

口径先行：全部数字来自本机（RTX 4060 Laptop 8 GiB、Windows 11、混合显卡模式）**单机小样本**实测，
每臂 2 个 native（1 冷 1 热，判热样本），每形状每次 native 只有 n=1。**不是统计显著结论，不给 p95，
不跨 GPU / 跨量化换算。** 生产默认仍是 3200 槽；本 profile 是**按形状限定**的可选档，
不是产品默认、不是合并/通过声明。

## 1. profile 是什么（语义与字段）

一份 profile = "在**这一套** model / exe / 设备 / argv / 门组合下，某个形状的实测结论"。
它不是调参建议，也不是自动套用的配置；它自带失效条件，条件不满足就必须弃用。

| 字段 | 本 profile 的值 | 语义 |
|---|---|---|
| `schema_version` | `strata-profile/1` | 研究 profile 模式；与 P3-4 的 `strata-tune-profile/1` 见 §6 映射 |
| `applies_to` | `output_tokens_min: 128` | **只对长输出成立**；低于该值不适用（§2） |
| `model` | `66a3ca88…`（`UD-IQ3_S`，13,676,723,168 B） | 已登记、此前全量核过的 SHA；本 profile 不再重算 hash |
| `engine` | exe `6c9e297f…`，source `03528a3c`，VS2022+CUDA 12.8.1 | 数值与二进制绑定；**不是本仓 HEAD 的构建**，套用前必须核 exe SHA |
| `device` | 4060 Laptop / 8188 MiB / 驱动 591.74 / sm 8.9 | 设备指纹；跨卡一律拒绝 |
| `argv` | 引擎参数逐字（见 §5） | 唯一相对 3200 默认档的差异是 `--qwen35-cache-slots 4096` |
| `env` | 生产三项（`ROUTER_WEIGHTS_DEVICE/PACKED_IDS/EXPERT_ORDER`） | 实测时的环境；未见其它门 |
| `gates` | `compact_slots/mtp/mtp_txn` 全关 | 任一开启即失效 |
| `evidence` | `results/multiagent-20261104/p1-slots/{F,G}/`（另含 A 基线、H 4224 参考） | 证据在**本地研究仓**，`results/` 不上 GitHub |
| `confidence` | same-day confirmed（+13.67%、+18.66%）；next-day pending | 同日两臂复现，**次日复核未做** |
| `recommendation` | `false` | 证据 + 同日复现 ≠ 推荐；不是选优依据 |
| `invalidators` | 7 条结构化规则（§4） | 由 `tools/check_profile.py` 逐条机器校验 |

## 2. 适用条件

- **适用**：长输出解码（本次形状 L：3896 token prompt + **128** 新 token，serve 路径，本机本 exe）。
- **不适用**：短 JSON（形状 S：max-new 16）。同两臂实测 **S16 −1.28%（F）/ −0.38%（G）**，
  4224 档 **−1.83%（H）**：槽数增加只买到长输出的 decode 收益，短输出反被挤占、无收益。
  因此**短 16-token JSON 继续用 3200 槽**，本 profile 不得作为默认档。
- **不适用**：输出 < 128 token 的任何请求；`OUTPUT_TOKENS_BELOW_MIN` 会在 check 阶段直接判 FAIL。
- **上下文**：实测 `--max-context 4096` 不变。**没有**把 context 抬到 8192，也没有验证 256 输出档；
  不能把本 profile 外推成"更大 context 也这样"。

## 3. 实测表（3200 vs 4096 vs 4224）

同一身份：exe `6c9e297f…`、模型 `66a3ca88…`、t8 / ub4096 / c4096、MTP OFF、COMPACT OFF、serve 路径；
一 native 一租约、严格串行、全部 exit 0、无 fence。基线为 3200 槽臂（A）。

| 臂 | 槽 | L128 热 TPS | ΔL128 | S16 热 TPS | ΔS16 | held 热 TPS | Δheld | Prefill 热 TPS | ΔPrefill | GPU 峰值 MiB | 余量 MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A（基线） | 3200 | 44.82 | — | 52.06 | — | 45.10 | — | 1820.3 | — | 6670.6 | 1517.4 |
| F | **4096** | 50.95 | **+13.67%** | 51.39 | −1.28% | 49.67 | +10.13% | 1807.5 | −0.71% | 7720.6 | **467.4** |
| G | **4096** | 53.18 | **+18.66%** | 51.86 | −0.38% | 47.39 | +5.08% | 1818.5 | −0.10% | 7720.6 | **467.4** |
| H（参考） | 4224 | 52.19 | +16.45% | 51.11 | −1.83% | 48.57 | +7.70% | 1796.7 | −1.30% | 7870.6 | 317.4 |

读数说明：

- "热"= 同 native 内第 2 个样本（冷/热两样本，判热）；`held` 为 post-flush 段的实际测量值。
- **GPU 峰值是 100 ms 采样的整卡高水位（sampled whole-card high-water），不是分配器精确峰值**；
  余量 = 8188 − 峰值，因此只是同一口径下的相对判据。
- 槽数与显存峰值在本机呈严格线性：每 +128 槽 ≈ +150 MiB（A 6670.6 → 4096 的 7720.6 = 7×150；
  4224 的 7870.6 = 8×150），与实测两点一致，属同机观测，不外推别的卡/别的量化。
- 4224 不比 4096 快，且余量只剩 317.4 MiB，故**选 4096**、弃 4224。
- 百分比按 rig 记录的原始值；用上表 2 位小数 TPS 反算会有 0.01–0.02 pp 的舍入差。

## 4. 失效规则（任一命中即失效，不做换算、不静默沿用）

| code | 触发条件 | 判定来源 |
|---|---|---|
| `MODEL_SHA_MISMATCH` | 目标模型 SHA 前缀 ≠ `66a3ca88`；目标 SHA 未知 = 失效（fail closed） | 调用方提供的 SHA |
| `QUANT_MISMATCH` | quant ≠ `UD-IQ3_S`。**禁止跨量化换算**：槽数/显存/收益都不可移植 | fingerprint `quant` |
| `DEVICE_MISMATCH` | GPU 型号 / 显存总量 / 驱动 / 计算能力任一变化。**跨卡一律拒绝**，不缩放、不换算 | fingerprint `device` |
| `ARGV_MISMATCH` | `--qwen35-cache-slots 4096`、`--qwen35-threads 8`、`--qwen35-ubatch 4096`、`--max-context 4096` 任一与运行配置不一致 | fingerprint `argv` |
| `GATE_TOGGLED` | COMPACT（`compact_slots`）或 MTP（`mtp`/`mtp_txn`）任一开启 | fingerprint `gates`/`env` |
| `OUTPUT_TOKENS_BELOW_MIN` | 请求输出 < 128 token（短 JSON 不适用，见 §2） | 调用方 / fingerprint `output_tokens` |
| `HEADROOM_BELOW_MIN` | 目标机实测整卡余量 < 300 MiB | fingerprint `headroom_mib`（或 `vram_total_mib − gpu_peak_mib`） |

`tools/check_profile.py` 按 profile 里的 `invalidators` 逐条执行：**profile 声明了它没实现的规则 →
直接 FAIL（fail closed）**，不会因为"看不懂就放过"。

```text
python tools/check_profile.py --profile profiles/long-output-4060-4096.json \
    --fingerprint-json fp.json            # -> OK / FAIL: <code>: <reason>; rc 0/1
```

fingerprint 由调用方提供（本工具不探测、不开 GPU）：

```json
{"model_sha256": "66a3ca88…", "quant": "UD-IQ3_S",
 "device": {"gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "vram_total_mib": 8188,
            "driver": "591.74", "compute_capability": "8.9"},
 "argv": ["--qwen35-cache-slots", "4096", "--qwen35-threads", "8",
          "--qwen35-ubatch", "4096", "--max-context", "4096"],
 "headroom_mib": 467.4, "output_tokens": 128,
 "gates": {"compact_slots": false, "mtp": false, "mtp_txn": false}}
```

`tools/strata_fingerprint.py`（P3-1）产出的 `{"value","source"}` 指纹（顶层 `gpu` 而非 `device`）
也可直接喂进去；`headroom_mib` 缺失时用 `vram_total_mib − gpu_peak_mib` 计算，两者都没有则 FAIL。

## 5. 如何应用（引擎 argv + 环境）

相对 3200 默认档，**只改一个参数**（`--qwen35-cache-slots 3200` → `4096`）：

```text
--qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models/pack --serve \
  --max-context 4096 --qwen35-threads 8 --qwen35-cache-slots 4096 --qwen35-ubatch 4096
```

```text
STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1
STRATA_QWEN35_PACKED_IDS=1
STRATA_QWEN35_EXPERT_ORDER=count_desc
```

套用前必须过三道门：①`tools/check_profile.py` 返回 `OK`；②exe SHA = `6c9e297f…`；
③启动日志出现 `native expert tier: 4096 slots`（实测 `actual-configuration.json` 记录的实际值）。
任何一道不过就回 3200，不做"折中档"试凑。

## 6. 未来 `strata tune` 的接入点

P3-4 的 `strata_tune.py` 用的是 `strata-tune-profile/1`（字段 `model_sha256_prefix` / `quant` /
`device_fingerprint` / `argv{slots,ubatch,threads,ctx,compact}` / `confidence: ok|wide|uncalibrated` /
`recommendation` / `invalidators[{code,rule}]`）。本 profile 是同一信息的研究侧写法，映射关系：

| `strata-profile/1`（本文件） | `strata-tune-profile/1` |
|---|---|
| `model.sha256_prefix` | `model_sha256_prefix` |
| `model.quant` | `quant` |
| `device.{gpu,vram_total_mib,driver,compute_capability}` | `device_fingerprint.{gpu_name,vram_total_mib,driver_version,compute_capability}` |
| `argv`（引擎参数表） | `argv{slots:4096, ubatch:4096, threads:8, ctx:4096, compact:false}` |
| `evidence.primary/supporting` | `evidence`（≥1 条；无证据只能写 `uncalibrated`） |
| `confidence`（人读字符串，含 pending） | `confidence: wide`（同日两臂、次日未复核时**不得**写 `ok`） |
| `invalidators`（7 条） | `invalidators`（P3-4 现为 5 条＝同码 4 条＋`UNCALIBRATED`；本文件多出 `GATE_TOGGLED`/`OUTPUT_TOKENS_BELOW_MIN`/`HEADROOM_BELOW_MIN`，需并入其规则表才生效；`UNCALIBRATED` 在本文件不适用，因为 `confidence` 是人读字符串而非三值枚举） |
| `recommendation: false` | `recommendation: false` |

接入点（不改 P3-4 现有语义，只加一个入口）：

1. `strata_tune.py check` 继续只认 `strata-tune-profile/1`，**不要**把本文件直接喂给它；
   需要时先用上表机械转换（纯字段搬运，不做任何数值推断）。
2. 建议新增 `strata_tune.py check --profile-schema strata-profile/1`（或一个 `convert` 子命令），
   让同一条 check 路径既能校验 P3-4 生成的 profile，也能校验研究侧 profile。
3. `applies_to.output_tokens_min` 这类**形状门**是 P3-4 目前没有的概念（它按 workload 参数判），
   接入时要把"输出 < 128 即不适用"作为独立拒绝码，而不是折进 `ARGV_MISMATCH`。
4. `confidence=wide`（同日复现、次日未复核）应当是**上限**：本 profile 在次日复核完成前不进 `ok`。

## 7. 未闭合

- **次日复核**：同身份、同 fixture 的 4096 vs 3200 复跑未做。当前只有同日两臂（+13.67% / +18.66%），
  置信上限就是 `wide`。
- **256 输出档**：未测。它需要合法更大的 context（c8192 级），不能从 128 档外推。
- **并发**：未测。单请求 serve 路径的结论不覆盖并发/多会话。
- **cache 命中率**：serve 路径的 hits/misses 计数为占位 0，未用于判定；真实命中率需 CLI/MTP 管线另立项。
- **样本量**：每臂 2 native、每形状 n=1，B 臂内部极差曾达 6.4%，故"同日两臂都 ≥5%"只是**下限证据**。
- **收益来源**：只观测到"L128 随槽数单调上升、显存线性增长"，未做逐层成本归因，不编分层收益。

## 8. 证据位置与复现

| 内容 | 位置 |
|---|---|
| profile 源数据（研究仓） | `results/multiagent-20261104/profiles/long-output-4060-4096.json` |
| 各臂原件 | `results/multiagent-20261104/p1-slots/{A,F,G,H}/`（`identity.json`、`native-*/result.json`、`raw/`、`actual-configuration.json`） |
| 汇总报告 | `results/multiagent-20261104/p1-slots/P1_REPORT.md` |
| 逐条校验 | `tools/check_profile.py` |

`results/` 属本地研究仓，**不在 GitHub**；上面的 SHA / MiB / TPS 均可在对应 `result.json`
与 `identity.json` 里逐项复查。跨机器复现时先过 §4 的 7 条失效规则——**不满足就不要用这份 profile**。
