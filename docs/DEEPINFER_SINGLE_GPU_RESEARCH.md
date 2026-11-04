# 主管复核：2026-10-22

C2/C3已从逐请求时间戳、实际argv及12份lease原件独立复核：12native/228协议完成，exit0、子树清、fence无残留。C2主业务收益不足5%；C3首128 E2E+4.89%、256+3.24%/+3.68%，仍不足5%，只记小样本方向线索，不默认t10。单卡研究结果NO-GO for adoption；不要求长文本逐Token恒等。

更正：Go observer实际134HTTP200、1HTTP429（累计17），不是worker报告的0新增/unknown；CPU7/6与harness2/1超限保留，不追认预算合规。发布可读方案表及独立重算证据见[评审包](research/deepinfer_20261021/README.md)。原始报告以下按历史保留；当前无新GPU/源码修改、main和生产默认不变。

# DeepInfer → Strata 单卡8GiB映射（DI01初稿）

来源：`references/deepinfer-20261021`，pin `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。本文只把DeepInfer当数据，不执行其脚本/模型；当前生产GGUF无置信头，禁止把其MTP接受率调度直接套到Strata。

## 1. 机制与证据

| DeepInfer机制 | 证据 | Strata对应调用边 | 本机适用性 |
|---|---|---|---|
| DFlash `ConfidenceHead`预测每个draft token的接受概率 | `deepinfer/model_executor/models/mtp/dflash.py:46-73`（类与`confidence_head_fused`）、`:159`（`enable_confidence=True`）、`:279-338`（`sample_drafts`写`confidence_rates`） | 无。Strata生产Qwen3.6 GGUF无该head/权重；`qwen35_mtp`是独立draft/verify路径 | **排除伪造**：可观测接受/验证计数，但不能输出置信率或据此调度 |
| 按历史接受率动态分配每请求draft长度`seqlen_q` | `csrc/engine/mtp_varlen.hpp:155`（`1 + draft_tokens.size()`）、`:160-207`（用`accept_rates`/`prev_accept_rates`两轮值算best_bsz与每seq长度）；`worker/worker.py:219`（`compute_seqlens_q_by_confidence_rates`） | 潜在候选：Strata MTP draft/verify轮（当前生产MTP OFF） | 需先有真实接受率/轮级延迟；本批0GPU，不登记性能候选 |
| `MtpState`保存draft/accepted/accept_rates/confidence_raw及schedule/process轮次 | `csrc/engine/sequence.hpp:118-139`；`llm_engine.py:699-729`回传`accepted_token_ids`与`next_n_token_confidence_rates` | `Qwen35BridgeStats`仅暴露Decode层/hit/miss/phase；无MTP轮级字段 | **可借结构**：请求/轮边界差分、accepted/offered分离；不引入新MTP状态 |
| StatsLogger聚合每步接受率、verified长度、置信误差 | `engine/stats_logger.py:211-236`（`_nextn_stats`）、`:304-309`（日志字段） | 现有T1_HOST_TIMELINE/POOL_PHASE与`strata_diagnostics` | **适用**：按请求输出有界JSON，明确available/null；不新建第二套profiler |
| Worker每步Forward→Post→Sample，ZOS把post延后一轮 | `worker/worker.py:52-54`、`:96-113`、`:143-187` | Strata serve的prefill/decode循环 | 只借“阶段边界”概念；Strata无ZOS重叠，不得相加阶段计时 |
| DSpark/Markov head、YoCo mask/compact | `v4_modules/heads.py:26-46`；`examples/yoco/README.md:3-41`（要求`n_prefill_layer < n_layer`的V4.1 HF ckpt） | 无对应模型 | **直接排除**（不同模型/需要权重） |
| SM100多卡基准 | `README.md:35`、`:83`（4/64 SM100） | RTX4060 Laptop 8GiB单卡 | **排除外推**，上游数字不换算本机 |

## 2. 本批落实的可观测边界

- gate：`STRATA_QWEN35_DECODE_DIAGNOSTICS=1`，严格等于`1`才开；默认OFF不取时钟、不调`qwen35_bridge_stats`、不写stderr、stdout/DONE字节不变。
- 复用`Qwen35BridgeStats`（`qwen35_bridge.hpp:60-89`）与`NativeTierStats`（`expert_source.hpp:464-496`），在serve请求边界取快照，按`prefill`/`decode`两段差分。
- 明确标注：`ms_ids`是host侧等待，包含上游GPU依赖，不是GPU busy；`ms_*`窗口可重叠，禁止相加当E2E；Decode H2D次数/字节没有现成counter → JSON null + `available:false`。
- `DONE`的`hits/lookups/tier`是协议占位零，不是测量；诊断JSON的`legacy_done`字段显式置null并说明。

## 3. 不适用/后续

- 置信率调度：生产GGUF无confidence head，**不实现、不伪造**。只有将来引入带head的模型并用真实接受率验证后才评估。
- YoCo/DSpark/SM100：模型或硬件前提不成立，排除。
- 下一批测量只做现有serve的off/on ABBA，采集E2E/TTFT/ITL/TPOT与诊断JSON；不把本批未运行的仪器开销或GPU收益写成结论。

## 4. DI01修复与验证记录（2026-10-21）

- review发现：实际`RequestDiag` C++ emitter在刚打开`counter_delta`/`cache`对象后把`first/cfirst`初始化为false，首字段前多输出逗号（`{,`），ON时整行不是合法JSON；旧Python测试用手写fixture，未覆盖实际C++输出。
- 修复：仅将`qwen35_serve.cpp`两处初始化改为true；stdout/DONE/state/数学/字段顺序/default-OFF均不变。源码SHA256由`1822cf02...`变为`e146d9e6...`。
- 验证：把实际生产块（pre/post）提取到CPU-only MSVC harness，仅stub `core::Qwen35BridgeStats`、`qwen35_bridge_stats`、`qwen35_native_enabled`；修复前GEN输出JSON decode失败，修复后GEN/SCORE/continuation/unavailable/native-off/cache通过，OFF零stderr。原始source/argv/stdout/stderr在`results/deepinfer-20261021/di01-review-repair1/harness/`。
- 增量引擎build成功（1/1，3步），新exe SHA256 `4f9311b483021fc9cf71055ba7d44d6fa4705ed53e7a240896ff0b00873fe1bf`。本批0 GPU/0模型：诊断仍未在真实serve跑过，仪器开销、JSON在DONE后的flush与真实计数均未验证；无速度结论。
- 既有Python测试未在本批重跑（3次CPU测试调用已用于harness及其重试，Python源码未变、上一轮9/9通过）；这是本批明确缺口，不是通过。

## 5. DI02资源守卫中止与工具定点修复（2026-10-21）

- 实际GPU结果只有1次legacy-ea06 native lease：二进制实际启动（pid 10028）、模型加载到`READY`、exit0、子树0、fence未保留；0条GEN/SCORE/STATE发送。运行中`commit_available`最低510.7MiB<既有1024下限，`ResourceMonitor`正确锁存并中止。旧probe的`native_launches:0`来自`len(successful_results)`，是错误计数；原件未覆盖，主管更正为1次native/model-load attempt、0成功arm、0协议请求。
- 计数修复：新增持久化ledger，分列lease attempt/binary launch/model READY/successful arm/protocol attempted/completed；构造失败时读取raw `native/result.json`与child log，确认launch才计native，缺少receipt时记`unknown`而不是0；每命令（含STATE）发送前检查per-native24与总120并先落盘；失败命令仍消耗attempt；`--max-arms`只接受1..已登记预算，不做负切片。
- 资源门修复：preflight在任何native前调用既有`ex.fingerprint`/`ex.admission`和`ResourceMonitor`当前RAM/commit/NVML；模型SHA使用登记值、只读header不重hash；失败写guards/reasons并fail-closed。禁止降阈值、减槽、改pagefile/关程序或秘密重试。
- 口径修复：GEN显式校验`DONE produced == len(tokens)`且`0<=n<=max_new`（含EOS），不放松EOS/cursor/session。
- portable回归：`deepinfer_emitter_regression.py`按稳定marker从真实`qwen35_serve.cpp`提取gate+`RequestDiag`块（当前lines143-322，block SHA256 `88e6500b...`），CPU stub core、MSVC一次实际编译，GEN/SCORE/cont/unavailable/OFF全部通过；harness exe `56dfa56b...`，未放入生产或提交。
- 本批0 GPU/0模型/0引擎build，因此仍无诊断ON/OFF真实开销、无TTFT/Prefill/Decode/E2E新数据，也无算法NO-GO结论。

## 6. DI02 receipt审计与资源口径（2026-10-21，CPU/只读）

- 修正：`reconcile_receipts`不再用`value or 0`把缺失的`protocol_sent/completed`变成0；任一唯一entry未知则aggregate为null，并单列`*_known_lower_bound`与`*_unknown_receipts`。`lease_attempts`按唯一receipt identity去重，复制receipt不算第二次launch；缺receipt与无identity receipt显式计数。历史raw原件未改。
- 验证：最终36项CPU（21 probe+15 diagnostics）一次通过；历史DI02 raw现为`protocol_sent=null`、known lower bound 0、unknown 1、`lease_attempts=1`、`binary_launches=1`、`model_ready=1`。
- 资源：本run走`--serve`的`FileExpertSource`，GGUF只读file mapping（非private commit）；3.66GiB专家tier是device分配。历史47样本commit 9686.1→510.7→9157.7MiB为加载期瞬态，physical available最低7906.95MiB、RSS峰值2770.28MiB；9.17GiB commit下降的driver/pinned/其他进程分项仍未闭合。
- 准入：`estimated_host_mib 14000`是既有登记的历史保守host/RSS估计，admission将其作为commit headroom（+1024）而非物理RAM最低需求。当前快照commit 10603.6MiB<15024MiB，保持0 GPU/0模型/0build，不降门、不重试；无性能结论。

## DI04 宿主查询消除与线程配置研究记录（2026-10-21）
- C2：精确STRATA_QWEN35_SKIP_HIT_OVERLAP_PROBE=1仅移除单token回调中用于hit_overlap/hit_late/hit_inflight的native_ev_hit_done record+两次query；非fence、无数据依赖，默认OFF保持原序列，ON时三计数为null而非0。方法：实际C++控制块提取+mock CUDA harness（off/on/skip）→63项单测→单次隔离build→2诊断ON smoke+4诊断OFF ABBA，同binary/模型/3200/t8/u4096/c4096/MTP OFF。
- C2结果：主P/S/L/L256/SCORE增益-0.35%~+2.29%，均在1.60%~4.13%同块漂移内；仅短EOS续写+6.01%（n=1）超5% → NO-GO业务结论，token hash相等仅诊断。
- C3：同binary、仅每臂config threads 8 vs 10（无cofetch/skip-probe），另6 native/114协议。L0 E2E +4.89%（drift1.27%，decode TPS+8.2%）、L2561 +3.68%（decode+5.0%）、L2560 +3.24%，P/S/SCORE约0；无主工作负载达到预登记≥5% E2E门槛，故不默认t10。L1 token hash在t8/t10间不同且t10的B1/B2彼此不同（结构门仍过），作为确定性与质量风险记录，等未来重复长输出研究。
- 原件：results/deepinfer-20261021/di04-autonomous-long/{c2-skipprobe,c3-threads10,checkpoint.json,final-report.md}。
