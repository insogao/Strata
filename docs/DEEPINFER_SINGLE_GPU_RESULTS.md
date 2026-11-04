# 主管复核：2026-10-22

C2/C3已从逐请求时间戳、实际argv及12份lease原件独立复核：12native/228协议完成，exit0、子树清、fence无残留。C2主业务收益不足5%；C3首128 E2E+4.89%、256+3.24%/+3.68%，仍不足5%，只记小样本方向线索，不默认t10。单卡研究结果NO-GO for adoption；不要求长文本逐Token恒等。

更正：Go observer实际134HTTP200、1HTTP429（累计17），不是worker报告的0新增/unknown；CPU7/6与harness2/1超限保留，不追认预算合规。发布可读方案表及独立重算证据见[评审包](research/deepinfer_20261021/README.md)。原始报告以下按历史保留；当前无新GPU/源码修改、main和生产默认不变。

# 最新完成与续接 2026-10-21T08:31:01.188276+00:00
DI02最终三组全部COMPLETE：ONB1/ONB2/OFFA2各19协议、exit0、cleanup无fence，共57操作；每native约54/54/57秒，因此模型已正常卸载。历史累计7native/102完成操作（含之前失败/部分请求），诊断GPU预算已耗尽不重跑。ON/OFF指诊断，MTP始终OFF。P1 1702.50–1788.63TPS，L1 33.60–43.96TPS，L2561 37.56–40.75TPS；SCORE1 2.073–2.084s。OFF端点/插入间隔有明显漂移，不能宣称算法收益或精确仪器扰动PASS。实际EOS/长度/同病例tokenhash、SCORE->Continue已留原件；小样本不是业务准确率/全状态Gate。逐命名请求复核：results/deepinfer-20261021/di02-final-three/supervisor-per-case-review.json，不使用混入held/EOS的shape聚合。
下一候选1/2登记 D:/work/Strata魔改/results/deepinfer-20261021/di03-id-act-cofetch/manifest.json：仅expert_source单token cached-tier ID+activation同流cofetch(defaultOFF)，先核旧证据/依赖与CPU，再单次独立build-deepinfer-di03；本批0GPU，旧4f9311/ea06和生产默认冻结。不是再次静态内存调查，非权重预取。主管后续另登记同binary OFF/ON实际验收。原ACTIVE20不变。

# DI04 自动长包结果 2026-10-21T10:12Z
C2（skip-hit-overlap-probe，defaultOFF）已完成1次隔离build、6 native/114协议ABBA：全部exit0/cleanup清，结构门/EOS/JSON/STATE/hash通过；主P/S/L/L256/SCORE E2E增益-0.35%~+2.29%，均在1.60%~4.13%同块漂移内，仅17-token EOS +6.01%（n=1）超5% → NO-GO、保持OFF。C3（同binary、仅threads 8 vs 10、无算法flag）另6 native/114协议：L0 +4.89%（漂移1.27%，decode TPS +8.2%）、L2561 +3.68%、L2560 +3.24%，P/S/SCORE约0；无主工作负载达到≥5% E2E门槛，且t10的L1 token hash与t8不同、B1/B2彼此也不同（结构门仍过）→ 不默认t10，记为长输出decode方向的后续线索。GPU peak两轮均7750.11MiB，无OOM/NaN/state失败。包总计12 native/228协议、2个独立1200s块；CPU/harness命令因保留失败断言各超1次，已如实记录。未commit/push/PR。证据：results/deepinfer-20261021/di04-autonomous-long/{checkpoint.json,final-report.md,c2-skipprobe/,c3-threads10/,cpu-skipprobe-harness/,build-deepinfer-long.log}；docs仅追加，不覆盖历史。

# DeepInfer专项当前结果（主管事实记录）

|方案|考虑与改动|结果与取舍|证据|
|---|---|---|---|
|DI01 请求级诊断|复用既有累计counter，默认关闭，Prefill/Decode差分，未知H2D为null|初次隔离build成功；CPU9项最终通过；主管发现实际C++ JSON首逗号问题，未通过测量门；已定点修复并CPU harness验证|本地 results/deepinfer-20261021/di01-observability/codex-review-20261021.md 及 di01-review-repair1/|
|DI02 资源守卫|legacy ea06 native实际启动并READY，0协议发送；commit最低510.7MiB<1024触发既有监控|守卫正确中止：exit0、子树0、fence清；无OOM、无速度/TPS、无算法NO-GO；旧probe native=0为错误计数|results/deepinfer-20261021/di02-diagnostic-paired/ 原始probe-run/result.json、native/result.json、supervisor-actual-counts.json|
|DI02 工具/资源门修复|preflight接入fingerprint/admission+NVML/commit；launch/protocol ledger；发送前预算；portable真实C++块回归|CPU 19+15项通过；实际块一次MSVC编译后GEN/SCORE/cont/unavailable/OFF通过；0GPU/0模型/0引擎build；当前commit10773.5MiB<15024MiB，仍不授权GPU|results/deepinfer-20261021/di02-tool-resource-repair/ receipt-audit.json、admission-snapshot.json、emitter-regression/result.json|
|DI02 receipt未知/去重修正+资源口径审计|未知protocol不再`or 0`；lease按唯一receipt identity去重；补missing/unknown计数；只读核查RSS/file-backed RO/private commit/pinned/device分项|最终36项CPU全过；历史raw审计`protocol_sent=null`、known lower bound 0、unknown 1、lease_attempts=1；当前commit10603.6MiB<15024MiB，0GPU/0模型/0build，无性能结论|results/deepinfer-20261021/di02-resource-accounting-audit/ receipt-audit-after-fix.json、final-tests.log、source-sha256.json|

初次exe SHA256：8e07883a87b04a344e03bfc4422b996020543bce133ae9477573a13c7c829e8e。分支codex/deepinfer-single-gpu-20261021，base0a9389d2。0模型启动、0GPU测量，TTFT/Prefill/Decode/E2E/VRAM新数据均尚未测量。初次测试1次错误断言失败保留，最终9项通过不证明实际C++输出正确。

历史8热样本Prefill1761.32TPS、128Decode40.74TPS仅参考。最近明显改善来自用户解除GPU功耗限制/野兽模式，不属于本专项代码收益。无锁功率反向A/B，不能分配精确功耗因果比例。后续一律同日同模型/构建配对。

原有限guard在原子build结束后终止executor，未获checkpoint；这是主管收尾宽限设计问题，单独记录，不当作GPU或模型功能失败。原件保留。下一有界修复另有明确证据与预算，不抹除累计构建/测试数。

DI02本批没有启动模型/引擎/GPU native；只读admission在03:38Z记录commit available 10773.5MiB、GPU used 1696.34/8188MiB、physical available约19153MiB。由于10773.5<15024MiB（14000登记host estimate+1024 floor），preflight按设计拒绝，0 lease、0 binary、0协议。该拒绝是资源事实，不是性能/OOM/算法结论。累计保留：2次engine build、1次实际GPU native（DI02 legacy，0请求）、16个HTTP429；本批没有新增429、没有commit/push。

2026-10-21审计批（results/deepinfer-20261021/di02-resource-accounting-audit/）：最终36项CPU一次跑过；历史raw receipt现报告`protocol_sent=null`、已知下界0、unknown receipt 1，`lease_attempts=1`，不再把未知伪装成零。只读快照03:51:58Z commit available 10603.6MiB<15024MiB；47样本commit最低510.7MiB是加载期瞬态且随后恢复，不能推全部Prefill上限。`estimated_host_mib 14000`是历史保守commit估计，不是15GiB物理RAM需求；当前0GPU/0模型/0build、0新429、无性能或算法结论。

主管收口（2026-10-21T04:06Z）：最终sourceSHA与36项CPU raw日志已核，portable实际C++回归已核；无新的功能修改或GPU重复。当前commitavailable12593.2MiB<15024MiB。CPU观测工具完成不等于优化完成，性能候选0/2，待外部资源变化后继续；不重复派发相同静态工作。原20min监督保留，main/生产默认不动。

## 2026-10-21同日真实基线（主管从原件恢复）

在生产66a3 UD-IQ3_S、ea06原exe、3200/t8/u4096/c4096/MTP OFF下，一进程完成19协议操作，exit0/子树清/无fence，模型随后卸载。测量工具收尾读取资源目录错误，不代表这些原生请求失败；原失败不覆盖。

|用例|本次数据|口径|
|---|---:|---|
|P1 Prefill|1759.87 token/s|3896输入/out1，第二次P；1个样本|
|L1 Decode|37.96 token/s|3817输入/实际128输出，length结束；完整请求5.432s|
|L2561 Decode|39.02 token/s|3817输入/实际256输出，length结束；完整请求8.581s|
|SCORE0/1|2.092/2.095s|3896输入，完整原生协议至DONE，不含HTTP/tokenizer|
|真实EOS|17 tokens/stop|独立续接冒烟，末token248046，cursor检查通过|
|整卡显存采样峰值|7566.72MiB|100ms采样，不是精确allocator峰值|
|进程RSS采样峰值|15025.32MiB|含映射驻留，不能等同私有提交量|
|最小提交余量|1470.8MiB|未触发1024MiB运行守卫，startup样本不能代表全负载|

TPS基于(n−1)/(last−first)的5ms文件观察，非原生精确flush。小样本非稳态/p95；这些是旧引擎同日参考，不是DeepInfer提速。尚无新二进制诊断OFF/ON完整对照，性能候选0/2。
证据：研究仓results/deepinfer-20261021/di02-l256-retest/supervisor-baseline-recovered.json及原raw。最近明显历史提升仍来自用户显卡功耗解禁，不改归因。
