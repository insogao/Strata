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
