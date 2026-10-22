# 主管复核：2026-10-22

C2/C3已从逐请求时间戳、实际argv及12份lease原件独立复核：12native/228协议完成，exit0、子树清、fence无残留。C2主业务收益不足5%；C3首128 E2E+4.89%、256+3.24%/+3.68%，仍不足5%，只记小样本方向线索，不默认t10。单卡研究结果NO-GO for adoption；不要求长文本逐Token恒等。

更正：Go observer实际134HTTP200、1HTTP429（累计17），不是worker报告的0新增/unknown；CPU7/6与harness2/1超限保留，不追认预算合规。发布可读方案表及独立重算证据见[评审包](research/deepinfer_20261021/README.md)。原始报告以下按历史保留；当前无新GPU/源码修改、main和生产默认不变。

# DeepInfer专项优化候选登记

当前性能候选：0/2；尚无新GPU诊断或可回收关键路径证据，不预先宣布优化成功。

DI02资源门（2026-10-21）：实际legacy native到READY但commit headroom最低510.7MiB<1024，守卫中止，0协议/0有效样本；工具已补preflight admission与真实launch ledger，portable C++ emitter回归通过。当前只读admission仍为commit 10773.5MiB<15024MiB登记floor，GPU未授权。资源未变化前不重跑native、不降门、不减槽、不关其它程序；本轮不是算法NO-GO，也没有速度结论。

DI01定点修复完成并主管审核后，DI02需另行登记唯一GPU租约、固定66a3模型/同binary开关诊断、3200槽和确切输入，测冷/热、短输出、128/256、4K SCORE与SCORE→Continue。逐请求DONE前延迟与整批wall/请求间隙同时记录，因为诊断stderr在DONE flush之后。字段缺失是未知，不是零；重叠计时不得相加。诊断开关扰动与产品收益分列。

再按真实关键路径至多选两个单变量候选；业务E2E默认至少5%且超过漂移，无不可接受Prefill/质量/状态/资源退化，否则NO-GO并保留基线。无新证据不重开ring/WC/PREAD/预取/core0/DOWN21/2200槽/GU未复现路线。生产默认/main/冻结件保持。

## DI02审计收口（2026-10-21）

- CPU已验收：最终36项（21 probe+15 diagnostics）一次跑过；receipt audit已把未知`protocol_sent/completed`保持null并在`*_unknown_receipts`显式计数，已知0仍为0；`lease_attempts`按唯一receipt identity去重，复制receipt不再算第二次launch；缺receipt单列。未改live测量、engine、阈值。
- 资源仍阻断：当前03:51:58Z commit available 10603.6MiB < `max(14000,spec)+1024`=15024MiB；无测速/新候选，0 GPU/0模型/0引擎build。
- 资源口径：15024MiB是保守commit准入floor，不是15GiB物理RAM需求；历史DI02加载期commit最低510.7MiB是瞬态，不能外推全部Prefill上限。详见`DEEPINFER_SINGLE_GPU_BOTTLENECKS.md`分项表与未知项。候选登记仍为0/2。

## DI04 候选取舍（2026-10-21）
- C2（defaultOFF，宿主hit-overlap查询消除）已实际完成CPU/build/6-native ABBA，结论NO-GO、保持OFF；不得把短EOS 6.01%或L0 2.29%当作≥5%业务收益，也不得重跑洗预算。
- C3（threads 8→10，单变量、无算法flag）已实际完成6-native ABBA：长输出decode TPS +5~8%、L0 E2E +4.89%、L256 +3.24~3.68%，未达预登记≥5% E2E门槛；且t10 L1 token hash与t8不同、B1/B2彼此不同。结论NO-GO默认化，保持threads 8；仅作为未来长输出重复/确定性研究的lead，不安装。
- 禁止把ON/OFF漂移、短续写或token hash包装成生产/质量/算法PASS；旧cofetch partial失败保留、不重跑。包内CPU/harness命令各超登记1次（保留失败断言），已记录。
- 证据：results/deepinfer-20261021/di04-autonomous-long/{c2-skipprobe,c3-threads10,checkpoint.json,final-report.md}。
