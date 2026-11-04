# 开启笔记本性能模式后的同引擎复测

2026-10-19 13:41–13:43（Asia/Singapore）。用户已自行开启性能模式；本轮不修改电源、时钟或功耗设置。**同模型/二进制/参数的本次实测明显快于10月16日历史记录；不是同日切换模式的因果配对实验，也不是引擎代码提速。**

## 身份、执行与口径

- exe：`D:/work/StrataUnified/build-session-correctness/strata.exe`，SHA256 `ea06d39411e160f5784f9c4ef160f5e9bcc1746fb92fd9b798677aabb0dd0619`；gpu-lock逐次核实际SHA。
- GGUF：生产 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`，13,676,723,168 B，UD-IQ3_S。沿用此前完整验证身份，不重复全文件哈希。
- RTX4060 Laptop8GB，driver591.74；MTP OFF、3200 uniform专家槽、t8/u4096/c4096，原3项环境开关不变。
- 2个独立冷进程，17条协议操作/进程，严格串行；2/2 exit0、清理全子树为0、无保留fence、0重跑、0build、无功能源码变动。
- 注册预算：native≤2，每次≤240s，总续步窗1200s，等锁≤60s；显存上限8060MiB。实际含控制器约96秒；load/READY分别6.47/2.48秒。
- 复用现有LeasedNative、gpu_lock及Sampler；100ms资源监控＋250ms诊断，与旧P3-11同类仪表。不是无插桩产品速度。
- P/S输入3896 tokens，L输入3817 tokens；相同fixture及SHA。顺序P0/P1/S0/S1/L0/L1，与旧诊断一致；随后held decode和真实EOS；最后追加2次完整SCORE。
- Decode TPS为 `(n−1)/(last−first)`，依据fflush后文件观察、5ms polling；不是原生精确flush。n1不算Decode TPS。n16/n128均实际达上限、finish=length，不能称为EOS完成；另有独立EOS冒烟。

## 历史与当前结果

旧参考为10月16日同exe/模型/参数P3-11诊断两次进程；新值同样取两个进程对应热请求中位。跨日缓存/后台/温度/主机电源状态并未完全控制，不把百分比解释成严格性能模式因果效应。

| 指标 | 历史参考 | 本次性能模式 | 变化 |
|---|---:|---:|---:|
| P1：3896输入、out1 Prefill | 1383.36 token/s | **1772.22 token/s** | +28.11% |
| S1：16token Decode | 37.24 token/s | **50.24 token/s** | +34.91% |
| L1：128token Decode | 31.10 token/s | **40.75 token/s** | +31.03% |
| held：已有3817前缀、无新Prefill、128token Decode | 33.96 token/s | **44.59 token/s** | +31.30% |
| S1完整协议请求，16token length结束 | 3.178s | **2.340s** | −26.37% |
| L1完整协议请求，128token length结束 | 6.913s | **5.163s** | −25.31% |
| 3896输入完整SCORE，0生成 | 2.843s | **2.082s** | −26.79% |

SCORE为命令发送到完整DONE的原生协议往返，含全部Prefill/候选概率输出；**不含HTTP、tokenizer与业务Python处理**。旧SCORE来自P3-12 W11（1个参考），新为两个进程各自第二次SCORE（2样本）；请求前历史顺序与新顺序不同。不能把它与600条短输入的HTTP B1约1.01s直接比较。

逐进程结果：

| 请求 | 进程0 | 进程1 | 条件 |
|---|---:|---:|---|
| 首次P0 Prefill | 1119.51 | 1330.37 | token/s，冷进程首次请求；加载另计 |
| P1 Prefill | 1785.76 | 1758.68 | token/s，第二请求 |
| S0 Decode | 21.12 | 18.37 | token/s，进程首次多token Decode |
| S1 Decode | 49.08 | 51.39 | token/s，后续相同请求 |
| L0 Decode | 35.64 | 38.35 | token/s，首次长输出形状 |
| L1 Decode | 39.52 | 41.97 | token/s，后续长输出 |
| held Decode | 44.16 | 45.02 | token/s，无重复Prefill；实际128tokens |
| L1 TTFT | 2.046s | 2.041s | 文件观察时钟，含完整Prefill |
| L1 TPOT | 25.30ms | 23.82ms | 首token至末token |
| SCORE第二次 | 2.081s | 2.082s | 完整协议往返，actual3896输入、0输出 |

当前冷S0仍显著慢于热S1；性能模式没有消除初始化/内存驻留相关问题。样本小，不宣称稳态或可靠p95，也没有达到60TPS长输出目标。

## 功耗、资源及正确性

跑前空闲NVML enforced power limit约134W；**负载中动态变化**，不能当作恒定134W或实际耗电。L1纯Decode窗口（排除Prefill，9个250ms样本/进程）：

- enforced limit中位约115.08/115.50W，实际功耗中位61.57/60.31W；历史相同类窗口观察约38–40W限额/功耗。
- SM时钟中位2685/2670MHz，显存时钟8201MHz；这两个窗口的采样throttle flags均0。不能排除未采样间隔或NVML计数滞后。
- 整卡采样峰值7638.89/7636.73MiB（约7.46GiB）；host RSS峰值14985.11/14986.29MiB，可用RAM最低8045.92/8133.50MiB。没有越界/OOM/CUDA错误，监控错误0。
- 两轮S0/S1的JSON均完整、risk0/violation=false，token SHA与对应历史短回答相同；这只是fixture结构检查，不是业务准确率。
- 两轮真实EOS冒烟均17tokens、末token248046、finish=stop；held状态游标为3817＋实际输出数，没有重复Prefill。
- 四次SCORE全部实际3896输入、0生成，候选概率有限；四候选保存logprob和排序与旧W11一致。此fixture的SCORE选择仍是token16/risk1，不能把生成JSON risk0当作该评分准确率已通过。
- 没有重跑600条，也不以本轮少量样本继承或提升600条质量验收结论。

## 后续影响

性能模式后的**当前基线**应作为下一次上游v0.1.45实验的同日参考；不能继续拿低功耗历史值计算上游代码收益。先保留3200/t8/u4096/MTP OFF及现有协议，再做隔离同GGUF配对。冷S0和host/GPU专家路径仍需分别研究；不因本轮明显提升便认定所有历史下降根因已经定位。

本轮无参数/原生代码改变，不需要为此单独做功能PR或重复完整CPU回归。记录作为研究分支的证据提交，main、生产冻结件、既有PR分支不动。

## 原件

本地：`D:/work/Strata魔改/results/performance-mode-retest-20261019/`

- `manifest.json` / `spec.json` / `identity.json`：注册、输入/构建身份及配置。
- `native-0/`、`native-1/`：raw/child_00.log、逐token观察transcript、NVML/host遥测、JobObject/锁/清理回执。
- `result.json` / `analysis.json` / `receipt-checks.json` / `completion.json` / `evidence-sha256.json`：原始结构化结果、复算、收尾及SHA索引。
- 历史：`results/unified-p3-next-20261016/diagnostic-result.json` 与 `D:/work/StrataUnified/experiments/next-20261016/workload-reference/baseline-profile/measurements.json`。
- 入仓摘要：[summary.json](evidence/performance_mode_20261019/summary.json)。
