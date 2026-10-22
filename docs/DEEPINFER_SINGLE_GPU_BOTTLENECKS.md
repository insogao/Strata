# 主管复核：2026-10-22

C2/C3已从逐请求时间戳、实际argv及12份lease原件独立复核：12native/228协议完成，exit0、子树清、fence无残留。C2主业务收益不足5%；C3首128 E2E+4.89%、256+3.24%/+3.68%，仍不足5%，只记小样本方向线索，不默认t10。单卡研究结果NO-GO for adoption；不要求长文本逐Token恒等。

更正：Go observer实际134HTTP200、1HTTP429（累计17），不是worker报告的0新增/unknown；CPU7/6与harness2/1超限保留，不追认预算合规。发布可读方案表及独立重算证据见[评审包](research/deepinfer_20261021/README.md)。原始报告以下按历史保留；当前无新GPU/源码修改、main和生产默认不变。

# 单卡8GiB Decode瓶颈观测边界（DI01）

当前事实：生产8热样本Prefill1761.32、128 Decode40.74 token/s的近期提升归因GPU功耗解禁/野兽模式，不是新代码；本批0 GPU，无新性能结论。

## 已知但未闭合

- **IDs等待**：`ms_ids`由host在router ids到齐前后计时，包含上游GPU依赖；它是host观测等待，不是GPU busy，也不能单独证明CPU瓶颈。
- **重叠窗**：`ms_launch`/`ms_misses`/`ms_combine`是阶段计时；CPU misses与GPU hits可重叠。任何把阶段相加得到的E2E都无效。
- **H2D**：现有`prefill_stream_bytes`只覆盖prompt冷ring；Decode的H2D次数/字节无counter。诊断JSON输出null，不用0占位。
- **DONE**：serve的`hits/lookups/ram/file/offloaded`是固定占位零，不能当真实缓存命中。
- **Native tier是否启用**：未启用时Decode hit/miss/phase字段不适用，诊断JSON置null并给`native_tier:false`。
- **可见性**：诊断只给请求边界差分，不能定位到逐token或逐层成本；逐层成本需另立有界探针，不能从本轮平均外推。

## 下一批可执行测量（本批不运行）

- OFF/ON ABBA：同exe、同GGUF66a3、3200槽、MTP OFF、t8/u4096/c4096；冷/热短、128/256、4K SCORE、SCORE→Continue各独立样本。
- 256用例先核`input + max_new <= 实际ctx`（例如3817+256=4073≤4096），不无理由升8192；EOS与长度截断分列。
- 记录JSON可见性、EOS完成E2E、Prefill/TTFT/ITL/TPOT、`(n-1)/(last-first postflush)` TPS、实际cache/VRAM/Graph；n<2不计TPS。
- 仪器开销以OFF/ON配对比较；本批未测，禁止先验宣称“无开销”。

## DI01修复状态（2026-10-21）

- 实际C++ emitter的非法JSON已修（`counter_delta`/`cache`两处first初始化），实际生产块CPU-only harness验证通过；但未跑GPU serve，上列各项仍是未知，不是零。
- DI02必须用同一新exe做默认OFF/ON ABBA；用exact env屏蔽旧probe；记录完整批次wall（含DONE后destructor flush）与请求间隙；同轮记录state/cache/VRAM/Graph。stats getter会扫描cache payload并可能flush其它诊断，不能只拿DONE前时间宣称无扰动。
- 本批未测仪器开销/性能，不登记性能候选。

## DI02资源事实（2026-10-21）

- 实际legacy native加载期100ms采样：整卡峰值7390.60MiB、physical available最低7906.95MiB、commit available由初始9686.1MiB降到最低510.7MiB；低于既有1024MiB硬下限，guard中止。它是host commit压力，不是GPU 8060MiB超限，也不是OOM或模型不兼容。
- 03:38只读admission快照：commit limit 37824.5MiB、commit available 10773.5MiB，仍低于已登记`estimated_host_mib 14000 + 1024`=15024MiB；fingerprint header model size 13676723168B、SHA仍为66a3登记值；GPU 1696.34/8188MiB used、physical available约19153MiB。preflight因此fail-closed，0 lease/0 native。
- 结论边界：本轮没有有效性能样本、0协议请求、0 TPS；不能据此判断ON诊断开销、Decode瓶颈或算法收益，也不能用缩小槽/降低门槛换取重跑。新GPU native必须等外部资源真实变化并由主管另行登记。

## DI02独立资源口径表（2026-10-21，只读）

- 实测快照03:51:58Z：commit limit 37824.5MiB、available 10603.6MiB；physical total 32448.5MiB、available 18485.6MiB；NVML 1465/8188MiB；swap used 782.2MiB。该快照仍<15024MiB准入，未降门。
- 历史DI02 47个100ms样本：process 6.056s、READY后exit0、0协议；GPU used峰值7390.60MiB；physical available最低7906.95MiB；commit available 9686.1→最低510.7→最后9157.7MiB；采样RSS峰值2770.28MiB。commit下降后会恢复，属加载期瞬态，不能当稳态或全部Prefill峰值。
- 源事实：本run走`--serve`的`FileExpertSource`，GGUF以`CreateFileMappingW(PAGE_READONLY)`+`MapViewOfFile`只读映射（`expert_source.cpp:1083-1084`）；模型文件13,676,723,168B，llama loader记录`CPU_Mapped model buffer size=12633.97MiB`。这是file-backed RO页，不是private commit；`ArenaExpertSource`的`VirtualAlloc(MEM_RESERVE|MEM_COMMIT)`（`pinned.cu:111`）属`generate`路径，不是本run。
- 源事实：3200槽×1,228,800B=3.66GiB是`ExpertCache`的device分配（`expert_cache.cpp:561-582`），不是host commit；bridge默认每层pinned块约74KiB（K=8/H=2048/rows=1，40层约2.9MiB，`qwen35_bridge.cpp:793-799`）；cold stager为有界ring，按blob大小计（约百MiB级，Prefill才用，本run无Prefill）。
- 未知：9.17GiB系统级commit下降无法由上述已登记私有分配完全解释；commit是system-wide采样，可能含其他进程、CUDA/WDDM pinned/driver与文件缓存口径，当前证据未分项。RSS峰值仅2.77GiB，不能据此把RSS当commit或反推private bytes。
- `estimated_host_mib=14000`来自既有登记experiment/evidence spec，是历史host/RSS保守估计；admission把它当commit headroom要求（`strata_experiment.py:226`；probe再取`max(14000,spec)+1024=15024MiB`，`deepinfer_diagnostic_probe.py:299-305`）。它是保守commit估计，不是物理RAM最低需求；本审计未找到14000的逐项来源推导。

## DI04 观测更新（2026-10-21）
- C2实测：每有hit的decode层原有1次hit_done record+2次query（L128约5080层），移除后主工作负载E2E变化-0.35%~+2.29%，与1.60%~4.13%的OFF端点漂移同量级 → 这些宿主驱动入口不是当前E2E的可见绑定成本。
- C3实测（同binary，仅threads 8 vs 10）：长输出decode TPS提升L0 +8.2%、L256约+4.4~5.0%，TPOT下降约4.8~7.6%，但E2E仅L0 +4.89%、L256 +3.24~3.68%，P/S/SCORE持平 → CPU池并行度是长输出decode的真实贡献项，但尚不足以跨过5% E2E门槛，也未在短请求上体现。
- 资源：两轮GPU peak均7750.11MiB；C3 host RSS peak +0.064%、min commit available -2.6%，在15%限内。t10出现L1 token hash差异且B1/B2彼此不同（t8的A1==A2），宿主配置本身引入非确定性，不能当免费提速。
- 结论：继续猜测“宿主入口是瓶颈”不成立；t10是长输出decode的候选方向，但需重复样本与确定性检查，当前不默认。
