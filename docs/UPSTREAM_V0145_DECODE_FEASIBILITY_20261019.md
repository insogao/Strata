# 上游 v0.1.42–v0.1.45：单卡 Decode 差异与可行性

日期：2026-10-19。状态：**源码/模型头部研究完成；尚未构建、加载或测量上游引擎，不声明性能提升。**

本分支 `codex/upstream-v0145-decode-review` 从本地 PR #9 的 `4704d8b4dabf73d13e93c9378982c2ce1867e75e` 建立，仅交付研究。没有合并 upstream/main，没有改功能源码、默认配置、模型、生产冻结件或 PR #9 分支。OpenCode Go 可用于后续有界适配；本轮未派发、未发模型请求。

## 1. 裁决与建议

1. **不能直接在现有服务上设置几个上游变量就期待加速。** 当前 exe 运行 `backends/qwen35_llama` 的 llama 图＋Strata 专家桥；根目录的原生引擎是另一条路径。根目录 `src/`、`include/` 与上游 v0.1.44.1 完全一致，已含 v0.1.42/43 的尾部跳过和自动 PCIe 比例，但当前 Qwen35 服务不经过这些入口。
2. **优先下一步：同生产 GGUF 的隔离 v0.1.45 加载/单请求冒烟。** 头部支持检查没有发现必须换量化或换模型的障碍；这是“可以进入运行验证”，不是“已经能够运行”。上游发布时没有验证本机的 Windows＋4060＋UD-IQ3_S 组合。
3. **不能拿上游最高约 40% 当本机预期。** 本文件没有 Q8_0 权重，部分新 Q8 专用 Decode 内核不会命中；8GB 必须用混合专家，区别于上游全 VRAM 的主要展示场景。
4. **SCORE 服务替换暂为 NO-GO。** 上游 Qwen3.6 服务不支持本地 `SCORE/CANDLOGPROBS/CAPS/STATE/SAVE/LOAD` 合同。其离线全词表 logits 导出可以做诊断，不能冒充完整 SCORE 耗时或 Session 兼容性。
5. **不优先移植 TAIL_SKIP。** 它改变数学输出，而且现有 T1 桥没有此机制；原样移植 top-10 规则到本地 top-8 需要另立质量实验。先确认保持完整专家计算的路径是否已有足够收益。

唯一建议的下一动作是隔离构建/同文件加载资格验证；通过后才开展配对性能与评分适配。没有证据支持现在重写 CUDA 或切生产。

## 2. 事实身份与研究边界

| 对象 | 固定身份 / 复核结果 |
|---|---|
| 本地源码 | insogao/Strata，`4704d8b4dabf73d13e93c9378982c2ce1867e75e`；研究前工作树干净 |
| v0.1.42 | `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`；发布 2026-10-10 16:10:01 UTC |
| v0.1.43 | `7695859aa13afb248bdb4d36f9b2e95707db079d`；发布 2026-10-11 14:40:33 UTC |
| v0.1.44 | `ed4773eb7c8f1f1531ba2bc70e0e1987de2661af`；发布 2026-10-13 19:26:51 UTC |
| v0.1.44.1 | `296b59d2f73c8f04a805df29c2fa97655adace78`；发布 2026-10-14 00:13:09 UTC；也是本地与 v0.1.45 的 merge-base |
| v0.1.45 | `477ac13cbe8e582c20484871ce459d20d82df09d`；发布 2026-10-17 06:53:17 UTC |
| 当前实际 exe | `D:/work/StrataUnified/build-session-correctness/strata.exe`；139,308,544 B；本轮复算 SHA256 `ea06d39411e160f5784f9c4ef160f5e9bcc1746fb92fd9b798677aabb0dd0619` |
| 实际 CMake 源目录 | `D:/work/StrataUnified/strata/backends/qwen35_llama`；CUDA arch 89；QWEN35/native experts/MMQ K-quants/Q6K 打开 |
| 生产模型 | `D:/work/StrataUnified/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`；13,676,723,168 B；已登记完整 SHA256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`；本轮仅重读头部，不重算全文件 SHA |
| 研究配置 | RTX4060 Laptop 8188 MiB；单请求；3200 专家槽；t8/u4096/c4096；MTP OFF；不安装/替换默认配置 |
| 执行成本 | builds=0，native=0，GPU trace=0，模型下载=0，生产功能修改=0 |

远端 tag 只取到 `refs/research/upstream/v*`；没有移动本地普通 tag、分支或上游主线。源码逐字节副本、发布 API 原文、SHA 清单及头部盘点保存在：

`D:/work/Strata魔改/results/upstream-v0145-review-20261019/`

关键原件：`source-manifest.json`、`local-path-audit.json`、`production-header-audit.json`、`baseline600-recomputed.json`、`v*.release.json`、`sources/v0.1.45/`。头部原件包含全部 733 项类型/shape/offset；不含模型 payload 或600条私有输入。脱敏审计摘要同步入仓 `docs/evidence/upstream_v0145_20261019/audit.json`。

## 3. 三条引擎路径，避免重复移植

| 路径 | 本地包含情况 | 当前服务是否使用 |
|---|---|---|
| 根目录 Flash-Next 原生引擎 `src/program/generate.cpp`、`src/core/expert_source.cpp` | v0.1.44.1 原样；已有 tail skip、PcieAuto、resident RAM、90% 命中覆盖默认守卫 | **否**。本地 `--qwen35-model` 提前分流，不进入该原生 serve loop |
| 根目录 Qwen3.6 原生引擎 `src/qwen35/q36_engine.cu` | 已有 v0.1.44.1 版本；不是完全没有原生 Qwen 支持。尚无 v0.1.45 的 `q36_kernels.cuh` 加速、并行/视觉扩展 | **否**。当前 build 的 CMake 源是后端子目录 |
| `backends/qwen35_llama` | `UPSTREAM_QWEN35.json` 标明起源 v0.1.41＋固定 llama.cpp b11476；其后有本地 Prefill、Q6、实验和协议补强，不应直接称作“纯 v0.1.41” | **是**：generate.cpp:2028–2061 → qwen35_serve.cpp → llama_decode → qwen35_bridge.cpp → 专家服务 |

当前 T1 专家路径 `backends/qwen35_llama/src/core/expert_source.cpp:3915`：GPU cache hits 发到 GPU；misses 经 `expert_native_misses` 在 CPU 计算再上传部分结果。**不是**上游原生 verifier 的“misses 再分 PCIe GPU/CPU”路径。桥接初始化 `qwen35_bridge.cpp:1837–1885` 没有建立 resident complement 或设置原生 verifier 的 PCIe 自校准。

`pcie_num` 的后端通用多行服务仍有代码，不代表当前 MTP OFF T1 分支会使用。新增 env 必须证明确实命中调用点，不能只看启动接受了变量。

## 4. 版本变化与四项重点

下表源码位置全部以固定 v0.1.45 SHA 为准；本地路径特别注明。

| 方案 | 适用条件、当前包含情况 | 开销 / 输出影响 | 上游证据与本机裁决 |
|---|---|---|---|
| `STRATA_ROUTE_TAIL_SKIP`，v0.1.42 起 | 原生 Flash-Next verifier，非全 VRAM、单请求等条件；rank 默认7；v0.1.43 加入高覆盖停用守卫。根目录已有，当前后端 T1 无跳过实现，新 q36 引擎也不调用它 | 路由分组检查少量成本；被跳过 miss 专家贡献为0，其他权重不重归一化。**有意改变 logits/输出**。不是无损优化；不能由 Decode 收益推导单次完整 Prefill SCORE 收益 | `src/core/expert_source.cpp:3767–3781,3850`；generate.cpp:9375；[65268d99](https://github.com/Niko1221/Strata/commit/65268d99)、[#1668](https://github.com/Niko1221/Strata/pull/1668)。3060 IQ3_XXS 五交错轮总体 +15.6%，不是本机/本模型；优先级低，需独立 top-8 质量设计 |
| 自动 PCIe/CPU share，v0.1.42/43 | 原生 verifier；无显式 `--pcie-frac`，可做 GPU miss；CUDA 单卡，排除 batch/peer/全驻留等。根目录 `PcieAuto` 已有；Windows 的 canonical Q2_0 特例才需 `STRATA_PCIE_AUTO_CANON=1`，不能把它当当前 UD-IQ3_S 桥的总开关 | 复用 CPU pool 计时；48/96/再96窗口最多调3次；比例0.10–0.60、步长0.05；需要校准时段，冷/热须分开。CPU/GPU 舍入不同，输出可能变化；公式系数1.85是上游拟合 | generate.cpp 中 PcieAuto / `pcie_frac_for_gbps`；[aa7b0cb1](https://github.com/Niko1221/Strata/commit/aa7b0cb1)。3060 IQ3_XXS +30%，A4000比例不变无收益；当前桥没有相应 GPU-miss 通路，**不是低成本开关试验** |
| `STRATA_PCIE_BALANCE=1`，v0.1.45 新 opt-in | 与上一项不同：原生 verifier 按层拟合 GPU/CPU 成本。根目录 v0.1.44.1 和当前桥均未含这一新功能；q36 新引擎没有接入此调度器 | GPU clock stamps、CPU采样、遗忘最小二乘、每64个有miss层探索一次；时序与 CPU/GPU 分配会改变数值。不得不经测量预设更优 | `src/core/verify.cpp`、`include/strata/core/expert_source.hpp`、generate.cpp:3349；[04be241a](https://github.com/Niko1221/Strata/commit/04be241a5a7ec016716cb50aab99be82a801e4fc)。DETAILS 单 A4000 +5.6%，单3060 −1.6%/P100 −1.3%；release 对 A4000 张数描述不一致，本报告不据此放大收益。**暂缓** |
| RAM 专家常驻副本及 lend-region 修复 | 旧原生引擎已有 `--resident-experts`/预算机制；v0.1.45 修复预算尚有余量却没保存 Prefill 借走槽的副本。根/后端有基础容器，均无此次 `budget_lent` 修复；当前桥未启用完整 complement 生命周期。新 q36 则用另一套全专家 host arena | 内存、commit、可选 page-lock、初始化复制；保留 lend 副本额外消耗 RAM，不能把 mmap 文件大小与所有驻留页重复简单相加。存储优化本身不删除专家；分配和时间变化可能间接影响动态 cache/浮点路径 | expert_source.cpp:2700–2782,3255；[f2e142f4](https://github.com/Niko1221/Strata/commit/f2e142f4440f314a56d7afa5ca1c5a7a3fdd1701)、[#1896](https://github.com/Niko1221/Strata/pull/1896)。提交自带6900XT单次数据仅支持读盘/Prefill改进，release另列3060五对 +7% Decode/+2.5% Prefill；不能混成同一实测。当前桥需存储生命周期适配，先查本机冷读/硬缺页，不盲目移植 |
| Qwen3.6 新 Decode 内核，v0.1.45 | **本模型家族的优先候选**。根目录仅有旧 q36 引擎，当前后端完全没有 q36 独立内核；适用于完整新引擎或选择性接入其算子 | 共享 Q8 gate/up/激活量化融合、Q8一warp一行、norm＋量化融合、路由＋分组、MoE输出＋下一层量化；省 launch/中间读写，部分需要额外打包/verify scratch。上游目标是相对其旧 q36 同 bits，**不是对当前 llama 桥自动同 bits** | q36_engine.cu:1575–1832；q36_kernels.cuh；[c9e7f6dd](https://github.com/Niko1221/Strata/commit/c9e7f6dd5afb011a1631b2ad710558468a967e9d)。`STRATA_Q36_OLD_KERNELS=1` 对照，`STRATA_Q36_VERIFY=1 --no-graph` 诊断；上游18,360次真实激活比对0差异，只覆盖其已测组合。允许进入隔离验证，不能先承诺40% |

版本补充：[v0.1.42](https://github.com/Niko1221/Strata/releases/tag/v0.1.42) 首次 CUDA tail/share；[v0.1.43](https://github.com/Niko1221/Strata/releases/tag/v0.1.43) 主要新增 AMD 默认、Windows opt-in/缓存守卫；[v0.1.44](https://github.com/Niko1221/Strata/releases/tag/v0.1.44) 引入原生 Qwen3.6；[v0.1.44.1](https://github.com/Niko1221/Strata/releases/tag/v0.1.44.1) 主要为兼容热修；[v0.1.45](https://github.com/Niko1221/Strata/releases/tag/v0.1.45) 新 q36 内核/Prefill 和选择性性能 PR。

## 5. 新 Decode Kernel 不应混成一组

### Qwen3.6 专用：优先调查

`q36_engine.cu:1599` 的 `q8_small` 要求 `w.type==8`；`:1696` 的 `shared_gu` 要求 gate/up 都是 Q8_0。**生产文件 Q8_0 张量数是0**，因此这两项不会命中，Q8 packed weight 副本亦不产生收益。norm/量化、router/group、MoE输出融合仍可能命中；不能把整组在 Q4/Q6 XL 文件上的收益直接迁移。

`STRATA_Q36_OLD_KERNELS` 是按“变量存在”检查，不是严格值检查：恢复新内核应删除该变量，不能设 `=0`。VERIFY 同理，诊断结束应删除。新 q36 的通用激活 q8_1 不意味着权重被换量化，GGUF payload 保持不变。

v0.1.45 还扩大自动 prompt chunk，并优化 gated delta rule 读下一 token/同步。全驻留候选有4096/2048/1024，混合模式按剩余显存还可退到512；不能说8GB自动获得4096。较大 chunk 会改变长 prompt 浮点结果；本地 `--qwen35-ubatch 4096` 与 q36 `--prefill/--chunk` 不是同一实现/显存含义。需分别锁参数与比较自动配置，不能把 chunk 收益记成 Decode 内核收益。依据：[93f0f48f](https://github.com/Niko1221/Strata/commit/93f0f48f)、[6f4697f6](https://github.com/Niko1221/Strata/commit/6f4697f6)、[55bae973](https://github.com/Niko1221/Strata/commit/55bae973)。

### Flash-Next 的 CUDA opt-in：大部分不适配当前图

| 开关 | 本地状态与适配判断 | 输出、开销与证据 |
|---|---|---|
| `STRATA_ATTN_DAILY`、`STRATA_QSA_ROWS` | 根/后端无 v0.1.45 新开关；用于 QSA，当前 Qwen35 的全注意力/GDN llama 图不能通过 env 改用 QSA | 上游各自 bit-exact gate；注意力分块/merge或寄存器查询；30K数据不能外推4K；DETAILS:2228–2229；[#2075](https://github.com/Niko1221/Strata/pull/2075)、[#2059](https://github.com/Niko1221/Strata/pull/2059) |
| `STRATA_EXPDEC` | 同上尚无；属于共享 grouped expert 层，**比 QSA 更有选择性移植可能**，但需核 IQ2_S gate、T1 和实际 launch 是否覆盖，不能直接开 | 共享内存保存 q8_1，增加 shared-memory/寄存器压力；上游目标 bit-exact；DETAILS:2230；[#2058](https://github.com/Niko1221/Strata/pull/2058)。先排在原生引擎冒烟后 |
| `STRATA_SIDE_MIXER` | Flash-Next hyper-connection；Qwen35 没有该算子 | 图分支/临时缓冲；不可用本机当前图；[#1889](https://github.com/Niko1221/Strata/pull/1889) |
| `STRATA_B6_MMVQ_ROWS` | 多列MMVQ复用，单请求 MTP OFF T1 没有目标多列复用窗口 | 上游目标 bit-exact；增寄存器用量；无本轮价值；[#1418](https://github.com/Niko1221/Strata/pull/1418) |
| `STRATA_QSA_SPLIT_ATTN` | 额外项，并非所有“新kernel”都无损；sm80+ INT8 KV QSA，不是当前 Qwen35 图 | 舍入改变，需局部缓冲/合并；[#1888](https://github.com/Niko1221/Strata/pull/1888)。不列本轮候选 |
| `STRATA_SM75_PERF` | 4060为sm89，排除 | Turing专用；不能按2080Ti速度换算 |

依据：[固定版 DETAILS](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/DETAILS.md#cuda-opt-in-switches-measured-for-0145-rtx-3060-and-rtx-a4000-sm_86-all-off-unless-you-set-them)。表中上游速度主要来自 IQ3_XXS/30K，不作本机收益估计。

### CPU 补充：避免“有CPU优化就全移植”

当前后端 `iq_avx2_rows.inl`、`kq_avx2.cpp` 与 v0.1.44.1 对应文件一致，尚无 v0.1.45 新增部分。但：

- `79376a27` 的寄存器累加改的是 **Q5_1/Q8_0 多 token expert dots**，本文件 routed experts 是 IQ2_S/IQ3_S/IQ4_XS，不是直接命中的瓶颈；不是“给 IQ3 CPU 通路提速30%”。
- `STRATA_CPU_IQ_ALL_EXACT1` 为新 IQ singleton opt-in，但受 ggml AVX2且无AVX512等守卫限制，并放在 contiguous-blob `native_gu_rows`。当前 T1 生产路径使用 slice jobs / `native_gu_rows_slices`，不能换文件就认定命中；7945HX也需核实际ggml ISA。
- 原生 Flash-Next router BF16 dot 的寄存器优化，不等于当前设备 router 的优化。

## 6. 完全同 GGUF 的加载可行性

本轮只读原文件约11MB头部，733张量：F32=361、Q6_K=252、IQ2_S=80、IQ3_S=37、IQ4_XS=3。40主层、256专家/top8、n_embd2048、expert_ff512、head_dim256、GDN state128/conv4；没有 nextn_predict_layers/blk.40，符合 MTP OFF 研究，不要求换成15.3GB MTP副本。

| 检查 | 静态结果 |
|---|---|
| 架构与维度 | `qwen35moe`；符合 q36 loader 的关键尺寸；`family_layout.hpp:31–86` 允许 n_mtp_layers 默认0 |
| Norm/router/GDN标量加载 | 所需361个张量均 F32，符合 `upload_f32`；未发现类型拒绝项 |
| Dense/head | Q6_K 在 `native_mmvq_supported` 列表中 |
| Token embedding | 也是Q6_K。**隔离构建必须显式 `STRATA_Q6K_EXPERTS=ON`**：虽然 experts本身无Q6，embedding的 `embed_type_supported → is_iq` 对type14也受该构建门影响 |
| Expert payload | 37层GU IQ2_S/D IQ3_S，3层GU IQ2_S/D IQ4_XS；GU/up类型一致；native expert支持这些type及2048×512几何 |
| 已测上游组合 | 发布主要覆盖MTP GGUF的UD-Q4_K_XL/UD-Q6_K_XL/Q8_0，未证明本生产IQ3文件＋Windows硬件 |
| 最终结论 | **静态 GO 进入隔离加载；运行兼容未判定**。还需构建、真实加载、所有层、EOS/NaN/下一请求检查；不能称“直接可替换” |

固定源：[q36 loader](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/qwen35/q36_engine.cu#L959)、[类型/embedding支持](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/kernels/cuda/iq_kernels.cu#L3443)、[Qwen3.6说明](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/QWEN36.md)。

## 7. 3200槽、RAM和Graph不能偷换

生产 pack 的全部专家payload为11,574,181,888 B（约10.78GiB）；全驻留不可能在8GB卡实现。上游 q36 hybrid 会：

- 在host arena保存全部专家副本，尝试 pinned，失败后 pageable；初始化时间和实际RAM/commit压力必须计量。
- 分层分配统一数量 `--expert-slots N`，**不是本地全局总槽数**。40层×80=3200；误传3200会每层clamp256，绝不是当前3200槽配置。
- `--cpu-threads` 在该版本直接传给 `ExpertPool(n_workers)`，host默认也参与计算；不能机械地两边都传8就声称相同并发。本地 t8 已换算7 workers＋host，隔离端也须按实际8个计算lane核对（初始参数为7 workers）。
- 80/层的裸slot字节约3,616,931,840 B，另有alignment、staging、dense、KV、workspace等，不能据此宣布显存可容；不同分布/动态替换策略必须登记。
- 主层hybrid时 `E.use_graph=false`（q36_engine.cu:2897附近）；即使新内核更快，丢失整图覆盖或更多host参与仍可能抵消。
- 使用双层expert staging；当前后端的workspace借槽/恢复生命周期不同。不能把相同槽总数说成相同专家驻留策略。

所以需要两类结论分开：①上游新引擎整套方案相对本地方案；②同一个上游exe内 OLD_KERNELS on/off 的增量收益。前者包括cache/图/调度/算子差异，不能归因于某一kernel；后者也不覆盖旧版本全部Prefill变化。

## 8. Score / Decisions / Session 差异

上游 q36 `Serve::run`（:3675）只处理GEN/BGEN、STOP/BSTOP等原协议；其他命令返回期望GEN的ERR。没有本地候选概率、评分结束原因、状态容器及revision/owner合同。当前本地 `qwen35_serve.cpp`、`score_api.py`、`strata_unified.py` 的补强不得整套覆盖。

上游离线 `--tokens ... --n 1 --dump-full-logits ...` 可导出prompt末全词表向量，候选15/16/17/18取排序，再用全词表LogSumExp计算logprob。但该路径输出一个greedy pick且不提供本地完整Score服务：

- 可作为相同prompt的数值/排名诊断；不能把进程启动＋dump、或仅prefill_ms，标为常驻完整SCORE E2E。
- 正式SCORE比较前，需隔离的只读score接口，返回完整DONE/count/finish/状态语义，拒绝不支持的Session能力。不得伪造CAPS，或用Top-K闭集归一化代替全词表概率。
- 已有本地keep/continue、SAVE/LOAD和取消修复保持原样；新后端未验证时只能隔离研究，不能路由生产会话过去。

## 9. 可复用的600条基线

参考标签是用户批准的 **synthetic_design_label**，不是人工业务金标。固定600条、每档150、seed2026101609；不得按新预测重抽样。

| 方法 | Exact | ±1 | MAE | 严重错误 | 风险3召回 |
|---|---:|---:|---:|---:|---:|
| A0 JSON | 51.83% | 94.33% | 0.5417 | 34/600，5.67% | 123/150，82.00% |
| A1数字 / B1 Logits | 49.83% | 92.83% | 0.5833 | 43/600，7.17% | 124/150，82.67% |

A1/B1同prompt、600/600预测相同。B1预测分布167/2/181/250，风险1召回仅1.33%；这本来就是开放短板，不允许因上游引擎改变排序后把历史偏置当新收益。

本轮重新读取20批共1800条原始结果，独立重算三张混淆矩阵和上述计数，与既有报告一致。600条B1均为有限logprob，保存预测全部属于四候选最大值集合，无保存精度下的top1并列；top2 margin≤0.001有1条、≤0.01有3条。只按已保存精度汇报，不虚构更高精度。原件 `baseline600-recomputed.json`。

匹配580条长驻请求的A0/A1/B1完整E2E均值1574.50/1180.81/1011.37ms，B1 Prefill962.21ms。**不是4K或128token Decode结果**：这600条实际prompt360–1667tokens，必须与固定4K/L形状分列，也不是本轮同日配对。

原始候选logprob/label_mass/top2 margin和prompt SHA可从 `results/unified-p4-balanced600-20261016/batch-*/private-results.jsonl` 逐case回查。公开索引在 `backends/qwen35_llama/tests/evidence/risk_balanced600_20261016/`，报告见 [P4完整对照](P4_BALANCED600_COMPARISON_20261016.md)。

新引擎必须保存四候选完整排序、top1翻转、near-tie margin、有限性和全词表归一化，不只比较最终risk。数值比较限定相同状态类/同输入；不把两次独立长文本逐token恒等设产品门。结构、游标、隔离、取消恢复仍是硬门。

## 10. 下一阶段有界执行方案（尚未启动）

以下是研究建议；实际派发仍需先填manifest、started_at、源码/build/exe所有权和剩余预算，不能把这段当作已有运行记录。

### A：同文件资格验证

- 在独立managed worktree固定 `477ac13...`，独立build/exe，绝不改当前验收对象；只构建CUDA sm89/q36依赖，Native experts与Q6K打开，不引入多卡/batch/vision/MTP。初始chunk显式512用于加载资格，再分别评估同chunk与自动chunk，不能以失败后暗改配置重试。
- 初批最多1次build（原子上限1200s）；20min总续步窗到期不再启native。编译失败保留日志，不自动换toolkit/模型/供应商或重试；据具体错误单独登记有限修复。
- 下一有界批最多1个native/240s，锁等待≤60s，真实同GGUF、c4096、t8、80槽/层；先做短输入/实际EOS及正常清理，再少量4K和L128检查；诊断插桩另批。
- 预先核Windows DLL PATH、GPU锁/JobObject、合法argv、GPU/hostRAM预算；不得依据“约3.37GiB槽”跳过启动前资源准入。OOM、非法数值、清理失败、context/EOS错误即停止该候选，不自动缩槽洗失败。
- 只做加载成功仍不通过SCORE/Session，不据单次bench结果开PR宣称提速。

### B：同日配对单请求性能

- 固定同GGUF、同token ids、MTP0、同线程、合法context、独立同机时段，首次/后续长驻分开。P3896/out1、S3896/out16、L3817/out128，沿用已有fixture SHA；128不是强制忽略EOS，实际早停须照录并对可比完成长度单列。
- 分开实验：本地现状 vs 上游方案；以及上游同exe新/旧kernels（old变量删除/存在）。一次改变少量变量；另列实际chunk、缓存分布、专家字节、Graph覆盖，不能把不同驻留策略隐藏为“同3200”。
- 起始ABBA四个native，每个上限180s，最多12请求；只在漂移或新证据需要时追加注册。禁止GPU计时并行CPU编译/全文件hash/其他模型负载。
- 记录加载、warmup、Prefill、HTTP/协议TTFT、完整请求E2E、实际n/EOS/finish、逐token时间、TPOT与(n−1)/(last−first)TPS、真实整卡峰值、hostRSS/available/commit。原生自报n/decode_ms另列，不冒充逐tokenTPS；n<2不算TPS。
- SCORE接口适配之前，该栏标 **NOT_MEASURED_PROTOCOL_GAP**。不能用GEN1替代SCORE完整E2E。保持现有桥为Score服务基线。

### C：评分适配后才跑质量与业务速度

- 先做冻结小冒烟与同状态类全词表/四候选检查；如果早期出现非法数值或新严重错误，不盲跑600。
- 冒烟过后沿用600全部样本，新的同日baseline/candidate交错，A1/B1严格同prompt；保留A0 JSON方法，必要时先跑分层小批再注册剩余批。历史600可做质量对照，不冒充同日性能配对。
- 出具Exact/±1/MAE/严重错误/risk3召回和4×4矩阵；专列risk3→0/1、相差≥2、candidate排序翻转及边界样本。不得把机器参考一致率称人工准确率。
- 首轮采用保守筛选：新增状态污染/NaN/OOM/CUDA/协议错误立即NO-GO；风险3漏报/严重错误增加不自动接受；其他指标结合配对误差和逐case证据说明，不隐瞒互有得失。没有明确可接受质量依据不生产合并。
- 只有1%–3%且落在漂移内，**性能NO-GO，保留原基线**。明显收益也须Prefill/SCORE/质量/状态门齐备后才提交独立实验PR；研究GO不等于产品发布通过。

## 11. 当前结论及未知

本轮完成了发布/提交固定、双源码路径核对、实际exe身份复核和同模型头部盘点；**没有新的GPU性能或600条质量测量**。当前锁无有效任务记录，未运行strata/编译进程；OpenCode桌面进程存在，不代表有施工session。

仍未知：v0.1.45对本IQ3混合格式的Windows实际加载/全层数值、8GB真实峰值与host压力、新引擎相对当前桥的收益、完整SCORE适配成本和600条排序变化。本报告没有把这些未知写成通过，也没有沿用上游其他量化/显卡数字宣称本机提速。
