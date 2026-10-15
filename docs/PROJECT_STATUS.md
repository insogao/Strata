# PROJECT_STATUS — Strata Unified（持续自主开发 + 异步外部审查）

> 维护者：本仓主管线（Codex）。**每次完成里程碑即更新，不等待反馈。**
> 各任务的逐条证据在旧研究仓 `D:/work/Strata魔改/results/opencode-execution/` 与 `docs/`；本文件只做索引与状态。

更新时间：2026-10-15（**阶段 2 收尾＋PR#2 评审补强 P2R-1…P2R-4a**：见 §4.0b；阶段 1 交付汇总见 `docs/PHASE1_DELIVERY.md`）

## 0. 一句话状态

**第一阶段（核心迁移）已完成并通过自测**：①本仓独立构建 exit 0；②**经官方服务入口**的普通聊天/连续请求/EOS/**取消**/状态隔离全部达成（含引擎级 `finish=cancel` 原文）；③构建与运行**不依赖旧研究树**（中立路径实跑）；④与生产冻结基线的 ABAB 配对回归 **decode PASS、prefill INCONCLUSIVE（中位 +7.15% 在带内，配对噪声；无回归证据）**。阶段 2（统一 Backend/Logits Decisions）可自主启动（**不合并 main**）；阶段 3 未开始。

## 1. 三大阶段进度

| 阶段 | 状态 | 关键门槛（自测） | 证据 |
|---|---|---|---|
| 阶段 1 核心迁移 | **完成（自测达成）** | ①独立构建 exit 0；②官方服务入口的聊天/连续/EOS/取消/隔离；③不依赖旧研究树；④与本产基线配对回归不劣化 | `docs/PHASE1_DELIVERY.md`（四项逐条证据） |
| 阶段 2 统一 Backend + Logits Decisions | 未开始（可启动） | 双 Backend 能力表可查询；`/v1/chat/completions` 回归；`/v1/score`、`/v1/decisions` 文档化子集；状态语义与质量回归 | 接口草案见 350/351 报告（E1–E9、四类缓存边界） |
| 阶段 3 Hardware/Workload Autotuner | 未开始 | 安全预算内搜索；4060 上识别 2700/3200 实测差异；第二张卡验证（若不可得则如实标注待办） | — |

## 2. 当前分支与提交

- 分支：`integration/qwen35-llama`（`origin` = `insogao/Strata`，`upstream` = `Niko1221/Strata`）
- PR：#1（自动随分支更新；**标题/正文编辑受 PAT 权限限制，见 §5 阻断项**）
- 提交链（自 base `296b59d2` 起）：
  - `27c987ca` 导入 `backends/qwen35_llama`（源自只读实验树 `phase2-mtp232`：HEAD `9bb9b6a`＋17 dirty＋2 untracked；vendor/llama.cpp live 态）
  - `e0fd2dea` 冒烟＋保真报告（task 355）
  - `c653bef1` ABAB 配对回归报告（task 358）
  - `b5e43917` 生产线同口径对照报告（task 359）
  - `0c118899` 第一阶段文档集（install/run/selection/versions/test-evidence）
  - `7386180d` PROJECT_STATUS（本文件首次建立）
  - `6d9543a8` **精简发布源码树**（task 361）：删 2,019 文件/29,888,754 B（旧项目历史材料），树 6,146→**4,128** 文件（vendor 3,697 占大头）；`docs/backends/qwen35_llama_excluded_material.md`（2,161 行逐文件 SHA 索引，可 `git show 27c987ca:<path>` 取回）；CMake 仅加 SYCL 守卫；**clean 全量重建 749 s / 383 目标 / exit 0** → 新 exe `8c2df0a7040224ccd3c7e960d84efcf9d741a6eb57bb93bf54bd07d71772cc07`（139,195,904 B）
  - （进行中）M2：官方 Server 行协议（task 362）
  - `131953d7` **官方 Server 行协议（serve 模式）**（task 362）：`qwen35_serve.cpp` 554 行；exe `2f47b88a…`
  - `17e8ce8d` 直连协议冒烟（task 363d）：连续 3 请求、隔离成立（req1==req3 同 SHA `12699c3c…`）
  - `1e7a0bda` 官方 Server E2E（task 364c）：聊天/连续 8+/隔离；取消有功能性证据（`cancel=True`、engine 7<128）
  - `4e32aae2` 修正 E2E 文档中的生产模型 SHA（主管全量复算 `66a3ca88…`）
  - `cca5f672` **中立放置**（task 365）：8/8 硬链接（同 inode）、pack=native_experts.txt、配置模板与三文档
  - `3503eace` serve 路径 **一行 stderr 诊断** ＋ 中立路径实跑（task 365b）：最终前 exe `f9f4485c…`（504 s 构建）
  - `fb1fc1bb`/`58d18259` **引擎级取消证据**（task 366）：`serve: req 3 finish=cancel produced=8`（官方 server 实时 STOP）
  - `d103d5ff` **最终配对回归**（task 367b）：6/6 native、ABAB、中立路径；decode PASS（+5.74%）、prefill INCONCLUSIVE（+7.15% 在带内、2/3 微小负配对）、两臂输出序列逐字节相同（`795a93ac…`）
  - `a96065ff` **PHASE1_DELIVERY.md**（阶段交付汇总）
- 开发产物（未入库，`.gitignore`）：`backends/qwen35_llama/build/strata.exe`（最终 `f9f4485c…`，139,215,872 B）
- PR #1 规模：变更文件 6,158 → **4,140**（其中 vendor/llama.cpp 3,697 为必需依赖）

## 3. 已完成的里程碑 M1（阶段 1 内部里程碑）

**M1＝源码导入＋独立构建＋保真/配对回归＋文档集**，自测通过。

| 项 | 结果 | 证据 |
|---|---|---|
| 导入 | 6,146 文件 / 217,233,773 B，19 个改动文件对归档＋live 双验 SHA 全等 | `strata-unified-import-354/` |
| 构建 | clean 全量 666 s、383 目标、exit 0 → `build/strata.exe` 139,208,192 B，sha256 `6062a714…` | 同上 |
| 导入保真 | vs 旧实验 exe `03bd7068`：4 请求可见 token id 序列同 SHA `f876b40d…`；finish=stop/eos=1；hits 3654 / misses 6586 / VRAM 7577.3 MiB 全等 | `strata-unified-smoke-355/`、`docs/backends/qwen35_llama_smoke.md` |
| ABAB 配对回归 | warm TPS A 33.062 vs B 34.470，Δ +4.26%（漂移带 4.248 内）→ PASS | `strata-unified-regression-358/`、`docs/backends/qwen35_llama_regression.md` |
| 生产线同口径 | vs 生产冻结 exe `0e00af8a`：6/6 native exit 0；6/6 输出 16-token 序列逐字节相同（`795a93ac…`）；prefill Δ +1.19%、decode Δ −1.87% → PASS | `strata-unified-prod-regression-359/`、`docs/backends/qwen35_llama_prod_regression.md` |

## 4. 进行中 / 下一步

### 4.0 M2/M3 进度（2026-10-14 21:0x）

- **M2 官方 Server 集成**：`--serve` 行协议已实现（task 362，commit `131953d7`；exe `2f47b88a…`）；直连协议冒烟通过（task 363d，连续 3 请求＋**状态隔离** req1==req3 序列同 SHA `12699c3c…`；生产模型短答 `stop`@2 属正确行为）；文档 `docs/backends/qwen35_llama_serve_protocol.md`、`docs/backends/qwen35_llama_serve_smoke.md`。
- **M3 官方服务入口 E2E**：**已达成**（task 364c，commit `1e7a0bda`，主管修正模型 SHA → `4e32aae2`）：E1 聊天 HTTP 200＋文本＋EOS；E2 常驻连续 8+ 请求全 200；E4 隔离（同 prompt 热重复 4/4 逐字节一致、不同 prompt 不同）；**E3 取消**＝流式中止后 server 记 `disconnect,cancel=True`、引擎**只产出 7 个 token（<128）**、下一请求正常；**引擎字面 `DONE … cancel` 未落盘（官方 server 内部 drain）⇒ 登记为未取得**，将由 365b 的诊断行补齐。租约形态：`gpu-lock hold`（`DENY_BASENAMES` 禁解释器作 declared binary）＋补偿证据（跑前 exe SHA 复算、引擎 READY、收尾锁释放无残留）。
- **M3 中立放置**：task 365 进行中（模型/pack/profiles → `D:/work/StrataUnified/models*` 硬链接；配置模板重指）⇒ 随后 **365b**（serve 路径加一行 stderr 诊断并重建，从中立路径实跑并补引擎级取消证据）→ **366** 用最终 exe 对生产冻结 `0e00af8a` 做同口径配对回归 → **367** 阶段交付汇总＋PR #1 正文。

### 4.0b 阶段 2 进度（2026-10-15 00:3x）

| 项 | 状态 | 证据 |
|---|---|---|
| P2-1 能力表 ＋ 只读 `SCORE` | 完成 | commit `cf134310`（exe `66a48ad1…`；能力字段与实现一致，无虚报） |
| P2-2 CAPS/SCORE 的 GPU 验证 | 完成 | commit `e01e4feb`（零 `T` 行＝不生成 token；`pos=3868`；CLI 0.141 s 不加载模型） |
| P2-3 `/v1/score`＋`/v1/capabilities`＋最小 `/v1/decisions` 薄层服务 | 完成 | commit `7f5392ce`（4 新文件/1424 行；**19 单测全绿**；官方/引擎零改动；SGLang 子集边界文档） |
| P2-4 服务的 GPU 端到端 | 完成（带数值告警） | commit `228796f6`（与引擎 stdout **4dp 全等**；**冷↔热最大 |Δ|=0.0111**，精度告警入档；根因仅假设） |
| P2-5 `SAVE`/`LOAD`/`STATE` 单会话状态（含 GDN） | 完成 | commit `f6e7d048`（exe `6dee0c66…`；先导临时对象不半恢复；path gate） |
| P2-6 `SAVE`/`LOAD` GPU parity | 完成 | commit `596e936f`（同进程/跨进程逐 id 全等；失败不半恢复；可重入字节相等；`state_export` 保持 true） |
| P2-7 双 Backend 适配器＋Batching 契约 | 完成 | commit `17c9ea1f`（官方后端能力表如实 false；不支持端点 501；契约 v1；既有 19＋新 13 单测全绿） |
| 阶段 2 交付汇总 | 完成 | `docs/PHASE2_DELIVERY.md` |
| P2R-1 `SCORE cand=` 显式候选评分＋`/v1/decisions` 三概率语义（PR#2 评审 P0-1/P0-2 的 0-GPU 部分；分支 `phase2r/review-gaps`） | 完成（0 GPU 交付；**GPU 对拍已由下行 P2R-2 完成并通过**） | 同一分支首个提交 `P2R-1: ...`（作者 insogao）；新 exe `d0321a98…`（139,257,344 B，旧 `6dee0c66…`）；`test_score_api.py` **34 全绿**（新增 ≥12 用例）＋`test_backend_adapters.py` **13 全绿**（未改）；`CANDLOGPROBS`＝同一 prefill/行/lse，无第二次 forward；无 `cand=` 请求 stdout 逐字节兼容（唯一改动＝兜底 `ERR expected:` 加 `[cand=…]` 提示）；decisions 恒返 `label_mass`＋`covers`/`tolerance`/`calibrated:false`/`probability_scopes`，删 `label_mass_omitted`；低 mass 无任何置信/阈值字段 |
| P2R-2 新 exe 的 GPU 对拍与真例验证（A 兼容/B 候选/C 漂移/D 服务真例；分支 `phase2r/review-gaps`） | 完成（**A/B1/B2/B3/C/D 全项通过**；4 native、严格串行、一租约一 native、无新围栏） | 旧 `6dee0c66…` 直连（N1）＋新 `d0321a98…` 直连（N2/N3）＋新 exe 服务腿（N4，hold＋补偿；服务 `engine_sha256` 匹配）；**A** 同位置 3/3 `LOGPROBS` 逐字节相等（cold P_short／warm1／warm2 P_long；DONE 与 stderr 除 `prompt_ms`/`decode_ms` 外一致）；**B1** `topk=0 cand=<warm2 top-5>` 回转值与同进程 `LOGPROBS` 5/5 逐字节相等；**B2** spread 10 候选全部有限且 ≤0、{top-5}∪{spread} 并集 Σexp=**0.99888066**≤1、共有 id 1 两表同值；**B3** `SCORE` top-1＝`GEN 1` 首 token（P_short 271、P_long 90）；**C** 近差值对 1/5202 三次重复逐字节零漂移（同 host warm#1→warm#2 实测 |Δ| 达 **0.0327**，与 0.0111 并列只报事实，根因仍未定位）；**D** P_open yes/no `label_mass`=**6.9975e-05**（≪1）、D1-D3 三概率字段齐且 `conditional` 4dp 内和=1、`covers`/`tolerance`/`calibrated:false`/三 scope 齐、全响应无任何置信/阈值字段；P_open 低 mass 未改 prompt 事后凑数；证据 `results/opencode-execution/strata-unified-p2r2-candidate-gpu/`；e2e 文档 §6＋design/capabilities 的“待 P2R-2”行据实更新 |
| P2R-2b 近阈值排序敏感性量化（top-2000 类间对照；分支 `phase2r/review-gaps`） | 完成（1 native、一租约、无围栏；翻转真实存在且已量化；P2R-2 的 C caveat 由本节回答） | 同进程 `SCORE P_short topk=2000`×3＋`SCORE P_long topk=2000`×2（新 exe `d0321a98…`，rc=0，12.7 s，pos 22/22/22/3895/3895）；**同类重复 warm1→warm2 两份 `LOGPROBS` 逐字节相等**（max\|Δ\|=0.0000，79 个并列对零翻转，把 P2R-2 的 top-5 零漂移扩到 top-2000）；跨类：cold→warm1 max\|Δ\|=**0.0599**、中位 0.0098、1978/1989 非零、逆序 9837（τ=0.990049）、**相邻翻转 881**（漂移带内 1961 对，≈44.9%；最大翻转 gap **0.0383**）；first→repeat max\|Δ\|=**0.0713**、中位 0.0207、1987/1990 非零、逆序 13930（τ=0.985923）、**翻转 895**（带内 1966 对，≈45.5%；最大翻转 gap **0.0371**）；**最小翻转 gap=0.0000**（4dp 并列可翻）；最接近未翻对 4777/14149 gap 0.0000 margin **+0.0015**；top-k 截断边界 churn **11/11** 与 **10/10**（任务假设“id 集合必然相同”实测不成立，已按交集统计并单独报告）；阈值一句话：本机/本 IQ3_S/本 exe 跨类判序需 gap>≈0.038（保守取 band 0.06/0.07），同类重复任意非零 gap 皆稳；与 1.11e-2、3.27e-2 并列只报事实不编根因；证据 `results/opencode-execution/strata-unified-p2r2b-near-tie/`；e2e 文档 §P2R-2b＋design §8/§10 行据实更新 |
| P2R-3a `SCORE keep=1` 状态保持评分＋`GEN cont=1` 条件续写引擎接口（PR#2 评审 P0-3 的 0-GPU 部分；分支 `phase2r/review-gaps`） | 完成（0 GPU 交付；**GPU 证明待 P2R-3b**、服务暴露待 P2R-3c；SAVE/LOAD 非其前提亦非其验收） | 同一分支提交 `P2R-3a: ...`（作者 insogao）；新 exe `7c441a94…`（139,258,368 B，旧 `d0321a98…`）；`keep`/`cont` 只收字面 `1`（其它值/重复键→ERR）；`keep=1` 无 armed 会话＝reset 后从 0 读、答完 armed（`origin=score-keep`）、有 armed 会话＝cursor 追加不 reset（`DONE.reused`＝追加前 cursor）；`cont=1` 要求 armed（否则 `ERR cont=1 needs an armed session`）且 id 可空（纯续写 `prompt_read=0`）；失败/取消的 keep 请求 reset＋清会话（绝不从半截 append 续）；缺省路径逐字节不变（唯一文本改动＝兜底 `ERR expected:` 加 `[keep=1]`/`[cont=1]`）；CAPS 仅新增 `state_keep_score`/`session_continue`（既有字段一字不改）；`test_score_api.py` 34 全绿＋`test_backend_adapters.py` 13 全绿（未改测试、未碰 `score_api.py`）；证据 `results/opencode-execution/strata-unified-p2r3a-engine-keep/` |
| P2R-3b `SCORE keep=1`→`GEN cont=1` 的 GPU 证明（PR#2 评审 P0-3 的验收本体；分支 `phase2r/review-gaps`） | 完成（1 native、一租约、rc=0、24.2 s、无新 fence、锁释放无残留；**预登记 A1/A2/A3 按字面 FAIL、A4–A8 PASS**；无 SAVE/LOAD） | 静态 stdin（17 行、SHA `f26248c0…`）＋新 exe `7c441a94…`（跑前复算一致）＋ctx4096；引擎行为与 P2R-3a 契约完全一致：4 条续写 `prompt_ms` 0/0/0/25.9 vs 全 prefill 2704.8–3862.0、无 3896-token `PP`（**4K 前缀未被重 prefill**）；A4 首 token 90＝三条 `LOGPROBS` top-1；A5 `ERR cont=1 needs an armed session`＋其后 fresh `reused=0`/`prompt_read=3896`；A6 `origin="score-keep"`/`loaded=false`/`pos=3896`/文件字段 null。**A1 严格值 FAIL**（reused 3896/3911/3896/3912——参考请求失败不清会话、cursor 合法携带）；**A2/A3 FAIL 因参考不可执行**：plain `GEN <P_long>` 是 armed 续写而非全 prefill，`ERR session (...) + prompt (3896 tokens) (+ max_new) exceeds the context (4096)` 故无参考 T 序列（ctx4096 下 2×3896 不可能同会话）；同一运行内可得：kept-state 续写 vs **复位后全 prefill 参考 16/16 逐 token 相同**、第二轮 keep 复现同 16＋EOS、cursor 链（3911→`92,248046`）自洽；`keep=1` 追加超限守卫（3912+3896）按契约拒绝且会话保留；修正序列的逐对恒等复验为另行派发、本任务不做 | 证据 `results/opencode-execution/strata-unified-p2r3b-keep-gpu/`（RAW 原文、pre/post 状态、`client/p2r3b-compare.json`）；e2e 文档 §P2R-3b＋capabilities/session-state/serve-protocol 的"待 P2R-3b"行据实更新 |
| P2R-3b-2 P2R-3b 修正序列补证（128 与 append 变体恒等；分支 `phase2r/review-gaps`） | 完成（1 native、一租约、rc=0、39.0 s、无新 fence、锁释放无残留；**B1/B2/B3/B4/B5/B6 PASS**；B0 按字面 FAIL（两处任务文本偏差：执行序列为 §1 原文 3 keep+3 plain、`GEN 128 <P_long>,90` 参考 `prompt_read=3897`＝其全 prompt 长度）、substance PASS；12/12 `DONE`、0 `ERR`、0 越界） | 静态 stdin＝task.md §1 13 行逐字节（SHA `e13f7f14…`、无 SAVE/LOAD）＋新 exe `7c441a94…`（跑前复算一致）＋容量自检 4024/4024/4025/4025/3912/3912 全≤4096；每条参考前 plain `SCORE topk=1` reset → 参考 `reused=0` 全 prefill（`PP 3896`/`PP 3897`）；**逐 token 恒等**：`GEN 128 cont=1` vs `GEN 128 <P_long>` 17/17（含 EOS、stop）、`GEN 128 cont=1 90` vs `GEN 128 <P_long>,90` 16/16（stop）、`GEN 16 cont=1` vs `GEN 16 <P_long>` 16/16（length）；续写 `prompt_read` 0/1/0、`reused` 3896/3896/3896、`prompt_ms` 0.0/27.9/0.0 vs 参考 2663.6–4732.5 ms；`GEN 16 cont=1` 首 token 90＝前条 keep-SCORE top-1、两条 90 请求前序 `LOGPROBS` top-1 均 90；证据 `results/opencode-execution/strata-unified-p2r3b2-identity/`；e2e 文档 §P2R-3b-2＋§P2R-3b 更新段（保留首轮 FAIL 原文） |
| P2R-3c 服务层暴露：`/v1/score keep`、`/v1/decisions keep`、`POST /v1/continue`、`GET /v1/session`（PR#2 评审 P0-3 的通用接口部分；分支 `phase2r/review-gaps`） | 完成（**补交付**：原会话在 GPU 腿落证后异常终止、未重跑；工作树代码与证据同版本＝本提交；本任务 0 GPU 只跑单测＋文档＋提交） | 服务会话 11:25–11:28（`hold`＋补偿：preflight exe SHA/size 复核 `7c441a94…`、pre state、`READY` 握手、全 200 请求日志）**8/8 步 HTTP 200**：①能力表 `state_keep_score`/`session_continue` 透传＋`engine_sha256` 匹配；② decisions `keep=true`（响应 `keep:true`、`label_mass`=6.4474e-09）；③/⑥/⑧ session 如实（armed=true/origin=score-keep/pos=3896 → pos=3927 → armed=false/pos=0）；④⑤ `prompt_read` 0/1、`reused` 3896/3911、`prompt_ms` 0.0/27.3 vs 同会话全 prefill 3711.5/2723.3（**4K 前缀不重放**）；④⑤ 与 P2R-3b-2 B3/B2 **逐 token 16/16 相同**（跨运行、不同进程）；`text=null`（服务未带 tokenizer，非 500）；非 keep 清会话、continue 后仍 armed（生命周期表入档）；官方 q36 两端点 501（单测，显式拒绝不静默替代）；`test_score_api.py` **53 全绿**（新增 19＝ContinueSession 16＋OfficialBackendSession 3）、`test_backend_adapters.py` **13 全绿**；引擎源码零改动；原会话孤儿进程由主管清理（锁 `\x00`、无 fence、无残留） | 证据 `results/opencode-execution/strata-unified-p2r3c-service-keep/`（HTTP 原文、引擎诊断 4 行、`client/p2r3c-compare.json`）＋`strata-unified-p2r3c2-finish/`（单测日志、final-report）；e2e 文档 §P2R-3c（如实写明会话落证后异常终止、未重跑、代码与证据同版本） |
| P2R-4a 状态 blob 容器化＋模型指纹＋完整性校验（PR#2 评审 P1"状态与协议可靠性"的引擎部分；分支 `phase2r/review-gaps`） | 完成（0 GPU 交付；只 build＋单测＋哈希剥离自测，无引擎执行/租约/fence；**容器真机矩阵（损坏/错模型/错版本/重复恢复/隔离）待 P2R-4c**） | 同一分支提交 `P2R-4a: ...`（作者 insogao）；新 exe `b07b2c46…`（139,267,072 B，旧 `7c441a94…`）；SAVE 写 `<path>.tmp` 容器→原子替换，`SAVED bytes`＝**容器文件大小**（比旧口径恰多 96 B 头，已文档化；LOADED 同口径）；LOAD 固定顺序校验 大小→magic→version→header_bytes(v1 要求≥96 且长度自洽)→n_vocab→max_context(相等)→pos(≤ctx)→payload CRC32→model_fp，**每类不符独立 `ERR` 且校验失败不进入导入路径、活动会话（含 armed）一字不动**；旧裸 blob（0xaf143cd8 开头）明确 `ERR legacy raw state blob …`（迁移＝旧构建 LOAD 后新构建重 SAVE，已文档化）；指纹＝`SHA-256(size_le64‖首 min(1MiB)‖尾 min(1MiB))` 启动时算一次并缓存（只读 2 MiB＋8 B，非安全用途；pack 不在指纹内并写明理由）；`STATE` 新增 `container_version`(恒 1)/`model_fp_prefix`(16 hex)/`payload_bytes`，既有字段语义不变，`not_covered` 增 `pack`；命令集与 `ERR expected:` 文本不变；`test_score_api.py` 53 全绿＋`test_backend_adapters.py` 13 全绿（未改测试）；SHA-256/CRC32 源码逐字节剥离自测：FIPS 向量 5/5＋zlib CRC 向量 3/3＋LE 往返 3/3 全 OK（0 GPU） | 证据 `results/opencode-execution/strata-unified-p2r4a-state-container/`（build 日志、old/new exe SHA、单测日志、hash-selftest、git show --stat、final-report）；文档三件同步（session_state 含容器格式表/校验顺序与每条 ERR/指纹定位/旧 blob 迁移/"真机矩阵待 P2R-4c"；serve_protocol 三行；本文件） |

### 4.1 夜间任务队列（每轮跟进按此顺序取用；一次只跑一个）

| 序 | 任务 | 入口条件 | 交付与自测判据 |
|---|---|---|---|
| 1 | **362 M2 serve 协议外壳**（在飞） | — | `--serve`；READY 时机；GEN 解析（penalty 非默认→ERR）；PP/T/DONE 字段；EOS 计入 produced；STOP→cancel 且进程续；**同进程 >2 连续请求**；**每请求状态重置 reuse=0**；未知输入 ERR 后继续；EOF 退出 0；serve 未默认开实验门；clean 重建 exit 0；`docs/backends/qwen35_llama_serve_protocol.md` |
| 2 | **363 直连协议冒烟**（GPU 租约） | 362 交付且构建产物可用 | 用 `--serve` 直连喂 GEN：READY→PP→T→DONE 逐字段核对；连续 3+ 请求；中途 `STOP` 取消后进程继续；**状态隔离**（两个不同 prompt 输出互不污染）；EOS 命中；资源门与回落 |
| 3 | **364 官方 server E2E**（GPU 租约） | 363 通过 | 官方 `serve/server.py` 以 `exe+args+cwd+env` 配置驱动本后端：HTTP 普通聊天请求成功；连续请求；取消；状态隔离；**不打印 `INFO batch_slots`** 保持单请求且如实声明 |
| 4 | **365 中立放置**（0 GPU） | 364 通过 | 模型/pack/profile 放到**中立方**（`D:/work/StrataUnified/models*` 硬链接或复制＋文档），运行不再以旧研究树为 cwd；写 `versions_and_hashes`/`run_and_model_layout` 更新 |
| 5 | **366 最终配对回归**（GPU 租约） | 365 完成 | 最终 exe 对生产冻结 exe `0e00af8a` 同口径配对回归（prefill/decode 两指标 PASS 判据同 359）；输出序列一致性如实；结果写入 `docs/backends/qwen35_llama_prod_regression.md`（追加或新章） |
| 6 | **367 阶段交付汇总**（0 GPU） | 366 通过 | 更新本文件＋PR #1 标题/正文（迁移模块/构建依赖/验证范围/性能对照/已知限制/验收结果）；列明未测项；**不合并 main** |

> 阶段 1 门槛（全部满足才进入阶段 2 分支）：①本仓独立构建 exit 0；②经**官方服务入口**完成普通聊天＋连续请求＋EOS＋取消＋状态隔离；③构建/运行不依赖旧研究工作树；④与旧冻结基线配对回归不劣化。当前：①④已达成；②③由 362–366 补齐。
> 每完成上表一项：更新本文件 §1–§3、把证据写进 `SUPERVISION.md` 与 `construction-state.json`，然后取下一项；**不等待用户在线验收**。

### 4.2 原计划（已被 4.1 取代，保留作背景）

- **M2（官方 Server 集成）**：在 `qwen35_llama` 内实现官方 server 行协议（`READY`/`GEN`/`PP`/`T`/`DONE`/`STOP`），支持**连续请求、取消、状态隔离**；提供 unified 配置/启动入口（用户不必分别管理两个项目）。
- **M3（统一部署＋产品回归）**：经官方服务入口的 E2E（普通聊天、连续请求、EOS、取消、状态隔离）；构建/运行不依赖旧研究工作树；最终产物重跑与旧冻结基线的配对回归。
- **阶段 1 已完成**（见 `docs/PHASE1_DELIVERY.md`）→ 按用户工作流**自主创建第二阶段开发分支**（自 `integration/qwen35-llama` 通过测试的提交起；**不合并 main**）。
- **阶段 2 起步计划（草案）**：①Backend 能力表（capabilities 可查询：模型格式/MTP/并发/logits 支持）；②`read_candidate_logits` 只读接口（Prefill 后不生成、状态不变）；③`/v1/score`、`/v1/decisions` 的**文档化兼容子集**；④Session 句柄与状态导出/导入边界（KV＋GDN 同存）；⑤官方原生 Backend 适配器与 qwen35_llama 的**同接口**化。

## 5. 阻断与待决项

| # | 项 | 状态 |
|---|---|---|
| B1 | PR 标题/正文自动更新 | **已解决**：用户已授予 `Pull requests: Read and write`；PR #1 标题/正文由主管自动维护 |
| B2 | 生产口径 `TPS_true_flush` 不可得（无逐 token 时间戳门） | 已登记；decode 可比量取 `decode_evaluations_per_s` |
| B3 | MTP K1/T3 正确性门未闭合、标注 2-vs-3 质量争议 | 与迁移无关的既有项；**不**作为已验证前提 |
| B4 | 产品树自包含：模型/pack/profile 放置 | **已完成**（365：中立硬链接＋配置模板＋文档；365b/366 中立路径实跑） |
| B5 | 历史实验数据未入库 | 由 `docs/backends/qwen35_llama_excluded_material.md`（SHA 索引）＋旧研究仓追溯 |

## 6. 纪律（持续自主开发下仍然有效）

- 不合并稳定 `main`；不替换生产冻结件（GGUF `66a3ca88…`、exe `0e00af8a…`、M0573、PR#3、3200 默认、ACTIVE25）。
- 单 GPU owner；GPU 实验走租约；诊断数字不作产品速度；无跨量化比较。
- 破坏性操作、改变已批准的根本架构、权限扩大、生产切换 —— 停下等用户决策。
- 阻断性正确性问题（MTP 状态错误、生成结果不一致、数据损坏、CUDA 内存安全）不得以"自测通过"掩盖。
