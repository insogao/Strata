# Current continuation — 2026-10-16

Latest user scope and implementation supersede the historical next-step list below.
Single GPU only. Canonical development branch: codex/single-gpu-autotune, based
on PR #5 47c4b262. P3-5…P3-9 delivered for review; no production approval.
See [UNIFIED_SINGLE_GPU_DELIVERY.md](UNIFIED_SINGLE_GPU_DELIVERY.md) for exact
commands, NO-GO parameter results, 191 CPU tests, 31 real integration gates and
307 independent evidence checks. No CUDA/native/default server changes.

All three candidates failed the measured gain guards; selected_configuration=null.
Latest baseline reference is valid for current tools, but cross-day Decode slowed
and its cause is unlocated. Performance release gate remains open. The opt-in
single-entry API is a finite Windows/loopback/text-only/nonstream MVP, not a
complete production deployment. No extra independent long exact replay studies.
PR #4/#6 overlap is superseded by the canonical branch, not merged in sequence.
Do not restart the stopped automation or OpenCode. Main and frozen files stay unchanged.

The following handoff is preserved historical evidence.

---

# 交接文档：Strata Unified 开发线（2026-10-15 13:4x，由本线主管 Codex 书写）

> **最新接手结果（2026-10-15）**：用户授权 Codex 直接施工，暂不使用 OpenCode；范围为单 GPU 笔记本/PC，不做双卡施工。当前 `codex/session-correctness` 自 `f3610f82` 修复 F1…F4，100 CPU tests PASS，Windows CUDA 独立构建＋3 个串行 native 已完成并清理。旧17/18 token 问题复现；新版本单段/分段/逐 token 17 IDs 恒等，四项长度拒绝状态不变，保存恢复专项通过。初次状态测试相对路径错误与构建工具失败均留原件，详细说明及可供第三方复核的 raw 在 `docs/SESSION_CORRECTNESS_REVIEW_20261015.md`。HTTP keep/continue 客户端须升级 session id/revision；LOAD 不恢复 logits 的限制仍在。阶段三工具 PR #4 是另一条审查线，不在本分支；不合并 main，不自动开启定时任务。下文为原交接历史。

> 目的：让接手的 agent 在 **30 分钟内**能独立继续这条开发线：知道去哪读、哪些不能碰、怎么验收、哪里会踩坑、还剩什么。
> 本文件写在旧研究仓（监督/证据枢纽）；**同一内容将在 P3-3 收尾后复制进新仓** `D:/work/StrataUnified/strata/docs/HANDOFF_FOR_NEXT_AGENT.md` 并提交（避免与在飞会话的分支切换冲突）。

---

## 1. 一页速览

- **产品线**：把已验证的 mini-strata（Qwen3.6-35B-A3B UD-IQ3_S，RTX 4060 Laptop 8GiB）迁进官方 Strata 的新 fork，形成"官方服务入口 + 统一 Backend + 评分/决策/会话"的可交付形态。
- **仓库**：
  - 新主仓（开发）：`D:/work/StrataUnified/strata`（fork `insogao/Strata`；`upstream` = `Niko1221/Strata`）
  - 旧研究仓（证据/监督枢纽）：`D:/work/Strata魔改`（别名 junction：`D:/work/strata-qwen35`、`D:/work/.strata-qwen35-source` 指向同一处）
- **分支**（本地与 origin 同步）：
  - `integration/qwen35-llama` = `e46956bd` → **PR #1 → main**（阶段一）
  - `phase2/backend-interface` = `954a88e4` → **PR #2**（阶段二原型，base = 阶段一分支）
  - `phase2r/review-gaps` = `dc79e762` → **PR #3**（阶段二**评审补强**，base = 阶段二分支；14 文件 +3892/−354）
  - `phase3/autotuner` = `d350d768`（+ 本轮 P3-3 将追加）→ 阶段三，尚无 PR
- **当前状态**：**外部评审的 P0 三条 + P1 两条已全部交付并验收**（P2R-1…P2R-5）；阶段三 P3-1/P3-2 已验收，**P3-3（短名单 GPU 配对验证）在飞**，P3-4（`strata tune` CLI＋汇总＋独立 PR）待办。
- **合并决定权在用户**：本线**不合并 main、不切换生产**。

## 2. 读什么（按优先级）

1. `D:/work/StrataUnified/strata/docs/PROJECT_STATUS.md` —— **状态索引**（阶段/里程碑/分支提交链/阻断/下一步）。**注意：该文件按分支各有一份拷贝，内容不同步**：
   - `phase2r/review-gaps` 的拷贝最全（§4.0b 有 **P2R-1…P2R-5** 的逐条验收摘要）——**这是补强线的权威状态**；
   - `phase3/autotuner` 的拷贝仍是阶段二版（P2-1…P2-7），**补强与阶段三的行不在其中**；阶段三的事实以 `docs/backends/qwen35_llama_autotuner_foundation.md`（P3-1）、`qwen35_llama_offline_screen.md`（P3-2）、**`qwen35_llama_autotune_pilot.md`（P3-3，本轮结论在这里）** 为准（我已在 phase3 拷贝补了 §4.0c 指针）；
   - 合并（PR #3 → #2 → #1）后这些拷贝会自然收敛。
2. GitHub PR：
   - **PR #3**（当前审查对象）：描述里是"评审要求 → 状态 → 提交 → 关键证据"表＋已知限制＋合并顺序建议。
   - PR #2（阶段二原型）、PR #1（阶段一 → main）。
3. 契约文档（`docs/backends/`，每份自带证据节／待办标注）：
   - `qwen35_llama_serve_protocol.md`（行协议，含 `SCORE … cand=/keep=1`、`GEN … cont=1`）
   - `qwen35_llama_session_state.md`（状态容器 `STRATAST` v1：96B 头/CRC32/模型指纹/定序校验/每条 ERR）
   - `qwen35_llama_score_api_design.md`（`/v1/score`、`/v1/decisions`、`/v1/continue`、`/v1/session`、失败模型与链路隔离）
   - **`qwen35_llama_score_api_e2e.md`**（逐任务真机证据节：§6 P2R-2、§P2R-2b、§P2R-3b、§P2R-3b-2、§P2R-3c、§P2R-4c/4d）
   - `qwen35_llama_capabilities_and_score.md`、`qwen35_llama_backend_adapters.md`（含 20 格路由/能力矩阵＋单入口配方＋未统一清单）、`qwen35_llama_batching_contract.md`
   - 事实锚点：`versions_and_hashes.md`、`run_and_model_layout.md`、`qwen35_llama_excluded_material.md`（未入仓历史材料 SHA 索引）
4. **原始证据（只在旧研究仓，GitHub 上没有）**：`D:/work/Strata魔改/results/opencode-execution/strata-unified-<任务>/`
   - `task.md`（任务书/预登记契约）＋ `codex-<任务>-review-next.md`（**预登记验收判据**）
   - `native*/raw/child_*.log`（引擎 stdout/stderr 原文）、`job/`＋`gpu-lock-run.log`（租约与 argv）
   - `client/*compare*.json`（逐条比对中间值）、`state/`（状态容器与坏件）、`final-report.md`
   - `request-evidence.jsonl`（Go 路由/模型/HTTP200 证据）
5. 机器状态：`D:/work/Strata魔改/construction-state.json`（151 键；近期 `codex_p2r*_dispatch/_result`、`pr3_supplement_complete_20261015`、`handoff_20261015` 等）。
6. 时序日志：`D:/work/Strata魔改/SUPERVISION.md`（约 297 KB，**最新在末尾**，开头是 10-14 的旧历史；只读尾部）。
7. 历史/停止线文档（了解即可，勿混淆当前线）：`D:/work/Strata魔改/HANDOFF.md`（2026-10-14 上午 phase3-60tps R1 版，已过时）、`AGENTS.md`/`REQUIREMENTS.md`（2026-10-14 06:31 的 phase2.2 施工合同，仍含有效的通用纪律）。

## 3. 不能碰的东西（冻结件与禁令）

- 生产冻结：GGUF `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`（13,676,723,168 B）、exe `0e00af8a11274c62789f47c1b5d21aed811f33adb9b79608ffb7df5525a906b6`、M0573、生产默认 `--qwen35-cache-slots 3200`、thread ACTIVE25。
- **不合并 `main`**、不做生产切换（由用户决定）；不替换/覆盖上述冻结件。
- **命名歧义警告**：旧契约里的 "PR#3" 指**上游公开 PR**（冻结项）；`insogao/Strata` 的 **PR #3 是这条线的评审补强 PR**——同名不同物，别混。
- 禁用数字：`4656.79`、`1476.42`、`−1.12s`；诊断插桩数字不作产品速度；**无跨量化比较**。

## 4. 数值口径（本线硬规则）

- 引擎 `SCORE`/`CANDLOGPROBS` 的 logprob 只允许**同状态类**比较（同进程热态）。**同类重复逐字节稳定**；**跨状态类**实测带宽 1.1e-2～7.1e-2（0.0327／0.0599／0.0713 三批实测，随 prompt/状态而变），**根因未定位**——只报事实、不作机制声明。
- 判序规则（限定本机/单量化/exe）：跨类判序需 gap 约 >0.04（保守 0.06–0.07）。
- 单次/小样本必须标口径（n、是否稳态、是否 p95）；`TPS_true_flush` 由 flush 数组复算，n<2 不算 TPS。
- 状态容器：`LOAD` 之后上下文**没有 logits** ⇒ 纯 `GEN cont=1` **不可作 LOAD 后的首请求**（须带 ≥1 个追加 id）——已写入 `not_covered`。

## 5. 运行/纪律基础设施（照做，别自创）

- **单 GPU owner**：`D:/work/Strata魔改/tools/gpu-lock.py`（现 SHA `337426d6…`）。引擎直连用 `run`（declared binary＝**被测 exe 本身**、`--stdin-file`、一例一租约、严格串行）；服务驱动用例用 `hold`＋补偿证据（**禁止改名绕过 DENY_BASENAMES**）。清理围栏 `*.cleanup-required.json` **只有主管核到全子树零/owner 退出后才 release**；异常时保留围栏。
- **窗口纪律（20 分钟）**：以任务书 `task.md` mtime 为锚，每个 native 启动前程序化核余额；**负余额必须先 `touch task.md` 重锚并在证据里登记**——不得在负余额下启动新 native（P2R-4c 出现过一次偏差，已立规）。
- **派发契约**：`STRATA_OPENCODE_OBSERVER_LOG_DIR=<任务证据目录> C:/Users/65487/miniconda3/python.exe tools/opencode-go-executor.py --file <task.md> --file <review.md> -- '<单引号短指令>'`；启动后**核证**：`request-evidence.jsonl` 里模型＝`deepseek-v4.1-flash`、reasoning 流在场、**全 HTTP 200**、且**只有一个 opencode 会话**（禁双会话）。每条指令禁用反引号（会被 shell 展开执行）。
- **沙箱白名单与陷阱**：允许写 `D:/work/Strata魔改/*`、`D:/work/strata-qwen35/*`、`D:/work/.strata-qwen35-source/*`、`D:/work/StrataUnified/*`；**禁写 `/tmp`、`%TEMP%`**；**禁止以 `D:/work` 为根做 glob/递归搜索**（权限会 auto-reject 并**终止会话**——已发生过多次）；任务书里要**列全绝对路径**。
- 其它环境事实：git 需**显式代理**（repo-local `http.proxy=http://127.0.0.1:10808`；系统代理 git/curl 不读）；沙箱内自包含 Python 在 `D:/work/StrataUnified/runtime/python/`；引擎 cwd＝`D:/work/StrataUnified`，模型 `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`＋`models/pack`；argv 照抄既有任务原文。
- PR 权限：`insogao` fine-grained PAT 已含 `Pull requests: Read and write`（可改标题/正文）；缺权限时只报告具体缺项，不申请更大权限。

## 6. 失败模式与处置（这条线实际踩过的）

| 症状 | 处置 |
|---|---|
| 会话 1 分钟内死、`stderr.log` 有 `permission requested: external_directory (D:\work\*); auto-rejecting` | 任务书缺绝对路径＋做了根目录搜索 → 补"路径清单"节＋"禁止以 D:/work 为根 glob"铁规，`touch task.md` 重锚后重派（计为 1 次工具故障） |
| dispatch `exit≠0` 或 `exit 0` 但无 `completion.json` | **exit 0 ≠ 完成**：逐交付目录核收。若证据已落盘、仅文档/提交未做 → 派**补交付任务**（复用既有 harness、禁重跑 E2E/凑数、只用绝对路径）；若孤儿进程/锁/fence 残留 → 核 owner 退出后清理 |
| 会话死时留下 `strata.exe`/服务/`hold` 且锁被占 | 先取证（`tasklist`＋`Get-CimInstance Win32_Process` 取 commandline/parent），确认 owner 已退出且无 fence，再终止子树；**不得绕过租约** |
| 执行方如实报 FAIL | **这是期望行为**，不是执行方失误；先分辨"引擎缺陷 / 主管任务书缺陷 / 口径缺陷"，再决定修复任务（本线三次 FAIL 中有两次是我任务书的错） |
| 数字"看起来太漂亮" | 一律回 raw 原文复算（逐 token 比对、SHA 复算、`Σexp` 复算）；`final-report.md` 只是摘要 |

## 7. 方法（本线一贯的验收法）

1. 派发前**预登记**任务书＋验收清单（判据写成可机器复核的形式：字节相等/逐 token 相等/SHA/固定 ERR 文本/状态前后一致）。
2. 完成后**不只看汇报**：读 `native*/raw/*`、`client/*compare*.json`，必要时自己重算；对"应当失败"的路径（负例）同样取证。
3. 里程碑级：更新 `PROJECT_STATUS.md`＋PR 描述（证据表），把"未解释项/未测项/行为变化"写清楚。
4. 任务书**只写一个动作**；同工作树/GPU 串行；一个会话在飞时不开第二个。

## 8. 未闭合清单（接手后按此推进）

1. **P3-3（在飞）**：短名单前 2 候选 vs 基线、留出形状 S(3896→16)/L(3817→128,ctx8192) 的配对交错验证、预登记 PASS/NO-GO/INCONCLUSIVE＋与 P3-2 预测对照（含 exe 沿革备注）。按 `codex-p3-3-review-next.md` 的 10 条核收。
2. **P3-4**：`strata tune` CLI＋profile schema/失效规则＋阶段三交付汇总＋**独立 PR**（跨卡验证若不可得须标待办）。
3. **PR 评审与合并**：PR #3（可完整审查）→ #2 → #1 → main，合并由用户决定。
4. **数值根因**：冷/热差异（1.1e-2～7.1e-2）未定位；近阈值规则限定本机/量化/exe（跨卡/跨量化未验）。
5. **早期遗留**（来自旧契约，未收入本线）：256-token K 扫描的正式化；serve/EOS 口径扩展；MTP K1/T3 与风险 label=3 的既有争议（**不得**当已验证前提）。
6. **接口未统一项**（已在文档列明）：两监听器并存、官方 `RESTORE/SERR` 会话文法未并入、批处理契约 v1 未实现。
7. **可选**：把精简证据包（compare JSON＋raw 子日志＋task/review 文档）整理进 GitHub（当前 raw 证据不在 GitHub，纯 GitHub 审查者看不到）。

## 9. 别盲信清单

- **别盲信我的任务书字面值**：我犯过三处（P2R-3b 的参考序列把 armed 会话当复位用→越界 ERR；P2R-4c 漏了第二次 SAVE；"4 keep-SCOREs"实数不符）。预登记判据**要自己复核一遍**再派发。
- **别盲信 `final-report.md`**：回 raw。
- **别把预测当结论**（P3-2 的离线模型 hold-out 误差 9.7%/21.3%，只能拒绝不能选优）。
- **别把"自测通过"当正确性**：阻断性问题（状态错误/生成不一致/数据损坏/CUDA 内存安全）必须**真机证据**；本线的做法是"同进程同状态类逐 token 恒等＋负例＋拒绝后会话未坏"。
- **别重跑凑数**：失败留原件、登记，不洗预算。

---

**交接后建议的第一件事**：`P3-3` 的 10 条核收（若在飞未完则先等它）→ 派 `P3-4` → 收口阶段三 PR → 把 PR #3 交给评审/用户决定合并。
