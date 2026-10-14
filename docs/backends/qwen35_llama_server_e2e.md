# qwen35_llama 官方 Server E2E（任务 364 / 364b / 364c）

2026-10-14 20:44–20:52 +08:00。阶段 1 门槛的 M3 项：**经官方服务入口**（`serve/server.py`）使用 qwen35_llama 完成
普通聊天、连续请求、取消与状态隔离。前两次尝试的停因（沙箱外 `C:/Users/65487/miniconda3` 被权限 auto-reject；`tools/gpu-lock.py`
的 `DENY_BASENAMES` 禁止 python.exe 作 declared binary）与主管决定（改用工具自带 **hold** 模式 + 补偿证据，不做改名绕过）
见任务目录 `codex364c-hold-addendum.md` 与 state `gpu_lock_deny_basenames_20261014`。本文件记录实际执行、结果与证据边界。

## 0. 被测对象与 server config

- 引擎 exe：`D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe`
  - sha256 `2f47b88a9969d04ed5a6fafed6a561b80f51c2945d054e0828e9d9c4a297ebd1`，139,216,384 B。
  - **跑前任务自复算并逐一比对通过**（`pre-run-evidence-364c.json`）；本任务**未重新构建** exe。
- 模型（只读）：`D:/work/Strata魔改/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`，13,676,723,168 B，
  已登记完整 SHA `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`（生产冻结值；主管 2026-10-14 复算全文件确认）。
- tokenizer 包（中立位置，任务目录内清单+逐文件 SHA 见 `pre-run-evidence-364c.json`）：
  `D:/work/StrataUnified/models/qwen36-pack/tokenizer/`
  - `vocab.json` 5,737,005 B `4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4`
  - `merges.txt` 3,600,844 B `080cf95432173729d160346b7882bd2a644b07f7a3813a3eb73f4e66787bfde9`
  - `token_type.json` 744,960 B `5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b`
  - `chat_template.jinja` 8,057 B `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c`
  - `tokenizer.json` 554 B `2971ee07b719ef5dcac1f869fb6124fbb4e20f9dedf34e9a84f2010c98b518f5`
- server config（`server-config.json` 全文）：

```json
{
  "exe": "D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe",
  "cwd": "D:/work/Strata魔改",
  "args": ["--qwen35-model", "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf", "--qwen35-pack", "models", "--serve",
           "--max-context", "4096", "--qwen35-threads", "8", "--qwen35-cache-slots", "3200", "--qwen35-ubatch", "4096"],
  "tokenizer": "D:/work/StrataUnified/models/qwen36-pack/tokenizer",
  "model_name": "qwen36-35b-a3b-ud-iq3_s",
  "env": {"STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE": "1", "STRATA_QWEN35_PACKED_IDS": "1", "STRATA_QWEN35_EXPERT_ORDER": "count_desc"},
  "log": ".../strata-unified-server-e2e-364/engine-serve.log"
}
```

注：server 自己会为引擎 argv 前置 `--serve`（`serve/server.py:819`），config 的 args 里也带一个 ⇒ 引擎启动行里出现重复
`--serve`（惰性重复，无行为影响；engine-serve.log 第 1 行照录）。

- server 运行解释器（沙箱内自包含 Python）：`D:/work/StrataUnified/runtime/python/python.exe`，
  sha256 `f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716`（复算一致）。
- 环境补件（本次，登记）：第一次 server 启动 rc=1，`tools/strata_tokenizer.py:22 import regex` 缺 `regex` 模块
  （日志保留 `server-364c-first-attempt-stderr.log`）。经 pip 下载并 `--target` 装入沙箱 site-packages：
  `regex-2026.9.29-cp313-cp313-win_amd64.whl`（纯环境补件，不改任何源码/官方文件）。

## 1. 租约形态（hold）与绑定弱化登记

- GPU 工作由 server 拉起的引擎**孙进程**执行；`gpu-lock run` 无法把引擎声明为 binary，python 又在其 `DENY_BASENAMES` 中。
- 按主管决定使用工具自带 **hold** 形态：只取排他锁、写 audit、睡 N 秒、释放，**不声明 binary**（绑定更弱）。
  - 有效 hold：pid `15312`，`acquired_at 2026-10-14T20:45:47.048`，`hold-seconds 900`，`wait-seconds 60`，
    audit/ready 在任务目录（`hold-audit.json` / `hold-ready.json`）。
  - 首次 hold（pid 47708，20:44:02 取锁）因上条 regex 缺失的 server 崩溃而未产生任何 GPU 工作；20:45:0x 被
    `taskkill /PID 47708 /T /F` 结束（记录于 `hold-first-attempt-note.json`），随后起新 hold。两段 hold 均无 fence。
  - 收尾：20:51:16 `taskkill /PID 15312 /T /F`（server 先杀、hold 后杀，方法/rc 见 `state-post-364c.json`）；
    hold 未睡满 900 s 即被杀 ⇒ `hold-result.json` 不生成（如实登记，替代证据为锁探测）。
- **补偿证据（本次实际取得）**：
  1. 跑前引擎 exe SHA 与 `2f47b88a…` 一致（§0）。
  2. 引擎确由该 server 拉起：`engine-serve.log:1`
     `[strata] 2026-10-14 20:45:48 engine started: <exe> --serve …`（server 落盘行）；引擎加载行（40 层 × 256 experts、
     `n_ctx 4096`、native tier 3200 slots）同文件。
  3. **ready 证据边界**：引擎字面 `READY <ctx> stop` 由 server 从引擎 stdout 管道读走（`serve/server.py:1041` `_ready_pump`），
     官方 server 不持久化该行。可落盘的派生证据：server stdout
     `ready: http://127.0.0.1:8095/v1  (… context 4096 tokens)`（`serve/server.py:6922`；`max_context` 只可能来自 READY 解析，
     `:843-849`）。本报告不把“字面 READY 行已落盘”写成事实。
  4. 批末全子树零：无 `strata.exe` 残留；锁探测 `gpu-lock hold --hold-seconds 1` **一次取得、0 s 等待**（⇒ 锁已释放）
     并正常释放（`lock-probe-364c-result.json`）；`gpu.lock` SHA 与批前一致；`marker_exists=false`；GPU `1303 MiB, 0%`。
- 结论登记：**本次 exe 绑定弱化到 hold（无 declared binary）**，补偿证据如上；这是官方 server 自拉引擎的 E2E 必要代价。

## 2. 启动命令全文

```text
PATH 前置 D:\work\Strata魔改\runtime\cuda-12.8.1\bin        （引擎加载 cudart 用）
STRATA_BATCH_TRACE=1                                        （官方诊断：想把引擎控制行回声到 server stderr；见 §4 实测无效）
清空继承的 STRATA_*（除上述）、GGML*、LLAMA_GRAPH*、CUDA_LAUNCH_BLOCKING（三项生产门由 config env 注入引擎）

D:/work/StrataUnified/runtime/python/python.exe serve/server.py --engine strata \
  --config D:/work/Strata魔改/results/opencode-execution/strata-unified-server-e2e-364/server-config.json \
  --tokenizer D:/work/StrataUnified/models/qwen36-pack/tokenizer \
  --host 127.0.0.1 --port 8095
（cwd = D:/work/StrataUnified/strata；server stdout/stderr 全文落任务目录 server-364c-stdout.log / server-364c-stderr.log）
```

- 启动到可用：20:45:48 引擎启动 → `ready:`（context 4096）；首请求 20:46:12。无 API key（未设 `STRATA_API_KEY`）。
- 客户端：本地 python（stdlib `urllib`/`socket`，无新依赖），脚本见任务目录 `e2e-client-364c.py` / `e2e-client2-364c.py`。

## 3. 用例结果

### 3.1 pass 1（任务书字面口径：默认 thinking，`max_tokens=16`，`temperature=0`，串行同 server）

| 例 | 请求 | HTTP | finish | 文本 | 耗时 |
|---|---|---|---|---|---|
| E1 A1 | 非流式 prompt A（内存泄漏） | 200 | length | `content=null`，`reasoning_content="Here's a thinking process:\n\n1.  **Analyze User Input:**"` | 6,700.7 ms（冷，prefill 2.6 s） |
| E2 A2 | 同 A1 | 200 | length | 与 A1 **全消息逐字节一致** | 826.0 ms |
| E4 B1 | prompt B（TCP 三次握手） | 200 | length | 16 token 内与 A1 **相同**（thinking 前缀） | 1,498.2 ms |
| E3 S1 | 流式 prompt S，`max_tokens=128`，收 6 个 chunk 后客户端 abort | 200（HTTP/1.0） | 中止于 920 ms | 收到 6 chunks / 1,629 B | 920.0 ms |
| E3 A3 | 取消后 prompt A | 200 | length | 与 A1 **全消息逐字节一致** | 654.7 ms |

- `timings.cache_n=0`（reused=0 语义）；usage completion=16（A1/A2/A3）、18（B1）。
- pass 1 的字面 16-token 口径下模型处于 thinking 阶段：`content` 为 null、文本在 `reasoning_content`，**A 与 B 在
  16 token 内不可分辨** ⇒ E4 隔离需修正用例才能成立（见 3.2），如实登记。
- 修正摘要存 `client-summary-364c.json`（原脚本只比较了 null 的 `content`，已按原始响应重算并注明）。

### 3.2 pass 2（修正口径：`chat_template_kwargs={"enable_thinking": false}`，`max_tokens=64`，同 server 同租约）

| 例 | 请求 | HTTP | finish | 文本（摘要） | 耗时 |
|---|---|---|---|---|---|
| P2-A1 | prompt A | 200 | **stop（EOS）** | “内存泄漏是指程序在运行过程中动态分配了内存…耗尽系统可用内存的现象。” | 4,283.2 ms（较慢，95.96 ms/token） |
| P2-A2 | prompt A | 200 | **stop** | “…导致该内存无法被操作系统重新分配给其他程序使用，从而造成内存资源的持续浪费甚至耗尽。” | 1,597.8 ms |
| P2-B1 | prompt B | 200 | **stop** | “TCP三次握手是客户端与服务器在建立可靠连接前，通过三次交互（SYN、SYN-ACK、ACK）…的过程。” | 2,927.3 ms |
| R1/R2/R3 | prompt A ×3（补充复现） | 200 | **stop** | 与 **P2-A2 逐字节一致**（R1==R2==R3） | 1,416 / 1,105 / 1,082 ms |

- **E1'**：HTTP 200、文本非空（直接答案）、正常 EOS（finish=stop）✔。
- **E4'**：不同 prompt（A vs B）答案不同 ✔；同 prompt 热重复 4/4 逐字节一致（P2-A2 + R1–R3）✔；`reused/cache_n=0` ✔。
- **偏离（如实）**：P2-A1（该 server 的首个长答、明显偏慢的冷态运行）与 P2-A2/R1–R3 不同，但仍是合法 greedy 结果。
  假说：冷专家缓存路径的数值差异翻转了 greedy argmax；未在本任务判据内证明，未隐藏。取消后的**首个请求**（A3）
  逐字节等于 A1，隔离/清洁成立；P2-A1 并非取消后首答。

## 4. E3 取消：证据与边界

**事实链（服务器侧 + 引擎数字侧）**
- 流式请求 6 个 chunk 后客户端断连；server stdout 记录：
  `[strata] done: 7 tokens in 1 s (12.0 tok/s) (disconnect, cancel=True)`（`server-364c-stdout.log:16`）。
- `/metrics` 该请求记录（`metrics-final-364c.json`）：
  `finish="disconnect", output_tokens=7, engine_generated=7, decode_ms=582.7, prompt_read=22` ——
  `engine_generated` 取自引擎 DONE 段（`serve/server.py:4322` `segments.append(engine.last)`），证明引擎在 **7** 个 token
  处结束（max_new=128），即 STOP 被引擎接受并回 DONE（drain 完成、无 `EngineSilent`；server stderr 无任何异常）。
- 取消后的下一请求 A3 HTTP 200、与 A1 全消息一致（654.7 ms）⇒ 引擎状态未被污染。

**边界（不放大）**
- 引擎字面 `DONE … cancel` 行**未取得持久化证据**：官方 server 的单请求路径（`serve/server.py:1854` `generate()`）
  在消费端提前退出时于 `:1949-1975` 内部发 `STOP` 并静默 drain 到 DONE（`_parse_done`，`:1085`），该行不进 log/日志；
  `except GeneratorExit → finish="disconnect"`（`:4429-4430`）、`_watch_client`（`:5161`）、`cancel.set()`（`:5858`）为服务器侧链路。
- 本任务特意置 `STRATA_BATCH_TRACE=1`（官方诊断，`btrace`，`:724/:1208/:1242`）试图回声控制行，但该环境走
  `generate()` 路径（无 batch slots），其中无 `btrace` 调用 ⇒ server stderr 为 0 字节，未取得字面行。
- 因此按派发要求：**不写成“引擎级取消已验证”**；登记为：取消被 server 正确处置、引擎数字侧显示 STOP→DONE 已发生，
  但**引擎字面 `DONE … cancel` 行未触达**（建议后续如必须字面行，按任务 §4 申请一行 stderr 诊断重建，本任务未做，
  以保持被测 exe=`2f47b88a` 不变）。

## 5. 资源、预算与清理

- native/租约：GPU 引擎启动 **1** 次（首个 server 进程在 GPU 之前死于缺 `regex`，未起引擎；不占 native）、server 进程 2 个（1 死 1 活）；
  ≤4 预算内。串行、无并行 server。
- 无新 cleanup fence：`marker_exists=false`（批前批后）。
- 批后：无 `strata.exe` 残留（taskkill 树杀 server+hold；server 的 winjob `contain` 已随 server 结束）、
  `gpu.lock` 原文未变（SHA `6e340b9c…`）、GPU `1303 MiB, 0%`。

## 6. 未触达项 / 偏离（汇总）

1. 引擎字面 `DONE … cancel` 行未取得（§4 边界；未写成已验证）。
2. 任务书字面 E1（`max_tokens=16`、默认 thinking）返回的是 thinking 前缀（`content=null`、finish=length）；
   修正 pass 2 取得直接答案 + EOS（finish=stop）。两版均留档。
3. P2-A1 与同 prompt 热重复 4/4 不一致的单次偏离：冷/慢首答，未解释（假说见 §3.2），不隐藏。
4. 引擎 argv 重复 `--serve`（config 自带 + server 前置）：惰性，无行为影响。
5. `STRATA_BATCH_TRACE=1` 对该单请求路径无效（§4），不产生任何日志。

## 7. 证据清单

任务目录：`D:/work/Strata魔改/results/opencode-execution/strata-unified-server-e2e-364/`（52 文件 SHA256 清单：
`evidence-manifest-364c.json`）。关键件：

| 文件 | sha256 前缀 | 内容 |
|---|---|---|
| `pre-run-evidence-364c.json` | `cf57c577…` | exe/python SHA 比对 + tokenizer 包清单 |
| `hold-audit.json` / `hold-ready.json` | `045c8e81…` / `6d20e571…` | hold pid/acquired_at |
| `server-spawn-364c.json` | `e89669fa…` | server argv/env/cwd |
| `server-364c-stdout.log` | `c824adbc…` | ready 行、每请求 done 行（含 `disconnect, cancel=True`） |
| `server-364c-stderr.log` | `e3b0c442…`（空） | `generate()` 路径无 btrace ⇒ 0 B（§4） |
| `engine-serve.log` | `8a445e80…` | `engine started`、加载、上下文/专家行 |
| `request-evidence-364c.jsonl` / `…-pass2.jsonl` | `078a8293…` / `668766e8…` | E1–E4/pass2 原始请求与响应文本 |
| `raw-364c/` | — | 每例 HTTP 请求/响应、S1 断流原文、R 复现 transcript |
| `metrics-final-364c.json` | `bbf92275…` | 历史含 cancel 请求的引擎数字（engine_generated=7） |
| `state-pre/post-364c.json` | `7e6971bd…` / `c9f3f500…` | 批前后锁/GPU/进程 |
| `stop-364c.out`、`lock-probe-364c-result.json` | `f0a260ed…` / `2eb514a5…` | 清理与锁释放探测 |
| `server-364c-first-attempt-*.log`、`hold-first-attempt-*` | — | regex 事件与首次 hold 的过程件（保留） |

结论（一行）：**官方服务入口 E2E 成立**——普通聊天（200/文本/EOS，修正口径）、连续 ≥3 请求、同 prompt 热重复字节一致、
两 prompt 输出不同、取消后下一请求正常且无污染；**取消的引擎字面证据未触达**（server 内部 drain 不落盘），其余如实登记。

## 8. 365b 诊断行后的重跑（task 366）

2026-10-14 21:40–21:42 +08:00。目的：用 365b 的**带 stderr 诊断行的新 exe**、从**中立路径**重跑官方 server 取消
用例，取得**引擎级 `finish=cancel`**（server 把引擎 stderr 落到 config 的 `log` ⇒ 诊断行可直接落盘）。

### 8.1 被测对象（全部中立路径）

- 引擎 exe：`D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe`
  - sha256 `f9f4485c40f7d419387a3aabf4f311684ac0407f9a33436854336052ee5a33a5`，139,215,872 B（365b 诊断重建）。
  - **跑前任务自复算并比对通过**（`pre-run-evidence-366.json`）；本任务**未构建**。
- 模型：`models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`（中立树内硬链接，13,676,723,168 B；完整 SHA 66a3ca88… 见 §0）。
- pack：`models/pack/native_experts.txt` 2,672 B `91224828e55ccdf519a81cb2af5b9ca7df3178b25f7ffaa0782db731b2bbd6da`。
- tokenizer：`D:/work/StrataUnified/models/qwen36-pack/tokenizer/`（5 文件 SHA 与 §0 完全一致）。
- server config（`server-config-neutral-new.json` 全文；`log` = 任务目录内引擎日志）：

```json
{
  "exe": "D:/work/StrataUnified/strata/backends/qwen35_llama/build/strata.exe",
  "cwd": "D:/work/StrataUnified",
  "args": ["--qwen35-model", "models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf", "--qwen35-pack", "models/pack", "--serve",
           "--max-context", "4096", "--qwen35-threads", "8", "--qwen35-cache-slots", "3200", "--qwen35-ubatch", "4096"],
  "tokenizer": "D:/work/StrataUnified/models/qwen36-pack/tokenizer",
  "model_name": "qwen36-35b-a3b-ud-iq3_s",
  "env": {"STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE": "1", "STRATA_QWEN35_PACKED_IDS": "1", "STRATA_QWEN35_EXPERT_ORDER": "count_desc"},
  "log": ".../strata-unified-cancel-evidence-366/engine-serve-366.log"
}
```

- server：沙箱 python `f77193cf…` 起 `serve/server.py`（cwd=`D:/work/StrataUnified/strata`），port 8096；
  进程 env 清 `STRATA_*/GGML*/LLAMA_GRAPH*/CUDA_LAUNCH_BLOCKING` 后仅设 364c 同款 `STRATA_BATCH_TRACE=1`（本证据
  不依赖它），PATH 前置 `runtime/cuda-12.8.1/bin`（`server-spawn-366.json`）。`ready:` 行落 `server-366-stdout.log`。
- 引擎启动行（`engine-serve-366.log` 第 1 行，cwd 中立）：`engine started: D:/work/StrataUnified/...strata.exe --serve
  --qwen35-model models/Qwen3.6-...gguf --qwen35-pack models/pack --serve --max-context 4096 ...`（重复 `--serve` 同 §0 注）。

### 8.2 用例与结果

| 用例 | 请求 | HTTP | 结果 | 引擎 stderr 对应行 |
|---|---|---|---|---|
| C1（字面） | 非流式 16 tok, temp 0 | 200 | thinking 前缀（`content=null`，finish=length）——与 364 §6.2 同一字面口径偏离 | `req 1 finish=length produced=16` |
| C1b（修正） | `enable_thinking=false`, 64 tok | 200 | 完整答案 + EOS（finish=**stop**，44 tok） | `req 2 finish=stop produced=44` |
| C2（取消） | 流式 128 tok，6 chunk 后客户端 abort（363.7 ms） | `HTTP/1.0 200 OK` | abort 于生成中（`aborted_mid_stream=true`） | **`req 3 finish=cancel produced=8`** |
| C3（污染检查） | 非流式 16 tok（C1 同参数），abort 后 2 s | 200 | 正常返回（finish=length） | `req 4 finish=length produced=16` |

### 8.3 引擎 stderr 的 `serve: req` 行（照录，`engine-serve-366.log`）

```
serve: req 1 finish=length produced=16 prompt_ms=1632.2 decode_ms=815.0
serve: req 2 finish=stop produced=44 prompt_ms=183.2 decode_ms=1269.1
serve: req 3 finish=cancel produced=8 prompt_ms=211.2 decode_ms=183.2
serve: req 4 finish=length produced=16 prompt_ms=233.3 decode_ms=323.3
```

**`req 3 finish=cancel` 即 364 未触达的引擎级取消行，本次由 365b 诊断行在官方 server 实时 STOP 路径上取得。**
server 侧一致：stdout `[strata] done: 8 tokens in 0 s (44.0 tok/s) (disconnect, cancel=True)`；
`metrics-after-366.json` 该请求 `finish="disconnect", output_tokens=8, engine_generated=8`（引擎 DONE 段）。
取消路径的专家 prompt-loan 回填照常（`engine-serve-366.log` 该请求后 `slot 3171 read back byte-equal`），C3 200。

### 8.4 租约形态与补偿证据（hold）

- `tools/gpu-lock.py`（`337426d6…`）`hold --hold-seconds 900 --wait-seconds 60`，pid `31100`，
  `acquired_at 2026-10-14T21:40:29.309`；批末 `taskkill /T /F` 记录后返回（**未到期即杀 ⇒ `hold-result.json` 未写，
  登记**）。形态与 §1 相同（引擎是 server 孙进程，python 在 `DENY_BASENAMES`）。
- 补偿证据：跑前引擎 exe SHA 复算 `f9f4485c…` 通过；引擎 `engine started` 行落引擎日志；server `ready:` 行落定；
  批末：无 `strata.exe` 残留、锁释放探测首试即得（`lock-probe-366-result.json`，`wait.attempts=1, exit_code 0`）、
  `gpu.lock` 内容 SHA `6e340b9c…` 未变、无 `.cleanup-required.json`、GPU 回 `1233 MiB / 0%`（`state-post-366.json`）。

### 8.5 登记（不放大）

1. C1/C3 采用任务书字面 16-token 口径 ⇒ 输出为 thinking 前缀、`content=null`（364 §6.2 同一已知偏离）；文本+EOS 由
   C1b 修正口径验证。
2. `STRATA_BATCH_TRACE=1` 在 364 §4 已证无效，本次仅为 364c 同款登记项，不贡献证据。
3. hold 未自然到期（批末即杀；形态、pid、audit 全落盘）。

证据目录：`D:/work/Strata魔改/results/opencode-execution/strata-unified-cancel-evidence-366/`
（config、`engine-serve-366.log` 14,495 B `16413c9d…`、`server-366-stdout.log` `e1d9070e…`、`server-366-stderr.log` 空、
`request-evidence-366.jsonl` `36a4d898…`、`client-summary-366.json` `920b1dfa…`、`metrics-after-366.json` `795af8af…`、
`status-after-366.json`、`pre/state-post-366.json`、hold audit/ready/spawn、`server-spawn-366.json`、
`lock-probe-366-result.json` `656879cc…`、`raw-366/`、客户端脚本 `client-366.py`）。

结论（一行）：**365b 诊断行在官方 server 实时取消路径上取得引擎级 `serve: req 3 finish=cancel produced=8`**；取消后
下一请求正常、专家回填字节相等、锁释放且无 fence/残留——364 §4 的“引擎字面取消证据未触达”缺口由本任务闭合。
