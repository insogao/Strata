# qwen35_llama：中立路径实跑（任务 365b）

范围：在 365 的中立放置（`D:/work/StrataUnified/`）上实跑现役引擎，验证两件事：
① serve 路径新增的**引擎级 stderr finish 诊断行**（补 364 的"字面 finish 未落盘"缺口）；
② 从 `cwd=D:/work/StrataUnified` 以中立 model/pack 路径发起的真实请求可用（"运行不依赖旧树"的实跑验证）。

## 0. 被测件与运行方式

- exe：`backends/qwen35_llama/build/strata.exe`，**139,215,872 B**，
  SHA256 `f9f4485c40f7d419387a3aabf4f311684ac0407f9a33436854336052ee5a33a5`。
  365b clean 全量重建（`build.ps1`，384 目标，exit 0，504 s）；源码改动仅
  `src/program/qwen35_serve.cpp`（+3 行：请求收尾向 **stderr** 打印
  `serve: req <i> finish=... produced=... prompt_ms=... decode_ms=...`，stdout 协议/判定零改动）。
- 模型：中立硬链接 `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`（生产 GGUF；合同禁止重复全量 hash）；
  pack：`models/pack/native_experts.txt`（完成 365 实测的引擎实际所需唯一 pack 文件）。
- 运行方式：`tools/gpu-lock.py run`（declared binary = 该 exe 本身，`binary_sha256_actual` 与申报一致），
  `--stdin-file` 提供 stdin；严格串行、每例一租约；两例 gpu-lock exit 0，cleanup
  `all_exited/active_processes_zero` = true，无 fence 保留。

## 1. 中立性

| 项 | 值 |
|---|---|
| cwd | `D:/work/StrataUnified`（ASCII；CreateProcessW 宽字符 cwd） |
| argv | `--qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --qwen35-pack models/pack --serve --max-context <8192/4096> --qwen35-threads 8 --qwen35-cache-slots 3200 --qwen35-ubatch 4096`（全相对 ASCII 路径） |
| env | 仅 3 项生产门：`STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1`、`STRATA_QWEN35_PACKED_IDS=1`、`STRATA_QWEN35_EXPERT_ORDER=count_desc`；`STRATA_*/GGML_*/LLAMA_GRAPH*/CUDA_LAUNCH_BLOCKING` 已清 |
| PATH 前置 | `/d/work/Strata魔改/runtime/cuda-12.8.1/bin`（记录于 `pre-env/pre-N1.json` / `pre-env/pre-N2.json`） |
| stdin | 主管预生成：`n1-stdin.txt` SHA256 `844cf83ad7428caeac28ceba173f76d77771755a1cbc6ccbabfb2fb71a5694cc`（GEN512 P1 / STOP / GEN16）；`n2-stdin.txt` SHA256 `b35ab2d068f0828523f0d05670b2c1675da23fc8a562ae78f953227f19463b21`（GEN16 ×2） |

**结论：运行不依赖旧研究树的运行时依赖只剩 CUDA DLL**（PATH 前置 `runtime/cuda-12.8.1/bin` 提供
`cudart64_12.dll`/`cublas64_12.dll`/`cublasLt64_12.dll`）；exe、模型、pack 全部取中立路径，两例日志中
loader 实测 `loaded meta data ... 733 tensors from models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` 且请求正常完成。

## 2. 两例结果（stderr 诊断行照录）

### N1（ctx8192；`GEN 512 <P1>` / `STOP` / `GEN 16 <P1>`）

- `READY 8192 stop`；gpu-lock exit 0；进程 elapsed 16.36 s。
- **stderr**（本任务核心交付）：

```text
serve: req 1 finish=stop produced=2 prompt_ms=8801.8 decode_ms=123.0
serve: req 2 finish=stop produced=2 prompt_ms=2794.7 decode_ms=45.6
```

- 同期 stdout `DONE`：

```text
DONE 2 3869 8801.8 123.0 stop 0 0 0 0 0 0 0 0.0 3869 0
DONE 2 3869 2794.7 45.6 stop 0 0 0 0 0 0 0 0.0 3869 0
```

- **如实记（未达期望项）**：期望 req1 `finish=cancel`，实得 `finish=stop produced=2`（P1 在 EOG
  `248046` 处正确早停；req2 同样 stop@2）。机制：`--stdin-file` 把整个 43 KB 输入作为普通文件交给子进程
  stdin，独立读线程毫秒级消费完三行，`STOP` 在请求启动的 stale-reset 之前到达 ⇒ 按官方"请求间 STOP 视为
  stale"规则被丢弃，不构成"运行中取消"。引擎级 `finish=cancel` 行需要**运行中** STOP（如 364 官方 server
  的 live STOP 路径）；本 stdin 形态内没有产生该行，也没有把早停写成"中途取消"。

### N2（ctx4096；`GEN 16` ×2）

- `READY 4096 stop`；gpu-lock exit 0；进程 elapsed 10.77 s。
- **stderr**：

```text
serve: req 1 finish=stop produced=2 prompt_ms=3588.1 decode_ms=98.0
serve: req 2 finish=stop produced=2 prompt_ms=2726.4 decode_ms=33.6
```

- 两行 `finish=stop`：中立路径下的常驻服务正常（同进程连续两请求均被完整服务、无 ERR）。

## 3. 边界

- stderr 诊断行不属于行协议；stdout 协议与全部判定零改动（见
  `qwen35_llama_serve_protocol.md` §"Engine diagnostics on stderr"）。
- 本文不构成性能声明；N1/N2 为诊断实跑（单样本、冷/热首请求混列）。
- 证据（研究仓）：`results/opencode-execution/strata-unified-neutral-run-365b/`
  （`build-365b-clean.log`、`batch/serve-cpp-diff-365b.txt`、`job/job-N1.json`・`job-N2.json`、
  `audit/`、`result/result-*`・`digest-*`、`raw/raw-N*/child_00.log`、`pre-env/`、`batch/state-*`）。
