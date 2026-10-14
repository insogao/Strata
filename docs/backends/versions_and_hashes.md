# qwen35_llama：版本与 SHA 清单

采集时间：2026-10-14（Asia/Shanghai，354–360 各任务当日实测；360 对仓库内项目做了只读复核）。
每条给出来源指针，便于回溯。全部 SHA256 以小写十六进制记录。

## 1. 本项目（新仓）产物与提交

| 项 | 值 | 来源 |
|---|---|---|
| `qwen35_llama` exe | `backends/qwen35_llama/build/strata.exe`，**139,208,192 B**，SHA256 `6062a714c38744a6adc1cb052f3242c304a3ff32d158545933834455544dc9b4` | 354 clean 构建（`results/opencode-execution/strata-unified-import-354/build-354-clean.log`） |
| 导入提交链（当前 `integration/qwen35-llama`） | base `296b59d2f73c8f04a805df29c2fa97655adace78`（v0.1.44.1）→ import `27c987ca1ba3bbc691aea12306129e5846379b28` → smoke 文档 `e0fd2dea82117dc5b41dc76980de648de6b735d5` → 计时回归文档 `c653bef1eba47fbc7dd1b286a2707e470d00c319` → 生产线对照文档 `b5e439176ae2f462d399b52b2c2ebdbf58ea6fce` | `git log`（360 复核） |
| 提交历史备注 | 2026-10-14（355 之后）对提交链做过一次 `filter-branch`（修正 import 提交的作者），import 与其后继提交随之重写：旧 `22c31c5c`→现 `27c987ca`、旧 `d7d65eec`→现 `e0fd2dea`（两组各自树内容/提交信息相同）。354 报告写于重写前，故其正文仍引用旧 SHA。 | `git reflog`（360 复核） |

## 2. 导入源树与归档

| 项 | 值 | 来源 |
|---|---|---|
| 导入源树 | `results/worktrees/phase2-mtp232`（研究仓），HEAD `9bb9b6a556376f003d563f3e07a5d1e6dc91757f`（分支 `codex/mtp-fa-input332`）＋**17 dirty ＋2 untracked**（`qwen35_mtp_txn.hpp/.cpp`） | 354 `evidence-354.md`；353 §2.1 #17 |
| 导入规模 | 6146 文件 / 217,233,773 B；19 个增量文件与 353 归档、live 源双源逐字节相等 | 354 `import-19files.sha256.txt` |
| 归档 | `unified-assets-20261014/`：79 文件 / 6.3 MB；`MANIFEST.json` 78 项；**61/61 有外部源路径的条目源↔副本 SHA256 全等（0 mismatch）**；10/10 worktree 补丁 `git apply --check -R` 通过 | 353；`docs/phase3_60tps/UNIFIED_ASSET_INVENTORY.md` §8 |

## 3. 上游 pin 与 vendor

| 项 | 值 | 来源 |
|---|---|---|
| Strata 基线 | **v0.1.41 = `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`** | `backends/qwen35_llama/UPSTREAM_QWEN35.json` |
| llama.cpp pin | **b11476 = `988190680d5a89fce97de3c20df2c2813731fd61`** | 同上 |
| in-tree `vendor/llama.cpp` | **live 补丁态**，约 178 MB（177,637,522 B，360 复核）；**不可用归档补丁从纯 pin 重建**（归档补丁不含 `include/llama.h`/`src/llama-batch.cpp`，353 §4/G1） | `docs/backends/install_build.md` §5 |
| vendor 关键文件 | `include/llama.h` = `121aef7c9aa5204d255a30951022ef249a0f657faec28d1d545e2da501c26abb`；`src/llama-batch.cpp` = `b1809b99544c1d63e3a0cf306666b26d2d79c7d86042e69765f9a06346b1c59b`；`src/models/qwen35moe.cpp` = `f617c8b012be90b7a3d3e9adb3131308e9f58ea5e57f3853631b1d5620e9d3b1`；`ggml/src/ggml-cuda/ggml-cuda.cu` = `b877b3260ff56555858e389cdca4f5e9339a9ac2480789c4e1ef9e0d46840336` | 354 `vendor-keys.sha256.txt` |
| 新仓 base 版本 | v0.1.44.1（`setup.py:196` `MIN_ENGINE=(0,1,44,1)`；提交 `296b59d2`） | 仓库文件（360 复核） |

## 4. 模型（两条线，均不在本仓内）

| 线 | 路径（研究树，相对 cwd） | 大小 | SHA256 | 备注 |
|---|---|---|---|---|
| 生产 IQ3_S（冻结） | `models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` | 13,676,723,168 B | `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66` | 合同冻结 GGUF；359 只用（不重复全量 hash） |
| MTP 实验 IQ3_S | `models/experimental/mtp-qwen36-5bc3e238/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` | 15,346,432,288 B | `ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f` | 需 `pack-main40`（355/358 实测用；完整 SHA 已复验） |

模型权重**不进仓库**；放置约定见 `docs/backends/run_and_model_layout.md`（产品树自包含为待办）。

## 5. 对照 exe、工具、CUDA

| 项 | 值 | 来源 |
|---|---|---|
| 生产冻结对照 exe | `results/20261010-qwen35-milestone105/strata-105.exe`，**130,165,248 B**，SHA256 `0e00af8a11274c62789f47c1b5d21aed811f33adb9b79608ffb7df5525a906b6` | 359 §0；353 §3 |
| MTP 实验冻结对照 exe | `results/phase2_mtp/consistentSlice300/strata.exe`，**139,107,840 B**，SHA256 `03bd7068f284450cef90aa633bc248c72b7272ee143da9a54443d79db84dde5f` | 355/358 两臂；353 §3 |
| GPU 租约工具（研究仓） | `tools/gpu-lock.py`，SHA256 `337426d63d5e36f199b8bca3e46bea041987a5d68272f426a5976f50da2a7111` | 360 只读复算（2026-10-14） |
| CUDA | 12.8.1（`Cuda compilation tools, release 12.8, V12.8.93`）；运行时 DLL `cudart64_12.dll`/`cublas64_12.dll`/`cublasLt64_12.dll` 位于 `D:/work/Strata魔改/runtime/cuda-12.8.1/bin`（`D:\work\.strata-qwen35-source\runtime\cuda-12.8.1` 为其 Junction） | 354 构建日志；360 只读复核 |
| 构建工具链 | MSVC 19.44.35228.0（cl 14.44.35207）、CMake `3.31.6-msvc6`、Ninja（VS 2022 Build Tools 自带）、`CMAKE_CUDA_ARCHITECTURES=89` | 354 `evidence-354.md` / 构建日志 |

## 6. 本系列文档（同批复核对象）

| 文档 | SHA256（360 复核时） |
|---|---|
| `docs/backends/qwen35_llama.md` | `ed87a4f61d3766d5cffd862fc7939f371f895908e729aca96049a2f61dae742c` |
| `docs/backends/qwen35_llama_smoke.md` | `892b35f2dd91843416b55dc27ad0a93622cd067921c151cc2355a981ccaa4967` |
| `docs/backends/qwen35_llama_regression.md` | `52269c6e9c5f0be9207e11569d72cde02e5cba83e45afc6b2cba55821b90c445` |
| `docs/backends/qwen35_llama_prod_regression.md` | `27cdfea02eb05ecedfb6f627fb15f547f49ec3b2371533e03ccf974fa39ffd78` |
| `install_build.md` / `run_and_model_layout.md` / `backend_selection.md` / `versions_and_hashes.md` / `qwen35_llama_test_evidence.md`（360 新增） | 定稿 SHA256 见证据目录 `results/opencode-execution/strata-unified-docs-360/sources-360.md` §3 |
