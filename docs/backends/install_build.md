# qwen35_llama：安装与构建（任务 354 实测）

范围：`backends/qwen35_llama/`——从研究仓导入的 Qwen3.6/Qwen3.8-Flash MTP 研究后端（mini Strata
`v0.1.41` 树 + live 补丁态 `vendor/llama.cpp`）。它是**独立 CMake 工程**，不接入官方顶层构建，
必须单独配置/构建（原因见 `docs/backends/qwen35_llama.md` 的 "Build isolation"）。

本文数字全部来自 354 实测，证据目录：
`results/opencode-execution/strata-unified-import-354/`（`build-354-clean.log`、失败/修复日志、
`import-19files.sha256.txt`、`vendor-keys.sha256.txt`）。构建脚本只编译，不加载模型、不运行 exe、不碰 GPU。

## 1. 工具链要求与实测版本（2026-10-14）

| 组件 | 要求 | 实测版本 / 路径（354 构建日志原文） |
|---|---|---|
| MSVC | Visual Studio 2022 Build Tools，宿主 x64，`vcvars64.bat` | MSVC 19.44.35228.0；`cl.exe` 在 `...\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\` |
| CMake | Build Tools 自带（`build.ps1` 用 vswhere 定位） | `cmake 3.31.6-msvc6` |
| Ninja | Build Tools 自带 | `...\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe` |
| CUDA | 12.8.1 nvcc（`-CudaRoot` 或 `$env:CUDA_PATH`） | `Cuda compilation tools, release 12.8, V12.8.93` |
| GPU 架构 | `CMAKE_CUDA_ARCHITECTURES=89` | sm_89（本机 RTX 4060 Laptop 8 GiB） |

`build.ps1` 的先决条件：`vswhere.exe`（`%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\`）能找到带
`Microsoft.VisualStudio.Component.VC.Tools.x86.x64` 的安装。354 构建用的 CUDA 路径
`D:\work\.strata-qwen35-source\runtime\cuda-12.8.1` 是指向 `D:/work/Strata魔改/runtime/cuda-12.8.1`
的 Junction（同一目录）。

## 2. 构建步骤

入口脚本：`backends/qwen35_llama/build.ps1`（写一个 `cmd.exe` 批处理、先 `call vcvars64.bat`，再
cmake configure + `--build --target strata`；成功时打印产物路径/大小/SHA256）。

```
powershell -ExecutionPolicy Bypass -File backends/qwen35_llama/build.ps1                 # configure + build
powershell -ExecutionPolicy Bypass -File backends/qwen35_llama/build.ps1 -SkipConfigure  # 仅重建
# 参数：-BuildDir <dir>（默认 <backend>\build）、-CudaRoot <cuda-12.8.1>、-Jobs <n>（默认 8）
```

configure 旗标（354 实测所用，`build.ps1` 内固定）：

```
-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
-DCMAKE_CUDA_COMPILER=<cuda-12.8.1>\bin\nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=89
-DSTRATA_ENABLE_CUDA=ON -DSTRATA_QWEN35=ON -DSTRATA_GGML_DIR=<backend>\vendor\llama.cpp
-DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF
```

## 3. 354 实测结果（clean 全量）

- 命令：`powershell -ExecutionPolicy Bypass -File backends/qwen35_llama/build.ps1`；日志 `build-354-clean.log`。
- 结果：**exit 0，383 个 ninja 目标，wall 666 s（8 路）**；产物
  `backends/qwen35_llama/build/strata.exe`，**139,208,192 B**，
  SHA256 `6062a714c38744a6adc1cb052f3242c304a3ff32d158545933834455544dc9b4`。
- 构建尝试共 3 次：attempt1 因 MSVC 静态运行库 LNK1169 链接失败 → 以
  `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` 修复；attempt2 增量重链成功；attempt3 clean 全量成功
  （`build-354-attempt1-failure.txt`、`build-354-attempt2-linkfix.log` 留档）。
- 非确定性：MSVC `/DEBUG` 会把 PDB 签名/时间戳写进 exe，跨构建会话 exe 字节不可复现——
  只在同一次日志/同一构建内比较 SHA。

## 4. 运行前提（本机实测注意点）

- **CUDA 运行时 DLL 目录**：运行 exe 需要 CUDA 12.8 运行库。本机目录
  `D:/work/Strata魔改/runtime/cuda-12.8.1/bin` 含 `cudart64_12.dll`、`cublas64_12.dll`、
  `cublasLt64_12.dll`（2026-10-14 实测列出）。355/358/359 的实际做法是把该目录以 **POSIX 写法**
  前置到 PATH：`/d/work/Strata魔改/runtime/cuda-12.8.1/bin`（见 `docs/backends/run_and_model_layout.md`）。
- 本机实际回归（355/358/359）以 `cwd=D:/work/Strata魔改`（研究树）运行，engine 文件参数全部为
  **ASCII 相对路径**，中文绝对路径只出现在 CreateProcessW 的宽字符 cwd；先把
  `STRATA_*`/`GGML_*`/`LLAMA_GRAPH*`/`CUDA_LAUNCH_BLOCKING` 清干净，再按用例逐项设置 env。
  （355/358/359 的实际 argv/env 见 `docs/backends/run_and_model_layout.md` 与各回归报告。）
- **中立运行布局（任务 365 就绪，2026-10-14）**：运行数据已硬链接到中立位置
  `D:/work/StrataUnified/models/`（含 `pack/native_experts.txt`）与 `D:/work/StrataUnified/profiles/`；
  `cwd` 可用 `D:/work/StrataUnified`，engine argv 用 `--qwen35-pack models/pack`
  （官方 server 模板 `backends/qwen35_llama/server-config.template.json`）。布局与校验详见
  `docs/backends/run_and_model_layout.md` §2.1、`versions_and_hashes.md` §4.1；从新路径的**实跑验证属
  任务 365b**，本条目为放置事实。
- 运行 exe 需要模型与 pack（见 `docs/backends/run_and_model_layout.md`）；模型权重不在本仓库内。

## 5. 依赖与 pin（不可用归档补丁从纯 pin 重建）

- Strata 基线：**v0.1.41 = `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`**（`UPSTREAM_QWEN35.json`）。
- llama.cpp pin：**b11476 = `988190680d5a89fce97de3c20df2c2813731fd61`**（同上）。
- in-tree `vendor/llama.cpp`：**约 178 MB（177,637,522 B，2026-10-14 本仓复核），导入的是 live 补丁态**。
  归档的手工补丁（6 文件口径）不含 `include/llama.h` / `src/llama-batch.cpp`，而 live 树在这两个文件里
  带有 runner 依赖的 `llama_batch_ext`/`llama_process` API——**不能仅凭归档补丁从纯 pin 重建 vendor，
  必须使用导入的 live 树本体**（353 资产盘点 §4/§9 G1；`docs/backends/qwen35_llama.md` 同结论）。
- 关键文件 SHA256（导入件，`vendor-keys.sha256.txt`）：
  - `vendor/llama.cpp/include/llama.h` = `121aef7c9aa5204d255a30951022ef249a0f657faec28d1d545e2da501c26abb`
  - `vendor/llama.cpp/src/llama-batch.cpp` = `b1809b99544c1d63e3a0cf306666b26d2d79c7d86042e69765f9a06346b1c59b`
  - `vendor/llama.cpp/src/models/qwen35moe.cpp` = `f617c8b012be90b7a3d3e9adb3131308e9f58ea5e57f3853631b1d5620e9d3b1`
  - `vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` = `b877b3260ff56555858e389cdca4f5e9339a9ac2480789c4e1ef9e0d46840336`
- 导入来源树：`results/worktrees/phase2-mtp232`，HEAD `9bb9b6a556376f003d563f3e07a5d1e6dc91757f`
  ＋17 个 dirty 文件＋2 个未跟踪 txn 文件；导入 6146 文件 / 217,233,773 B；19 个增量文件与
  353 归档、live 双源逐字节相等（`import-19files.sha256.txt`）。

## 6. 边界

- 本后端为研究件；MTP K1 链、COMPACT 等均 EXPERIMENTAL（default off），本进口不作任何质量/性能认证。
- 构建隔离是刻意的：官方顶层与本后端目标名冲突（`strata`、`strata_spec`、`strata_kernels_cpu`、…），
  且 vendor 是打补丁的 llama.cpp；不要把本后端加进官方 CMake 树。
