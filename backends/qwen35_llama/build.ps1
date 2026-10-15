# build.ps1 - standalone build entry for the qwen35_llama backend (imported Strata 0.1.41 tree +
# patched vendor/llama.cpp).  It is NOT wired into the official top-level CMakeLists.txt and must be
# configured/built separately - see docs/backends/qwen35_llama.md, section "Build isolation".
#
# Toolchain (the same recipe the source tree's builds used; reference:
# results/phase2_mtp/consistentSlice300/logs/build300.py in the research repo, read-only):
#   - MSVC vcvars64 from Visual Studio 2022 Build Tools (host x64)
#   - CUDA 12.8.1 nvcc (V12.8.93), pinned path below or $env:CUDA_PATH
#   - Ninja + cmake bundled with the Build Tools, generator Ninja, Release, CMAKE_CUDA_ARCHITECTURES=89
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build.ps1                    # configure + build target strata
#   powershell -ExecutionPolicy Bypass -File build.ps1 -SkipConfigure     # rebuild only
#   powershell -ExecutionPolicy Bypass -File build.ps1 -CudaRoot D:\cuda-12.8.1
#   powershell -ExecutionPolicy Bypass -File build.ps1 -BuildDir D:\scratch\q35 -Jobs 8
#
# Artifact: <BuildDir>\strata.exe  (prints path, size and SHA256 on success)
# This script only builds; it does not load a model or touch the GPU.

param(
    [string]$BuildDir = (Join-Path $PSScriptRoot 'build'),
    [string]$CudaRoot = '',
    [int]$Jobs = 8,
    [switch]$SkipConfigure
)
$ErrorActionPreference = 'Stop'

# --- Visual Studio Build Tools (vcvars64 + bundled cmake/ninja) --------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw "vswhere.exe not found: $vswhere" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'MSVC Build Tools (VC.Tools.x86.x64) not found' }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
$cmake  = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja  = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
foreach ($p in @($vcvars, $cmake, $ninja)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "toolchain file not found: $p" }
}

# --- CUDA toolkit (nvcc) ------------------------------------------------------------------------
if (-not $CudaRoot) {
    $candidates = @(
        @($env:CUDA_PATH, 'D:\work\.strata-qwen35-source\runtime\cuda-12.8.1') |
            Where-Object { $_ -and (Test-Path -LiteralPath (Join-Path $_ 'bin\nvcc.exe')) }
    )
    if ($candidates.Count -gt 0) { $CudaRoot = $candidates[0] }
}
if (-not $CudaRoot -or -not (Test-Path -LiteralPath (Join-Path $CudaRoot 'bin\nvcc.exe'))) {
    throw 'CUDA toolkit not found: pass -CudaRoot <cuda-12.8.1 dir> or set $env:CUDA_PATH'
}
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path
Write-Host "vcvars64 : $vcvars"
Write-Host "cmake    : $cmake"
Write-Host "ninja    : $ninja"
Write-Host "cuda     : $CudaRoot ($((& (Join-Path $CudaRoot 'bin\nvcc.exe') --version | Select-String 'release').Line))"

# --- one cmd.exe with vcvars64 loaded (Ninja needs the MSVC environment) ------------------------
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$bat = Join-Path $BuildDir '_qwen35_llama_build.bat'
$lines = @(
    '@echo off',
    ('call "' + $vcvars + '" >nul 2>&1'),
    'if errorlevel 1 exit /b 901'
)
if (-not $SkipConfigure) {
    $lines += ('"' + $cmake + '" -S "' + $PSScriptRoot + '" -B "' + $BuildDir + '" -G Ninja' +
        ' -DCMAKE_MAKE_PROGRAM="' + $ninja + '"' +
        ' -DCMAKE_BUILD_TYPE=Release' +
        ' -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared' +
        ' -DCMAKE_CUDA_COMPILER="' + (Join-Path $CudaRoot 'bin\nvcc.exe') + '"' +
        ' -DCMAKE_CUDA_ARCHITECTURES=89' +
        ' -DSTRATA_ENABLE_CUDA=ON' +
        ' -DSTRATA_QWEN35=ON' +
        ' -DSTRATA_GGML_DIR="' + (Join-Path $PSScriptRoot 'vendor\llama.cpp') + '"' +
        ' -DSTRATA_MMQ_KQUANTS=ON' +
        ' -DSTRATA_Q6K_EXPERTS=ON' +
        ' -DSTRATA_NATIVE_EXPERTS=ON' +
        ' -DSTRATA_BUILD_TESTS=OFF')
    $lines += 'if errorlevel 1 exit /b %errorlevel%'
}
$lines += ('"' + $cmake + '" --build "' + $BuildDir + '" --target strata --parallel ' + $Jobs)
$lines += 'exit /b %errorlevel%'
Set-Content -LiteralPath $bat -Value $lines -Encoding Ascii

& cmd.exe /d /c $bat
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Host "build failed: exit=$rc (log above; batch kept at $bat)"
    exit $rc
}
$exe = Join-Path $BuildDir 'strata.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "build exited 0 but artifact not found: $exe" }
$item = Get-Item -LiteralPath $exe
# Some Windows PowerShell installations lack Get-FileHash although the compiler is
# available. Stream the artifact through .NET so a successful link is not reported
# as a failed build merely because that optional cmdlet is absent.
$shaStream = [System.IO.File]::OpenRead($exe)
$shaAlgorithm = [System.Security.Cryptography.SHA256]::Create()
try {
    $sha = [System.BitConverter]::ToString($shaAlgorithm.ComputeHash($shaStream)).Replace('-', '').ToLowerInvariant()
} finally {
    $shaStream.Dispose()
    $shaAlgorithm.Dispose()
}
Write-Host ("OK exit=0 artifact={0} size={1} sha256={2}" -f $exe, $item.Length, $sha)
exit 0
