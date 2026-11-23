<#
Rebuild the gsplat CUDA extension with a chosen set of modules.

Why this exists: gsplat gates its CUDA modules behind BUILD_* environment
variables, and the semantics are easy to get wrong. From csrc/Config.h:

  * set none of them        -> build everything
  * set any of them to 1    -> build ONLY the ones set to 1
  * set some of them to 0   -> build everything except those

The earlier build of this clone used BUILD_2DGS=0 to save time, so
`rasterization_2dgs` was never compiled and the 2DGS trainer fails at runtime
with:

    AttributeError: '_OpNamespace' 'gsplat' object has no attribute 'rasterization_2dgs'

The op is declared and implemented in the sources; only the build excluded it.

By default this script builds 3DGS, 2DGS, Adam, relocation and losses, which is
everything the project's trainers use. 3DGUT stays off because nothing here
uses it and it is expensive to compile.

    powershell -ExecutionPolicy Bypass -File .\gaussian_splatting\build_gsplat_modules.ps1
    powershell -ExecutionPolicy Bypass -File .\gaussian_splatting\build_gsplat_modules.ps1 -Verify
#>
[CmdletBinding()]
param(
    [string]$VenvDir = "",
    [string]$GsplatDir = "",
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    [string]$VsDevCmd = "",
    [ValidateRange(1, 64)][int]$MaxJobs = 6,
    [switch]$With3DGUT,
    [switch]$Without2DGS,
    [switch]$Verify,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$VenvDir = if ($VenvDir) { $VenvDir } else { Join-Path $root ".venv-gsplat311" }
$GsplatDir = if ($GsplatDir) { $GsplatDir } else { Join-Path $root "third_party\gsplat" }
$VenvDir = [System.IO.Path]::GetFullPath($VenvDir)
$GsplatDir = [System.IO.Path]::GetFullPath($GsplatDir)
$python = Join-Path $VenvDir "Scripts\python.exe"
$LogFile = if ($LogFile) { $LogFile } else { Join-Path $root "runs\gsplat_build_2dgs.log" }

function Require-Path([string]$Label, [string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue)) { throw "$Label not found: $PathValue" }
}

function Test-Op([string]$Name) {
    # A registered op resolves through torch.ops; a missing one raises.
    $probe = @"
import sys, torch, gsplat
try:
    getattr(torch.ops.gsplat, '$Name')
    print('OK')
except Exception as error:
    print('MISSING: %s' % error)
"@
    $probeFile = Join-Path ([System.IO.Path]::GetTempPath()) ("gsplat_probe_" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -LiteralPath $probeFile -Value $probe -Encoding utf8
        $output = & $python $probeFile 2>&1 | Select-Object -Last 1
        return [string]$output
    }
    finally {
        Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
    }
}

Require-Path "Virtual environment python" $python
Require-Path "gsplat clone" $GsplatDir

if ($Verify) {
    Write-Host "Checking which rasterization ops the current build registers" -ForegroundColor Cyan
    foreach ($op in @("rasterization_3dgs", "rasterization_2dgs",
                      "rasterize_to_pixels_2dgs", "rasterize_to_indices_2dgs")) {
        $state = Test-Op $op
        $colour = if ($state -eq "OK") { "Green" } else { "Yellow" }
        Write-Host ("  {0,-28} {1}" -f $op, $state) -ForegroundColor $colour
    }
    exit 0
}

Require-Path "CUDA toolkit" $CudaHome
Require-Path "nvcc" (Join-Path $CudaHome "bin\nvcc.exe")

if (-not $VsDevCmd) {
    $candidates = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    )
    $VsDevCmd = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $VsDevCmd) {
        throw "No VsDevCmd.bat found. Install Visual Studio 2022 Build Tools with " +
              "'Desktop development with C++', or pass -VsDevCmd."
    }
}
Require-Path "VsDevCmd" $VsDevCmd

$build2dgs = if ($Without2DGS) { "0" } else { "1" }
$build3dgut = if ($With3DGUT) { "1" } else { "0" }
Write-Host "gsplat clone : $GsplatDir" -ForegroundColor Cyan
Write-Host "Python       : $python" -ForegroundColor Cyan
Write-Host "VsDevCmd     : $VsDevCmd" -ForegroundColor Cyan
Write-Host "Modules      : 3DGS=1 2DGS=$build2dgs ADAM=1 RELOC=1 LOSSES=1 3DGUT=$build3dgut" -ForegroundColor Cyan
Write-Host "MAX_JOBS     : $MaxJobs  (nvcc uses a few GB per job)" -ForegroundColor Cyan
Write-Host "Log          : $LogFile" -ForegroundColor Cyan

New-Item -ItemType Directory -Force (Split-Path -Parent $LogFile) | Out-Null

# The MSVC environment only exists inside a cmd session that has sourced
# VsDevCmd, so the build runs from a generated batch file rather than from here.
$script = @"
@echo off
call "$VsDevCmd" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
set "CUDA_HOME=$CudaHome"
set "CUDA_PATH=$CudaHome"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=$MaxJobs"
set "VERBOSE=1"
set "BUILD_3DGS=1"
set "BUILD_2DGS=$build2dgs"
set "BUILD_ADAM=1"
set "BUILD_RELOC=1"
set "BUILD_LOSSES=1"
set "BUILD_3DGUT=$build3dgut"
"$python" -m pip install -v --no-build-isolation --no-deps --force-reinstall -e "$GsplatDir"
exit /b %errorlevel%
"@
$batch = Join-Path ([System.IO.Path]::GetTempPath()) ("gsplat_build_" + [guid]::NewGuid().ToString("N") + ".cmd")
Set-Content -LiteralPath $batch -Value $script -Encoding ascii

try {
    Write-Host "`nBuilding. This recompiles the CUDA extension and takes a while." -ForegroundColor Yellow
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$batch`"") `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err"
    if ($process.ExitCode -ne 0) {
        Write-Host "`nBuild failed. Last lines of $LogFile :" -ForegroundColor Red
        Get-Content $LogFile -Tail 30 -ErrorAction SilentlyContinue
        Get-Content "$LogFile.err" -Tail 20 -ErrorAction SilentlyContinue
        throw "gsplat build failed with exit code $($process.ExitCode)"
    }
}
finally {
    Remove-Item -LiteralPath $batch -Force -ErrorAction SilentlyContinue
}

Write-Host "`nBuild finished. Verifying registered ops." -ForegroundColor Cyan
$failed = @()
foreach ($op in @("rasterization_3dgs", "rasterization_2dgs",
                  "rasterize_to_pixels_2dgs", "rasterize_to_indices_2dgs")) {
    $state = Test-Op $op
    if ($state -eq "OK") {
        Write-Host ("  {0,-28} OK" -f $op) -ForegroundColor Green
    }
    else {
        Write-Host ("  {0,-28} {1}" -f $op, $state) -ForegroundColor Red
        $failed += $op
    }
}
if ($failed.Count -gt 0) {
    throw "Build completed but these ops are still missing: $($failed -join ', ')"
}
Write-Host "`ngsplat now provides the 2DGS rasterizer." -ForegroundColor Green
