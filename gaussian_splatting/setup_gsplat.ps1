<#
Creates a Blackwell-compatible gsplat environment for RTX PRO 4000.

Run from a Visual Studio Developer PowerShell so `cl.exe` is on PATH.
CUDA Toolkit 12.8+ must be installed separately because gsplat JIT-compiles
CUDA extensions for the local GPU.
#>
[CmdletBinding()]
param(
    [string]$VenvDir = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv-gsplat311"),
    [string]$GsplatDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "third_party\gsplat"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing '$Name'. $Hint"
    }
}

Require-Command git "Install Git for Windows and run this script again."
Require-Command $Python "Install Python 3.10 or 3.11 and make it available as '$Python'."
Require-Command cl.exe "Open 'Developer PowerShell for VS 2022' after installing Visual Studio C++ Build Tools."
Require-Command nvcc "Install the full CUDA Toolkit 12.8+ and ensure its bin directory is on PATH."

$nvccText = (& nvcc --version | Out-String)
if ($nvccText -notmatch "release\s+(\d+\.\d+)") {
    throw "Could not determine the CUDA Toolkit version from nvcc."
}
if ([version]$Matches[1] -lt [version]"12.8") {
    throw "CUDA $($Matches[1]) is too old for Blackwell. Install CUDA Toolkit 12.8 or newer."
}

& $Python --version
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if (-not (Test-Path $VenvDir)) {
    & $Python -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python was not created at '$VenvPython'."
}

& $VenvPython -m pip install --upgrade pip

# PyTorch 2.9.1 + CUDA 12.8 has Windows wheels and Blackwell support.
& $VenvPython -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

if (-not (Test-Path (Join-Path $GsplatDir ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $GsplatDir -Parent) | Out-Null
    & git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git $GsplatDir
}

& $VenvPython -m pip install -e $GsplatDir
& $VenvPython -m pip install -r (Join-Path $GsplatDir "examples\requirements.txt")

& $VenvPython -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('capability:', torch.cuda.get_device_capability(0))"
Write-Host "Environment ready. Start training with .\gaussian_splatting\train_rtxpro4000.ps1" -ForegroundColor Green
