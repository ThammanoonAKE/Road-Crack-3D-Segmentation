<# Train a COLMAP-initialized 3D Gaussian Splatting scene within RTX PRO 4000's 24 GB VRAM. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Scene,
    [string]$Name = "scene",
    [ValidateRange(1, 8)][int]$DataFactor = 1,
    [ValidateRange(1000, 100000)][int]$MaxSteps = 60000,
    [ValidateRange(1000000, 5000000)][int]$MaxGaussians = 2000000,
    [ValidateRange(1000, 60000)][int]$CheckpointEvery = 10000,
    [string]$DenseInitDir = "",
    [ValidateRange(0, 5000000)][int]$DenseInitMaxPoints = 0,
    [ValidateRange(0.01, 10.0)][double]$InitScale = 0.1,
    [ValidateRange(0.001, 0.999)][double]$InitOpacity = 0.5,
    [ValidateRange(0.01, 1.0)][double]$DenseInitNormalScale = 0.15,
    [ValidateSet("interp", "raw", "ellipse", "spiral")][string]$RenderTrajPath = "interp",
    [ValidateSet("mcmc", "default")][string]$Strategy = "mcmc",
    [ValidateRange(0.0, 1.0)][double]$MeansLR = 0.00016,
    [ValidateRange(0.0, 1.0)][double]$ScalesLR = 0.005,
    [ValidateRange(0.0, 1.0)][double]$QuatsLR = 0.001,
    [ValidateRange(0.0, 1.0)][double]$OpacitiesLR = 0.05,
    [ValidateRange(0, 100000)][int]$RefineStopIter = 15000,
    [string]$VenvDir = (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv-gsplat311"),
    [string]$GsplatDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "third_party\gsplat"),
    [string]$RunsDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "runs"),
    # Passed to simple_trainer.py as given, e.g. @("--post-processing", "bilateral_grid").
    [string[]]$ExtraArgs = @(),
    # A checkpoint of run -Name to continue from, after a crash; the run
    # directory is then reused. Pass the same settings as the original run.
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$Scene = (Resolve-Path $Scene).Path
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Trainer = Join-Path $GsplatDir "examples\simple_trainer.py"
$ResultDir = Join-Path $RunsDir $Name

if (-not (Test-Path (Join-Path $Scene "images")) -or -not (Test-Path (Join-Path $Scene "sparse\0"))) {
    throw "-Scene must contain images/ and sparse/0/. Got '$Scene'."
}
if (-not (Test-Path $VenvPython) -or -not (Test-Path $Trainer)) {
    throw "gsplat is not installed. Run .\gaussian_splatting\setup_gsplat.ps1 first."
}
if ($Resume) {
    $Resume = (Resolve-Path $Resume).Path
    $inside = $Resume.ToLowerInvariant().StartsWith(([System.IO.Path]::GetFullPath($ResultDir) + "\").ToLowerInvariant())
    if (-not $inside) { throw "-Resume must be a checkpoint inside '$ResultDir'." }
    $ExtraArgs += @("--resume", $Resume)
}
elseif (Test-Path $ResultDir) {
    throw "Result directory already exists: '$ResultDir'. Choose another -Name instead of overwriting a run."
}

New-Item -ItemType Directory -Force $RunsDir | Out-Null
$lastStep = $MaxSteps - 1
$checkpointSteps = @()
for ($step = $CheckpointEvery; $step -lt $MaxSteps; $step += $CheckpointEvery) {
    $checkpointSteps += $step
}
if ($DenseInitDir) {
    $DenseInitDir = (Resolve-Path $DenseInitDir).Path
    # What simple_trainer.py loads; labels.npy and summary.json are prepare_dense_init.py extras.
    foreach ($required in @("xyz.npy", "rgb.npy", "normals.npy")) {
        if (-not (Test-Path (Join-Path $DenseInitDir $required))) {
            throw "Dense initialization is missing '$required' in '$DenseInitDir'."
        }
    }
    $initArgs = @("--init-type", "dense", "--dense-init-dir", $DenseInitDir,
                  "--dense-init-max-points", $DenseInitMaxPoints)
}
else {
    $initArgs = @("--init-type", "sfm")
}
if ($Strategy -eq "mcmc") {
    $strategyArgs = @("--strategy.cap-max", $MaxGaussians,
                      "--strategy.refine-stop-iter", $RefineStopIter)
}
else {
    $strategyArgs = @("--strategy.refine-stop-iter", $RefineStopIter)
}
$checkpointSteps += $MaxSteps
# Full validation renders every held-out camera and previously interrupted the
# long run halfway through. Keep it at the final step; checkpoints and PLYs are
# the lightweight progress artifacts during training.
$evalSteps = @($MaxSteps)

& nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

Push-Location $GsplatDir
try {
    # Quality profile: full-resolution input, MCMC densification, and a long
    # optimization.  On an RTX PRO 4000, pass -DataFactor 2 only if this scene
    # genuinely exhausts 24 GB VRAM.
    # tqdm and several CUDA libraries write normal progress messages to
    # stderr. Windows PowerShell otherwise promotes the first such line to a
    # terminating NativeCommandError when ErrorActionPreference is Stop.
    $ErrorActionPreference = "Continue"
    & $VenvPython $Trainer $Strategy `
        --data-dir $Scene `
        --data-factor $DataFactor `
        --result-dir $ResultDir `
        --max-steps $MaxSteps `
        @strategyArgs `
        @initArgs `
        --init-scale $InitScale `
        --init-opa $InitOpacity `
        --dense-init-normal-scale $DenseInitNormalScale `
        --render-traj-path $RenderTrajPath `
        --means-lr $MeansLR `
        --scales-lr $ScalesLR `
        --quats-lr $QuatsLR `
        --opacities-lr $OpacitiesLR `
        --save-steps $checkpointSteps `
        --eval-steps $evalSteps `
        --ply-steps $checkpointSteps `
        --save-ply `
        --disable-viewer `
        @ExtraArgs
    $trainingExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($trainingExitCode -ne 0) { throw "gsplat training failed with exit code $trainingExitCode." }
}
finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
}

Write-Host "Finished: $ResultDir\ply\point_cloud_$lastStep.ply" -ForegroundColor Green
