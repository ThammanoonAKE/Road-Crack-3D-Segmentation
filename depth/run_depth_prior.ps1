<#
Build an AI depth prior for the current COLMAP scene and train with it.

Four stages, each resumable and each verifiable on its own:

  1. Depth Anything 3 inference            (.venv-depth, NumPy 1.x)
  2. Per-camera scale and shift alignment  (.venv-gsplat311, NumPy 2.x)
  3. Gaussian Splatting with a depth prior (.venv-gsplat311)
  4. TSDF mesh extraction, with -ExtractMesh

Stage 1 runs in a separate environment on purpose: Depth Anything 3 pins
numpy<2 while the training and meshing environment needs NumPy 2. The stages
only ever communicate through files.

Stage 2 is not optional bookkeeping. A monocular network predicts depth up to
an unknown per-image affine transform, so its raw output cannot be compared
with COLMAP at all. The fit is solved against the exact sparse observations
COLMAP recorded for each image, and scored on observations held out of the fit.
#>
[CmdletBinding()]
param(
    [string]$Scene = "",
    [string]$DepthDir = "",
    [string]$AlignDir = "",
    [string]$ResultDir = "",
    [string]$Name = "road_depth_prior",
    # A DA3 preset name, or any Hugging Face repo id. The default names the 1.1
    # repo explicitly; for GIANT its weights are byte-identical to DA3-GIANT
    # (same SHA256), so this is a label, not a different model.
    [ValidatePattern("^(da3[a-z0-9-]*|[\w.-]+/[\w.-]+)$")]
    [string]$Model = "depth-anything/DA3-GIANT-1.1",
    [ValidateRange(112, 2048)][int]$ProcessRes = 1036,
    [ValidateRange(1, 64)][int]$ViewWindow = 8,
    [ValidateRange(0, 64)][int]$ViewStride = 0,
    [ValidateRange(0.0, 1.0)][double]$MaxMedianAbsRel = 0.12,
    [ValidateRange(1, 500000)][int]$PriorSamples = 40000,
    [ValidateRange(0.0, 1.0)][double]$PriorMinConf = 0.0,
    [switch]$DropSparseDepth,
    [ValidateRange(100, 200000)][int]$MaxSteps = 30000,
    [ValidateRange(0.0, 10.0)][double]$DepthLambda = 0.01,
    [ValidateSet("3dgs", "2dgs")][string]$Trainer = "2dgs",
    [switch]$ExtractMesh,
    [ValidateRange(64, 4096)][int]$MeshTargetGrid = 768,
    [ValidateRange(0.0, 10.0)][double]$MeshVoxelSize = 0.0,
    [ValidateRange(1, 64)][int]$MeshViewStride = 1,
    [string]$MeshOutput = "",
    [ValidateRange(0, 100000)][int]$Limit = 0,
    [switch]$SkipInference,
    [switch]$SkipAlignment,
    [switch]$SkipTraining,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$Scene = if ($Scene) { $Scene } else { Join-Path $root "data\scenes" }
$Scene = [System.IO.Path]::GetFullPath($Scene)
# A repo id contains a slash, which cannot go into a directory name.
$modelSlug = [System.Text.RegularExpressions.Regex]::Replace($Model, '[^A-Za-z0-9._-]', '_')
$DepthDir = if ($DepthDir) { $DepthDir } else { Join-Path $root "runs\depth\$modelSlug" }
$AlignDir = if ($AlignDir) { $AlignDir } else { Join-Path $root "runs\depth\${modelSlug}_aligned" }
$ResultDir = if ($ResultDir) { $ResultDir } else { Join-Path $root "runs\$Name" }
$DepthDir = [System.IO.Path]::GetFullPath($DepthDir)
$AlignDir = [System.IO.Path]::GetFullPath($AlignDir)
$ResultDir = [System.IO.Path]::GetFullPath($ResultDir)

$depthPython = Join-Path $root ".venv-depth\Scripts\python.exe"
$trainPython = Join-Path $root ".venv-gsplat311\Scripts\python.exe"
$sparseDir = Join-Path $Scene "sparse\0"
$imagesDir = Join-Path $Scene "images"

function Require-Path([string]$Label, [string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue)) { throw "$Label not found: $PathValue" }
}
function Invoke-Stage([string]$Label, [string]$Exe, [string[]]$Arguments) {
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    Write-Host "$Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}
function Format-Invariant([double]$Value) {
    return $Value.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture)
}

Require-Path "Scene images" $imagesDir
foreach ($name in @("cameras.bin", "images.bin", "points3D.bin")) {
    Require-Path "COLMAP sparse file" (Join-Path $sparseDir $name)
}
Require-Path "Training python" $trainPython

if (-not $SkipInference) {
    Require-Path "Depth python (.venv-depth)" $depthPython
    $arguments = @(
        (Join-Path $PSScriptRoot "run_ai_depth.py"),
        "--images", $imagesDir,
        "--sparse-dir", $sparseDir,
        "--output-dir", $DepthDir,
        "--model", $Model,
        "--process-res", $ProcessRes,
        "--view-window", $ViewWindow,
        "--view-stride", $ViewStride
    )
    if ($Limit -gt 0) { $arguments += @("--limit", $Limit) }
    if ($Resume) { $arguments += "--resume" }
    Invoke-Stage "1/4 Depth Anything 3 inference" $depthPython $arguments
}
else { Write-Host "1/4 inference: SKIPPED" -ForegroundColor Yellow }

if (-not $SkipAlignment) {
    Require-Path "Depth manifest" (Join-Path $DepthDir "depth_manifest.json")
    $arguments = @(
        (Join-Path $PSScriptRoot "align_depth_to_sparse.py"),
        "--depth-dir", $DepthDir,
        "--sparse-dir", $sparseDir,
        "--output-dir", $AlignDir,
        "--max-median-abs-rel", (Format-Invariant $MaxMedianAbsRel),
        "--write-maps"
    )
    if ($Limit -gt 0) { $arguments += @("--limit", $Limit) }
    Invoke-Stage "2/4 scale and shift alignment" $trainPython $arguments

    $report = Get-Content (Join-Path $AlignDir "alignment_report.json") -Raw | ConvertFrom-Json
    $accepted = $report.summary.images_accepted
    $total = $report.summary.images_total
    Write-Host ("Accepted {0}/{1} images, held-out median abs-rel {2:P2}" -f `
        $accepted, $total, $report.summary.heldout_median_abs_rel.p50) -ForegroundColor Green
    if ($accepted -eq 0) { throw "No image passed alignment; the prior would be empty." }
    if ($accepted -lt [int]($total * 0.5)) {
        Write-Warning "Fewer than half the images passed alignment. Inspect alignment_report.json before training."
    }
}
else { Write-Host "2/4 alignment: SKIPPED" -ForegroundColor Yellow }

if ($SkipTraining) {
    Write-Host "`n3/4 training: SKIPPED" -ForegroundColor Yellow
    Write-Host "Aligned prior ready at $AlignDir" -ForegroundColor Green
    exit 0
}

Require-Path "Alignment report" (Join-Path $AlignDir "alignment_report.json")
# --status prints one line per patched file and then a single verdict line.
$patchLines = @(& $trainPython (Join-Path $root "gaussian_splatting\patch_gsplat_depth_prior.py") --status)
if (-not ($patchLines | Where-Object { $_.Trim() -eq "APPLIED" })) {
    $patchLines | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    throw "gsplat is not patched for the depth prior. Run: `"$trainPython`" " +
          "gaussian_splatting\patch_gsplat_depth_prior.py --apply"
}
$patchLines | ForEach-Object { Write-Host $_ -ForegroundColor Green }

if (Test-Path $ResultDir) {
    throw "Result directory already exists: $ResultDir. Choose another -Name."
}

$trainerScript = if ($Trainer -eq "2dgs") {
    Join-Path $root "third_party\gsplat\examples\simple_trainer_2dgs.py"
} else {
    Join-Path $root "third_party\gsplat\examples\simple_trainer.py"
}
Require-Path "Trainer" $trainerScript

$env:GSPLAT_DEPTH_PRIOR_DIR = $AlignDir
$env:GSPLAT_DEPTH_PRIOR_SAMPLES = [string]$PriorSamples
$env:GSPLAT_DEPTH_PRIOR_MIN_CONF = (Format-Invariant $PriorMinConf)
$env:GSPLAT_DEPTH_PRIOR_KEEP_SPARSE = if ($DropSparseDepth) { "0" } else { "1" }

$arguments = @()
if ($Trainer -eq "3dgs") { $arguments += "default" }
$arguments += @(
    "--data-dir", $Scene,
    "--data-factor", "1",
    "--result-dir", $ResultDir,
    "--max-steps", $MaxSteps,
    "--depth-loss",
    "--depth-lambda", (Format-Invariant $DepthLambda),
    "--disable-viewer"
)
if ($Trainer -eq "2dgs") {
    # Surface-aligned regularisers; the 2DGS trainer has no --save-ply.
    $arguments += @("--normal-loss", "--dist-loss")
}
else { $arguments += "--save-ply" }
Invoke-Stage "3/4 $Trainer training with depth prior" $trainPython (@($trainerScript) + $arguments)

if ($ExtractMesh) {
    $checkpoint = Get-ChildItem (Join-Path $ResultDir "ckpts") -Filter "ckpt_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $checkpoint) { throw "No checkpoint under $ResultDir\ckpts to mesh." }
    $MeshOutput = if ($MeshOutput) { $MeshOutput } else { Join-Path $ResultDir "mesh\mesh_2dgs.ply" }
    $meshArguments = @(
        (Join-Path $root "gaussian_splatting\mesh_from_2dgs.py"),
        "--checkpoint", $checkpoint.FullName,
        "--data-dir", $Scene,
        "--data-factor", "1",
        "--output", $MeshOutput,
        "--target-grid", $MeshTargetGrid,
        "--view-stride", $MeshViewStride
    )
    if ($MeshVoxelSize -gt 0) {
        $meshArguments += @("--voxel-size", (Format-Invariant $MeshVoxelSize))
    }
    Invoke-Stage "4/4 TSDF mesh extraction" $trainPython $meshArguments
    Write-Host "Mesh: $MeshOutput" -ForegroundColor Green
}

Write-Host "`nDone. Result: $ResultDir" -ForegroundColor Green
