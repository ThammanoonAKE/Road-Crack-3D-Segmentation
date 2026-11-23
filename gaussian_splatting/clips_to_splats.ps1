<#
  clips_to_splats.ps1 - one Gaussian splat per video clip.
  ---------------------------------------------------------------------------
  For every video in -Clips, in its own folder under -Work:
    1. frames : the sharpest frame of every 1/-Fps s, full resolution
                (export\colmap_pipeline\extract_video_frames_for_colmap.py);
    2. colmap : SfM with sequential matching
                (export\colmap_pipeline\run_colmap_video_pipeline.ps1 -SfmOnly);
    3. scene  : the largest model undistorted to PINHOLE at -MaxImageSize px,
                laid out as gsplat wants it (images\, sparse\0\);
    4. depth  : with -DepthPrior, Depth Anything 3 depth maps aligned to the SfM
                points (depth\run_depth_prior.ps1 -SkipTraining);
    5. train  : gaussian_splatting\train_rtxpro4000.ps1, run road_crack_<clip>,
                with -DepthPrior supervised by those maps (--depth-loss) and
                named road_crack_<clip>_v2. On 152410 that took the haze out of
                views away from the photos.
  A stage whose output exists is skipped, so after a crash or reboot the same
  command carries on. -Stages limits the run, e.g. -Stages frames,colmap,scene
  to check every clip's registration before committing hours to training, and
  -Only VID_20261102_151605 to one clip. Run it from PowerShell (not -File) so
  the lists bind as arrays:

    .\gaussian_splatting\clips_to_splats.ps1 `
        -Clips C:\3D-segmentation-data\road_crack\raw -Work C:\3D-segmentation-data\road_crack
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Clips,
    [Parameter(Mandatory = $true)][string]$Work,
    [string]$RunPrefix = "road_crack_",
    [ValidateRange(0.5, 30.0)][double]$Fps = 6.0,
    [ValidateRange(640, 4000)][int]$MaxImageSize = 1920,
    [ValidateRange(1000, 100000)][int]$MaxSteps = 20000,
    # Stop densifying earlier for a clip whose splat outgrows the GPU: 152410 (513 frames)
    # reached 9M Gaussians at step 10,500 and the machine rebooted there twice.
    [ValidateRange(0, 100000)][int]$RefineStopIter = 15000,
    [string[]]$Stages = @("frames", "colmap", "scene", "depth", "train"),
    [string[]]$Only = @(),
    [switch]$DepthPrior,
    [ValidateRange(0.0, 1.0)][double]$DepthLambda = 0.05
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent


# COLMAP and tqdm log to stderr, which Windows PowerShell turns into a terminating
# error under "Stop"; judge native commands by their exit code instead.
function Invoke-Native([string]$What, [scriptblock]$Command) {
    $ErrorActionPreference = "Continue"
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

$python = Join-Path $root ".venv-gsplat311\Scripts\python.exe"
$colmap = Join-Path $root "export\colmap_pipeline\colmap_bin\bin\colmap.exe"
$Work = [System.IO.Path]::GetFullPath($Work)

$videos = Get-ChildItem -LiteralPath $Clips -File | Where-Object { $_.Extension -match '^\.(mp4|mov|m4v)$' } | Sort-Object Name
if ($Only) { $videos = $videos | Where-Object { $Only -contains $_.BaseName } }
foreach ($video in $videos) {
    $clip = $video.BaseName -replace '^VID_\d{8}_', ''   # VID_20261102_151605 -> 151605
    $dir = Join-Path $Work $clip
    Write-Host "== $($video.Name) -> $dir" -ForegroundColor Cyan

    $frames = Join-Path $dir "frames"
    if ($Stages -contains "frames" -and -not (Test-Path (Join-Path $frames "*.png"))) {
        # The extractor reads a folder of videos; give it one holding just this clip.
        $source = Join-Path $dir "video"
        New-Item -ItemType Directory -Force $source | Out-Null
        $link = Join-Path $source $video.Name
        if (-not (Test-Path $link)) { New-Item -ItemType HardLink -Path $link -Target $video.FullName | Out-Null }
        Invoke-Native "frame extraction" {
            & $python -I (Join-Path $root "export\colmap_pipeline\extract_video_frames_for_colmap.py") `
                --source-dir $source --output-dir $frames --target-fps $Fps
        }
    }

    $sfm = Join-Path $dir "colmap"
    if ($Stages -contains "colmap" -and -not (Test-Path (Join-Path $sfm "sparse\0\images.bin"))) {
        Invoke-Native "SfM" {
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "export\colmap_pipeline\run_colmap_video_pipeline.ps1") `
                -Images $frames -Out $sfm -Stride 1 -SfmOnly
        }
    }

    $scene = Join-Path $dir "scene"
    if ($Stages -contains "scene" -and -not (Test-Path (Join-Path $scene "sparse\0\images.bin"))) {
        # The mapper writes one model per connected component; keep the one with most images.
        $best = & $python -c "import pycolmap, pathlib, sys; ms = [p for p in pathlib.Path(sys.argv[1]).iterdir() if (p / 'images.bin').exists()]; print(max(ms, key=lambda p: pycolmap.Reconstruction(str(p)).num_reg_images()))" (Join-Path $sfm "sparse")
        Invoke-Native "image_undistorter" {
            & $colmap image_undistorter --image_path $frames --input_path $best --output_path $scene `
                --output_type COLMAP --max_image_size $MaxImageSize
        }
        $model = Join-Path $scene "sparse\0"
        New-Item -ItemType Directory -Force $model | Out-Null
        Get-ChildItem -LiteralPath (Join-Path $scene "sparse") -File | Move-Item -Destination $model
        $count = (Get-ChildItem -LiteralPath $frames -Filter *.png).Count
        & $python -c "import pycolmap, sys; r = pycolmap.Reconstruction(sys.argv[1]); print(f'registered {r.num_reg_images()} of {sys.argv[2]} frames, {r.num_points3D():,} points, mean reprojection error {r.compute_mean_reprojection_error():.2f} px')" $model $count
    }

    $prior = Join-Path $dir "depth_prior"
    if ($DepthPrior -and $Stages -contains "depth" -and -not (Test-Path (Join-Path $prior "alignment_report.json"))) {
        Invoke-Native "depth prior" {
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "depth\run_depth_prior.ps1") `
                -Scene $scene -DepthDir (Join-Path $dir "depth_raw") -AlignDir $prior -SkipTraining -Resume
        }
    }

    if ($Stages -contains "train") {
        $name = if ($DepthPrior) { "$RunPrefix${clip}_v2" } else { "$RunPrefix$clip" }
        $final = Join-Path $root "runs\$name\ply\point_cloud_$($MaxSteps - 1).ply"
        if (Test-Path $final) { continue }
        $ckpts = Join-Path $root "runs\$name\ckpts"
        # Phone exposure drifts within a clip; the bilateral grid takes it up so the colours do not average it.
        $extra = @("--post-processing", "bilateral_grid", "--strategy.absgrad", "--strategy.grow-grad2d", "0.0008", "--packed")
        if ($DepthPrior) {
            $env:GSPLAT_DEPTH_PRIOR_DIR = $prior
            $env:GSPLAT_DEPTH_PRIOR_SAMPLES = "40000"
            $extra += @("--depth-loss", "--depth-lambda", $DepthLambda.ToString([System.Globalization.CultureInfo]::InvariantCulture))
        }
        $train = @{
            Scene = $scene; Name = $name; DataFactor = 1; MaxSteps = $MaxSteps; CheckpointEvery = 2500; Strategy = "default"
            RefineStopIter = $RefineStopIter
            VenvDir = (Join-Path $root ".venv-gsplat311"); GsplatDir = (Join-Path $root "third_party\gsplat")
            RunsDir = (Join-Path $root "runs"); ExtraArgs = $extra
        }
        $last = if (Test-Path $ckpts) { Get-ChildItem $ckpts -Filter "ckpt_*_rank0.pt" | Sort-Object { [int]($_.Name -replace '\D', '') } | Select-Object -Last 1 }
        if ($last) { $train.Resume = $last.FullName; Write-Host "resuming from $($last.Name)" -ForegroundColor Yellow }
        elseif (Test-Path (Join-Path $root "runs\$name")) { throw "runs\$name exists without a checkpoint; remove it to start over." }
        $ErrorActionPreference = "Continue"
        & (Join-Path $PSScriptRoot "train_rtxpro4000.ps1") @train
        $ErrorActionPreference = "Stop"
        Remove-Item Env:\GSPLAT_DEPTH_PRIOR_DIR -ErrorAction SilentlyContinue
        if (-not (Test-Path $final)) { throw "training $name ended without $final" }
    }
}
