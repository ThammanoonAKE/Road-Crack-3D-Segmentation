<#
  run_colmap_video_pipeline.ps1 - SfM + dense MVS for a long video frame set.
  ---------------------------------------------------------------------------
  The existing run_colmap_pipeline.ps1 uses exhaustive_matcher, which compares
  every pair. That is fine for a few hundred images and impossible for
  thousands: 8,602 frames would be 37 million pairs. Video frames arrive in
  order, so this uses sequential_matcher over a sliding window plus vocab-tree
  loop detection, which is what closes a circuit back onto itself.

  Frames are culled by --Stride rather than copied, using COLMAP's image list,
  so no disk is spent duplicating a 1080p set.

  Stages are separated because the expensive one is dense MVS. Run -SfmOnly
  first and check how many images registered before committing hours to dense.

    powershell -File run_colmap_video_pipeline.ps1 `
        -Images data\project_unique_colmap_v4_10fps\images `
        -Out data\scenes_1080p -Stride 3 -SfmOnly
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Images,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Colmap = "",
    [string]$VocabTree = "",
    [ValidateRange(1, 60)][int]$Stride = 3,
    [ValidateRange(1, 10000)][int]$MaxImageSize = 1920,
    [ValidateRange(1, 100)][int]$MinNumPixels = 2,
    # Dense MVS costs per reference image, and a video set sees the same road
    # dozens of times. Halving the reference images halves the time at full
    # resolution; every frame stays available as a source view, and
    # stereo_fusion still merges across views.
    [ValidateRange(1, 20)][int]$DenseStride = 2,
    # COLMAP's own default is 32 GB. A small cache re-reads images from disk
    # constantly, which is pure loss on a machine with RAM to spare.
    [ValidateRange(1, 256)][int]$CacheSize = 24,
    [string]$CameraModel = "SIMPLE_RADIAL",
    [ValidateRange(1, 200)][int]$OverlapFrames = 15,
    [ValidateRange(0, 500)][int]$LoopDetectionImages = 40,
    # A junction is crossed from several headings, so one nearest neighbour per
    # image is not enough to find every revisit. Raise both of these for a grid
    # of streets rather than a single route.
    [ValidateRange(1, 50)][int]$LoopDetectionNeighbors = 1,
    [ValidateRange(0, 500)][int]$LoopDetectionAfterVerification = 0,
    [ValidateRange(1.0, 5.0)][double]$BaGlobalFramesRatio = 1.3,
    [ValidateRange(5, 200)][int]$BaGlobalMaxIterations = 30,
    [ValidateRange(1, 20)][int]$BaGlobalMaxRefinements = 2,
    [switch]$SfmOnly,
    [switch]$DenseOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
# $PSScriptRoot is not populated while param defaults are evaluated.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Colmap = if ($Colmap) { $Colmap } else { Join-Path $scriptDir "colmap_bin\bin\colmap.exe" }
# COLMAP 4.x indexes with FAISS; the older FLANN vocab trees crash the matcher.
$VocabTree = if ($VocabTree) { $VocabTree } else { Join-Path $scriptDir "vocab\vocab_tree_faiss_flickr100K_words32K.bin" }
$Images = [System.IO.Path]::GetFullPath($Images)
$Out = [System.IO.Path]::GetFullPath($Out)
$Colmap = [System.IO.Path]::GetFullPath($Colmap)
$database = Join-Path $Out "database.db"
$sparseRoot = Join-Path $Out "sparse"
$dense = Join-Path $Out "dense"
$listPath = Join-Path $Out "image_list.txt"

foreach ($item in @(@("Images", $Images), @("COLMAP", $Colmap))) {
    if (-not (Test-Path -LiteralPath $item[1])) { throw "$($item[0]) not found: $($item[1])" }
}
New-Item -ItemType Directory -Force $Out, $sparseRoot | Out-Null

function Require-Path([string]$Label, [string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue)) { throw "$Label not found: $PathValue" }
}
function Step([string]$Message) {
    Write-Host "`n==== $Message ====" -ForegroundColor Cyan
    Write-Host "     $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor DarkGray
}
# COLMAP parses numbers with a decimal point whatever the Windows locale.
function Format-Invariant([double]$Value) {
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}
function Invoke-Colmap([string]$Label, [string[]]$Arguments) {
    Step $Label
    Write-Host "colmap $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Colmap @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

if (-not $DenseOnly) {
    # Cull per source video so each sequence keeps its own even spacing.
    $all = Get-ChildItem -LiteralPath $Images -File |
        Where-Object { $_.Extension -match '^\.(png|jpg|jpeg)$' } |
        Sort-Object Name
    if (-not $all) { throw "No images under $Images" }
    $selected = New-Object System.Collections.Generic.List[string]
    $counters = @{}
    foreach ($file in $all) {
        # Frames are named <video prefix>_<index>_f<frame>_t<ms>.ext
        $key = ($file.Name -replace '_\d+_f\d+_t\d+ms\.\w+$', '')
        if (-not $counters.ContainsKey($key)) { $counters[$key] = 0 }
        if ($counters[$key] % $Stride -eq 0) { $selected.Add($file.Name) }
        $counters[$key] = $counters[$key] + 1
    }
    Set-Content -LiteralPath $listPath -Value $selected -Encoding ascii
    Write-Host "Frame selection: $($all.Count) available -> $($selected.Count) at stride $Stride" -ForegroundColor Green
    foreach ($key in ($counters.Keys | Sort-Object)) {
        $kept = [math]::Ceiling($counters[$key] / $Stride)
        Write-Host ("  {0,-28} {1,6} -> {2,5}" -f $key, $counters[$key], $kept)
    }

    if (-not ($Resume -and (Test-Path $database))) {
        Invoke-Colmap "1/6 feature_extractor" @(
            "feature_extractor",
            "--database_path", $database,
            "--image_path", $Images,
            "--image_list_path", $listPath,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", $CameraModel,
            "--FeatureExtraction.use_gpu", "1"
        )
    }
    else { Write-Host "1/6 feature_extractor: REUSED existing database" -ForegroundColor Yellow }

    $matchArgs = @(
        "sequential_matcher",
        "--database_path", $database,
        "--SequentialMatching.overlap", $OverlapFrames,
        "--SequentialMatching.quadratic_overlap", "1",
        "--FeatureMatching.use_gpu", "1"
    )
    if ($LoopDetectionImages -gt 0 -and (Test-Path -LiteralPath $VocabTree)) {
        # Without loop detection a circuit never closes and the model drifts.
        $matchArgs += @(
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path", $VocabTree,
            "--SequentialMatching.loop_detection_num_images", $LoopDetectionImages,
            "--SequentialMatching.loop_detection_num_nearest_neighbors", $LoopDetectionNeighbors,
            "--SequentialMatching.loop_detection_num_images_after_verification",
            $LoopDetectionAfterVerification
        )
        Write-Host ("Loop detection: ON  images={0} neighbors={1} after_verification={2}" -f `
            $LoopDetectionImages, $LoopDetectionNeighbors, $LoopDetectionAfterVerification) -ForegroundColor Green
        Write-Host "  tree: $VocabTree" -ForegroundColor DarkGray
    }
    else {
        Write-Warning "Loop detection OFF. A closed route will drift without it."
    }
    Invoke-Colmap "2/6 sequential_matcher" $matchArgs

    # Bundle adjustment dominates the mapper once the model is large, and it is
    # single-threaded CPU work here: --Mapper.ba_use_gpu exists but every COLMAP
    # Windows build tested (4.1 and 4.2) ships a Ceres compiled without CUDA or
    # cuDSS and silently falls back to CPU. Measured on this set, these four
    # settings tripled the early registration rate instead.
    Invoke-Colmap "3/6 mapper" @(
        "mapper",
        "--database_path", $database,
        "--image_path", $Images,
        "--output_path", $sparseRoot,
        "--Mapper.ba_global_frames_ratio", (Format-Invariant $BaGlobalFramesRatio),
        "--Mapper.ba_global_max_num_iterations", $BaGlobalMaxIterations,
        "--Mapper.ba_global_max_refinements", $BaGlobalMaxRefinements,
        "--Mapper.ba_global_ignore_redundant_points3D", "1"
    )
}

# The mapper writes one directory per connected component; keep the largest.
$models = Get-ChildItem $sparseRoot -Directory -ErrorAction SilentlyContinue
if (-not $models) { throw "mapper produced no model under $sparseRoot" }
$report = foreach ($model in $models) {
    $imagesBin = Join-Path $model.FullName "images.bin"
    if (-not (Test-Path $imagesBin)) { continue }
    [PSCustomObject]@{
        Name = $model.Name
        Path = $model.FullName
        Bytes = (Get-Item $imagesBin).Length
    }
}
$report = $report | Sort-Object Bytes -Descending
Write-Host "`nReconstructed components:" -ForegroundColor Cyan
foreach ($entry in $report) {
    Write-Host ("  {0,-8} images.bin {1,12:N0} bytes" -f $entry.Name, $entry.Bytes)
}
$best = $report | Select-Object -First 1
Write-Host "Largest component: $($best.Path)" -ForegroundColor Green

if ($SfmOnly) {
    Write-Host "`nSfM complete. Inspect the component count above before running dense." -ForegroundColor Yellow
    Write-Host "Re-run with -DenseOnly to continue." -ForegroundColor Yellow
    exit 0
}

$undistorted = Join-Path $dense "images"
$patchCfgPath = Join-Path $dense "stereo\patch-match.cfg"
if ((Test-Path $patchCfgPath) -and (Test-Path $undistorted) -and
    @(Get-ChildItem -LiteralPath $undistorted -File -ErrorAction SilentlyContinue).Count -gt 0) {
    Write-Host "4/6 image_undistorter: REUSED existing dense workspace at $dense" -ForegroundColor Yellow
}
else {
    Invoke-Colmap "4/6 image_undistorter (max $MaxImageSize px)" @(
        "image_undistorter",
        "--image_path", $Images,
        "--input_path", $best.Path,
        "--output_path", $dense,
        "--output_type", "COLMAP",
        "--max_image_size", $MaxImageSize
    )
}
# patch-match.cfg alternates a reference image with its source selection;
# fusion.cfg is one image per line. Thinning the reference list drops depth maps
# without removing any image from the pool a reference can match against, so
# coverage survives while the work shrinks.
$patchCfg = Join-Path $dense "stereo\patch-match.cfg"
$fusionCfg = Join-Path $dense "stereo\fusion.cfg"
Require-Path "patch-match.cfg" $patchCfg
Require-Path "fusion.cfg" $fusionCfg
foreach ($cfg in @($patchCfg, $fusionCfg)) {
    $backup = "$cfg.full"
    if (-not (Test-Path $backup)) { Copy-Item $cfg $backup }
}
# Always restore the full lists first, so a later run at a smaller stride is not
# silently capped by an earlier run's thinning.
Copy-Item "$patchCfg.full" $patchCfg -Force
Copy-Item "$fusionCfg.full" $fusionCfg -Force

if ($DenseStride -gt 1) {
    $lines = @(Get-Content "$patchCfg.full")
    if ($lines.Count % 2 -ne 0) {
        throw "patch-match.cfg has an odd line count ($($lines.Count)); expected reference/source pairs."
    }
    $keptRefs = New-Object System.Collections.Generic.List[string]
    $keptPairs = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $lines.Count; $i += 2) {
        if ((($i / 2) % $DenseStride) -eq 0) {
            $keptRefs.Add($lines[$i])
            $keptPairs.Add($lines[$i])
            $keptPairs.Add($lines[$i + 1])
        }
    }
    Set-Content -LiteralPath $patchCfg -Value $keptPairs -Encoding ascii
    Set-Content -LiteralPath $fusionCfg -Value $keptRefs -Encoding ascii
    Write-Host ("Dense reference images: {0} -> {1} at stride {2} (full resolution kept)" -f `
        ($lines.Count / 2), $keptRefs.Count, $DenseStride) -ForegroundColor Green
}
else {
    $refCount = (@(Get-Content $patchCfg).Count / 2)
    Write-Host "Dense reference images: $refCount (stride 1, every frame)" -ForegroundColor Green
}

Invoke-Colmap "5/6 patch_match_stereo" @(
    "patch_match_stereo",
    "--workspace_path", $dense,
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "1",
    "--PatchMatchStereo.cache_size", $CacheSize
)
Invoke-Colmap "6/6 stereo_fusion" @(
    "stereo_fusion",
    "--workspace_path", $dense,
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", (Join-Path $dense "fused.ply"),
    "--StereoFusion.cache_size", $CacheSize,
    "--StereoFusion.min_num_pixels", $MinNumPixels,
    "--StereoFusion.max_reproj_error", "3",
    "--StereoFusion.max_depth_error", "0.02",
    "--StereoFusion.max_normal_error", "25"
)

Write-Host "`nDense cloud: $(Join-Path $dense 'fused.ply')" -ForegroundColor Green
Write-Host "Depth maps retained at: $(Join-Path $dense 'stereo\depth_maps')" -ForegroundColor Green
