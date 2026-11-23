# Export every hole's splat to web\public\splats\hole_<n>_v2.ksplat (+ .json camera). The page loads hole_<n>,
# the Difix fine-tune of these (README), which its own scripts build; this keeps the pre-Difix build beside it.
# Source: the depth-prior retrain road_crack_<clip>_v2 when it has finished, else road_crack_<clip>.
# Each goes through clean_floaters.py, keeping only what the photos saw sharply (>= 10 views, one
# within 1.5x the usual distance, resolved within 2x the median footprint once slant is counted),
# then export_web_splat.py, cropped to 0.6 phone heights round where the photos aimed and 0.2 above
# or below the road, then make_ksplat.mjs. Skips holes whose .ksplat is newer than their source.
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent  # the repository (this file is web\tools\)
Set-Location $root
# The clips' work folders (clips_to_splats.ps1 -Work is <data>\road_crack): ROAD_CRACK_DATA, else <repository>\data
$data = if ($env:ROAD_CRACK_DATA) { $env:ROAD_CRACK_DATA } else { "$root\data" }
$ErrorActionPreference = "Continue"
$holes = @(@(1, '151605', 19999), @(2, '151720', 19999), @(3, '151928', 19999), @(4, '152216', 19999),
           @(5, '152251', 19999), @(6, '152326', 19999), @(7, '152410', 39999), @(8, '152618', 19999))
foreach ($h in $holes) {
    $n, $c, $step = $h
    $name = "hole_${n}_v2"
    $src = @("runs\road_crack_${c}_v2\ply\point_cloud_$step.ply", "runs\road_crack_$c\ply\point_cloud_19999.ply") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    $ksplat = "web\public\splats\$name.ksplat"
    if ((Test-Path $ksplat) -and (Get-Item $ksplat).LastWriteTime -gt (Get-Item $src).LastWriteTime) { Write-Output "hole ${n}: up to date"; continue }
    $scene = "$data\road_crack\$c\scene"
    $clean = "web\public\splats\${name}_clean.ply"; $ply = "web\public\splats\$name.ply"
    Remove-Item $clean, $ply -ErrorAction SilentlyContinue
    & .\.venv-gsplat311\Scripts\python.exe -I gaussian_splatting\clean_floaters.py --scene $scene --ply $src --output $clean `
        --every 1 --min-views 10 --max-distance 1.5 --max-footprint 2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "removed|Traceback" }
    & .\.venv-gsplat311\Scripts\python.exe -I gaussian_splatting\export_web_splat.py --scene $scene --ply $clean --output $ply `
        --crop 0.6 --max-height 0.2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "Gaussians ->|crop:|Traceback" }
    Push-Location web
    node --max-old-space-size=24000 tools\make_ksplat.mjs "public\splats\$name.ply" "public\splats\$name.ksplat" 1 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "splats ->|Error" }
    Pop-Location
    if (Test-Path $ksplat) { Remove-Item $clean, $ply }  # the .ksplat is what the page loads; the rest is regenerable
    Write-Output "hole ${n}: from $src"
}
