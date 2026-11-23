# 4K-in-tiles pilot for one road hole (difix_road_4k.py), continuing from its top-view Difix version (top_r3.pt,
# the web's hole_<n> or about to be): a short smoke test, 12000 steps on 4K frames in 1080p tiles with
# densifying, close views from above repaired by Difix tile by tile, 3000 steps with them, then scores and
# sheets at 4K against top_r3, and web\public\splats\hole_<n>_4k.ksplat. The web's hole_<n> stays as it is.
# -WaitFor LOG waits for "== queue done" in LOG first, so the GPU is free. Stages with outputs are skipped.
param([Parameter(Mandatory = $true)][string]$Clip, [Parameter(Mandatory = $true)][int]$Hole, [string]$WaitFor = "")
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot; Set-Location $root  # the repository
$gs = "$root\.venv-gsplat311\Scripts\python.exe"; $dx = "$root\.venv-difix\Scripts\python.exe"
$data = if ($env:ROAD_CRACK_DATA) { $env:ROAD_CRACK_DATA } else { "$root\data" }  # the work folders; clips_to_splats.ps1 -Work is $data\road_crack
$tests = $PSScriptRoot
$w = "$data\difix\road_$Clip"; $scene4k = "$w\scene_4k"
$src = "$data\difix\Difix3D\src"; $scene = "$data\road_crack\$Clip\scene"
$env:PYTHONUNBUFFERED = "1"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
function Stamp([string]$s) { Write-Output "== hole $Hole 4K $s $(Get-Date -Format HH:mm)" }
function Quiet { process { $l = "$_"; if ($l -notmatch "it/s\]|s/it\]|Warning|warn\(|^\s*$|RemoteException|^\[|Scene scale|Model initialized") { $l } } }

if ($WaitFor) {
    Stamp "waiting for the queue to finish"
    while (-not (Select-String -Path $WaitFor -Pattern "== queue done" -Quiet)) { Start-Sleep -Seconds 60 }
}
if (-not (Test-Path "$scene4k\sparse\cameras.bin")) {
    Stamp "4K scene"
    & "$root\export\colmap_pipeline\colmap_bin\bin\colmap.exe" image_undistorter --image_path "$data\road_crack\$Clip\frames" `
        --input_path "$data\road_crack\$Clip\colmap\sparse\0" --output_path $scene4k --output_type COLMAP 2>&1 | Select-Object -Last 1
}
$base = "$w\top_r3.pt"
$old = @("$w\round1", "$w\round2", "$w\round3", "$w\top1", "$w\top2", "$w\top3")

if (-not (Test-Path "$w\k4_r1.pt")) {
    Stamp "smoke test"
    # 700 steps reach the splits at +200, +300 and +400 (densifying runs from +100 to 60% of the steps)
    & $gs -I "$tests\difix_road_4k.py" train $Clip $base 700 "$w\k4_smoke.pt" 1 @old 2>&1 | Quiet
    if (-not (Test-Path "$w\k4_smoke.pt")) { Stamp "smoke train FAILED"; exit 1 }
    Remove-Item "$w\k4_smoke.pt"
    $t = "$w\hr_test"
    if (-not (Test-Path "$t\cam")) { & $gs -I "$tests\difix_road_4k.py" render $Clip $base $t 2>&1 | Quiet }
    if ((Get-ChildItem "$t\difix_ref" -Filter *.png -ErrorAction SilentlyContinue).Count -lt (Get-ChildItem "$t\render" -Filter *.png).Count) {
        & $dx -I "$tests\difix_fix.py" $src $t 2>&1 | Quiet | Select-String "repaired|Error"
        & $gs -I "$tests\difix_road_4k.py" stitch $t 2>&1 | Quiet
    }
    if ((Get-ChildItem "$t\full" -Filter *.png -ErrorAction SilentlyContinue).Count -lt 1) { Stamp "smoke Difix/stitch FAILED"; exit 1 }
    Stamp "4K tiles, densifying, 12000 steps"
    & $gs -I "$tests\difix_road_4k.py" train $Clip $base 12000 "$w\k4_r1.pt" 1 @old 2>&1 | Quiet
    if (-not (Test-Path "$w\k4_r1.pt")) { Stamp "4K train FAILED"; exit 1 }
}
$hr = "$w\hr1"
if (-not (Test-Path "$w\k4_r2.pt")) {
    if (-not (Test-Path "$hr\cam")) { Stamp "close views from above, tiled"; & $gs -I "$tests\difix_road_4k.py" render $Clip "$w\k4_r1.pt" $hr 2>&1 | Quiet }
    $n = (Get-ChildItem "$hr\render" -Filter *.png).Count
    if ((Get-ChildItem "$hr\difix_ref" -Filter *.png -ErrorAction SilentlyContinue).Count -lt $n) {
        Stamp "Difix on $n tiles"; & $dx -I "$tests\difix_fix.py" $src $hr 2>&1 | Quiet | Select-String "repaired|Error"
    }
    & $gs -I "$tests\difix_road_4k.py" stitch $hr 2>&1 | Quiet
    Stamp "4K tiles + Difix tiles, 3000 steps"
    & $gs -I "$tests\difix_road_4k.py" train $Clip "$w\k4_r1.pt" 3000 "$w\k4_r2.pt" 0 @old $hr 2>&1 | Quiet
    if (-not (Test-Path "$w\k4_r2.pt")) { Stamp "Difix-tile train FAILED"; exit 1 }
}

Stamp "score at 4K (before = top_r3, the web's version)"
$env:DIFIX_ROAD_SCENE = $scene4k
& $gs -I "$tests\difix_road.py" eval $Clip $base "$w\k4_r2.pt" "$w\eval4k" 2>&1 | Quiet
& $gs -I "$tests\difix_road_top.py" $Clip $base "$w\k4_r2.pt" "$w\top4k.jpg" 2>&1 | Quiet | Select-String "->|Error"
Remove-Item Env:\DIFIX_ROAD_SCENE

Stamp "web build"
if (-not (Test-Path "$w\k4.ply")) { & $gs -I "$tests\difix_road.py" ply $Clip "$w\k4_r2.pt" "$w\k4.ply" 2>&1 | Quiet | Select-String "Error" }
$clean = "$w\k4_clean.ply"; $ply = "$w\hole_${Hole}_4k.ply"
& $gs -I gaussian_splatting\clean_floaters.py --scene $scene --ply "$w\k4.ply" --output $clean `
    --every 1 --min-views 10 --max-distance 1.5 --max-footprint 2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "removed|Traceback" }
& $gs -I gaussian_splatting\export_web_splat.py --scene $scene --ply $clean --output $ply `
    --crop 0.6 --max-height 0.2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "Gaussians ->|crop:|Traceback" }
Push-Location web
cmd /c "node --max-old-space-size=24000 tools\make_ksplat.mjs $ply public\splats\hole_${Hole}_4k.ksplat 1 2>nul"
Copy-Item "$w\hole_${Hole}_4k.json" "splats\hole_${Hole}_4k.json" -Force
Pop-Location
Stamp "done"
