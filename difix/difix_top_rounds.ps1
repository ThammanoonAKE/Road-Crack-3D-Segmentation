# One road hole's top-view Difix pass: three rounds of (render views from above the hole, difix_road_topviews.py
# -> Difix -> 2000 steps) continuing from the hole's Difix fine-tune (ft_r3.pt, the web's hole_<n>), keeping
# its side views in the mix. Then score against ft_r3 on the held-out photos, sheets from above, and build
# web\public\splats\hole_<n>_top.ksplat; the web's hole_<n> stays as it is. Stages with outputs are skipped.
param([Parameter(Mandatory = $true)][string]$Clip, [Parameter(Mandatory = $true)][int]$Hole)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot; Set-Location $root  # the repository
$gs = "$root\.venv-gsplat311\Scripts\python.exe"; $dx = "$root\.venv-difix\Scripts\python.exe"
$data = if ($env:ROAD_CRACK_DATA) { $env:ROAD_CRACK_DATA } else { "$root\data" }  # the work folders; clips_to_splats.ps1 -Work is $data\road_crack
$tests = $PSScriptRoot; $w = "$data\difix\road_$Clip"
$src = "$data\difix\Difix3D\src"; $scene = "$data\road_crack\$Clip\scene"
$env:PYTHONUNBUFFERED = "1"
function Stamp([string]$s) { Write-Output "== hole $Hole top $s $(Get-Date -Format HH:mm)" }
function Quiet { process { $l = "$_"; if ($l -notmatch "it/s\]|s/it\]|Warning|warn\(|^\s*$|RemoteException|^\[|Scene scale|Model initialized") { $l } } }

$base = "$w\ft_r3.pt"
$ckpt = $base
$dirs = @("$w\round1", "$w\round2", "$w\round3")
foreach ($r in 1..3) {
    $d = "$w\top$r"; $dirs += $d; $next = "$w\top_r$r.pt"
    if (-not (Test-Path $next)) {
        if (-not (Test-Path "$d\render")) { Stamp "round $r render"; & $gs -I "$tests\difix_road_topviews.py" $Clip $ckpt $r $d 2>&1 | Quiet }
        $n = (Get-ChildItem "$d\render" -Filter *.png).Count
        if ((Get-ChildItem "$d\difix_ref" -Filter *.png -ErrorAction SilentlyContinue).Count -lt $n) {
            Stamp "round $r Difix"; & $dx -I "$tests\difix_fix.py" $src $d 2>&1 | Quiet | Select-String "repaired|Error"
        }
        Stamp "round $r fine-tune"; & $gs -I "$tests\difix_road.py" train $Clip $ckpt 2000 $next @dirs 2>&1 | Quiet
        if (-not (Test-Path $next)) { Stamp "round $r failed"; exit 1 }
    }
    $ckpt = $next
}

Stamp "score on held-out photos (before = the web's Difix version)"
& $gs -I "$tests\difix_road.py" eval $Clip $base $ckpt "$w\evaltop" 2>&1 | Quiet
& $gs -I "$tests\difix_road_top.py" $Clip $base $ckpt "$w\top_after.jpg" 2>&1 | Quiet | Select-String "->|Error"

Stamp "web build"
if (-not (Test-Path "$w\top.ply")) { & $gs -I "$tests\difix_road.py" ply $Clip $ckpt "$w\top.ply" 2>&1 | Quiet | Select-String "Error" }
$clean = "$w\top_clean.ply"; $ply = "$w\hole_${Hole}_top.ply"
& $gs -I gaussian_splatting\clean_floaters.py --scene $scene --ply "$w\top.ply" --output $clean `
    --every 1 --min-views 10 --max-distance 1.5 --max-footprint 2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "removed|Traceback" }
& $gs -I gaussian_splatting\export_web_splat.py --scene $scene --ply $clean --output $ply `
    --crop 0.6 --max-height 0.2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match "Gaussians ->|crop:|Traceback" }
Push-Location web
cmd /c "node --max-old-space-size=24000 tools\make_ksplat.mjs $ply public\splats\hole_${Hole}_top.ksplat 1 2>nul"
Copy-Item "$w\hole_${Hole}_top.json" "splats\hole_${Hole}_top.json" -Force
Pop-Location
Stamp "done"
