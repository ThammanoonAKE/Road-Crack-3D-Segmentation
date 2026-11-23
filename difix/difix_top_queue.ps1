# Top-view Difix (difix_top_rounds.ps1) for every road hole besides 7, the pilot. Skips holes whose
# web\public\splats\hole_<n>_top.ksplat exists, so after a reboot just run it again. Once a hole is built it
# makes the 1:1 top-view sheet (top_after_1to1.jpg) and drops the round-1 and round-2 checkpoints.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot  # the repository
$scratch = Split-Path $PSScriptRoot -Parent
$holes = @(@(1, '151605'), @(2, '151720'), @(3, '151928'), @(4, '152216'), @(5, '152251'), @(6, '152326'), @(8, '152618'))
foreach ($h in $holes) {
    $n, $c = $h
    $w = "C:\3D-segmentation-data\difix\road_$c"
    if (Test-Path "$root\web\public\splats\hole_${n}_top.ksplat") { Write-Output "== hole $n already done"; continue }
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\difix_top_rounds.ps1" -Clip $c -Hole $n *>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath "$root\data\difix_top_$c.log" -Append
    if (Test-Path "$root\web\public\splats\hole_${n}_top.ksplat") {
        & "$root\.venv-gsplat311\Scripts\python.exe" -I "$scratch\crop_top2.py" "$w\top_after.jpg" "$w\top_after_1to1.jpg" | Out-Null
        Remove-Item "$w\top_r1.pt", "$w\top_r2.pt" -ErrorAction SilentlyContinue
        Write-Output "== hole $n built $(Get-Date -Format HH:mm); C: free $([math]::Round((Get-PSDrive C).Free / 1GB)) GB"
    } else {
        Write-Output "== hole $n FAILED $(Get-Date -Format HH:mm), see data\difix_top_$c.log"
    }
}
Write-Output "== queue done $(Get-Date -Format HH:mm)"
