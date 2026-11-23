# Difix for every road hole still without one (hole 6 is done): difix_road_rounds.ps1 per hole, largest last.
# Skips holes whose web\public\splats\hole_<n>_difix.ksplat exists, so after a reboot just run it again. Once a
# hole is built, its round-1 and round-2 checkpoints go (the final ft_r3.pt stays).
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot  # the repository
$holes = @(@(1, '151605', 19999), @(2, '151720', 19999), @(3, '151928', 19999), @(4, '152216', 19999),
           @(5, '152251', 19999), @(8, '152618', 19999), @(7, '152410', 39999))
foreach ($h in $holes) {
    $n, $c, $step = $h
    if (Test-Path "$root\web\public\splats\hole_${n}_difix.ksplat") { Write-Output "== hole $n already done"; continue }
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\difix_road_rounds.ps1" -Clip $c -Hole $n -Step $step *>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath "$root\data\difix_road_$c.log" -Append
    if (Test-Path "$root\web\public\splats\hole_${n}_difix.ksplat") {
        Remove-Item "C:\3D-segmentation-data\difix\road_$c\ft_r1.pt", "C:\3D-segmentation-data\difix\road_$c\ft_r2.pt" -ErrorAction SilentlyContinue
        Write-Output "== hole $n built $(Get-Date -Format HH:mm); C: free $([math]::Round((Get-PSDrive C).Free / 1GB)) GB"
    } else {
        Write-Output "== hole $n FAILED $(Get-Date -Format HH:mm), see data\difix_road_$c.log"
    }
}
Write-Output "== queue done $(Get-Date -Format HH:mm)"
