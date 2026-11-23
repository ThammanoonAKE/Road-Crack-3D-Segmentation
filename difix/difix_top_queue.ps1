# Top-view Difix (difix_top_rounds.ps1) for every road hole besides 7, the pilot. Skips holes whose
# web\public\splats\hole_<n>_top.ksplat exists, so after a reboot just run it again. Once a hole is built,
# its round-1 and round-2 checkpoints go (the final top_r3.pt stays).
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot  # the repository
$data = if ($env:ROAD_CRACK_DATA) { $env:ROAD_CRACK_DATA } else { "$root\data" }  # the work folders; clips_to_splats.ps1 -Work is $data\road_crack
$holes = @(@(1, '151605'), @(2, '151720'), @(3, '151928'), @(4, '152216'), @(5, '152251'), @(6, '152326'), @(8, '152618'))
foreach ($h in $holes) {
    $n, $c = $h
    $w = "$data\difix\road_$c"
    if (Test-Path "$root\web\public\splats\hole_${n}_top.ksplat") { Write-Output "== hole $n already done"; continue }
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\difix_top_rounds.ps1" -Clip $c -Hole $n *>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath "$root\data\difix_top_$c.log" -Append
    if (Test-Path "$root\web\public\splats\hole_${n}_top.ksplat") {
        Remove-Item "$w\top_r1.pt", "$w\top_r2.pt" -ErrorAction SilentlyContinue
        Write-Output "== hole $n built $(Get-Date -Format HH:mm); $([math]::Round((Get-Item $data).PSDrive.Free / 1GB)) GB free on the data drive"
    } else {
        Write-Output "== hole $n FAILED $(Get-Date -Format HH:mm), see data\difix_top_$c.log"
    }
}
Write-Output "== queue done $(Get-Date -Format HH:mm)"
