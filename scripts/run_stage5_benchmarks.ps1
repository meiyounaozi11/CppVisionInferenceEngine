[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BenchmarkExe,
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$Csv = (Join-Path (Get-Location) 'benchmark_results\stage5_representative.csv'),
    [int]$Warmup = 20,
    [int]$InferenceRepeat = 100,
    [int]$EndToEndRepeat = 20,
    [int]$Repetitions = 3
)

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Csv) {
    Remove-Item -LiteralPath $Csv -Force
}

function Invoke-Run([string]$Mode, [int]$Workers, [int]$Capacity, [int]$Intra, [int]$Repeat) {
    & $BenchmarkExe --model $Model --image $Image --mode $Mode `
        --workers $Workers --input-capacity $Capacity --result-capacity $Capacity `
        --warmup $Warmup --repeat $Repeat --intra $Intra --inter 1 --csv $Csv
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 5 benchmark failed: mode=$Mode workers=$Workers capacity=$Capacity intra=$Intra"
    }
}

foreach ($workers in @(1, 2, 4)) {
    1..$Repetitions | ForEach-Object { Invoke-Run 'inference-only' $workers 4 1 $InferenceRepeat }
}
foreach ($intra in @(1, 2, 4)) {
    1..$Repetitions | ForEach-Object { Invoke-Run 'inference-only' 1 4 $intra $InferenceRepeat }
}
foreach ($workers in @(2, 4)) {
    1..$Repetitions | ForEach-Object { Invoke-Run 'inference-only' $workers 4 2 $InferenceRepeat }
}
foreach ($capacity in @(1, 4, 8)) {
    1..$Repetitions | ForEach-Object { Invoke-Run 'inference-only' 4 $capacity 1 $InferenceRepeat }
}
foreach ($workers in @(1, 2, 4)) {
    1..$Repetitions | ForEach-Object { Invoke-Run 'end-to-end' $workers 4 1 $EndToEndRepeat }
}

Write-Host "Stage 5 aggregate results: $Csv"
