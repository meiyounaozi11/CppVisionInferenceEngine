param(
    [string]$Model = '',
    [string]$Image = '',
    [string]$BuildDir = '',
    [int]$Repeat = 100,
    [int]$Warmup = 10,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $scriptRoot '..\models\representative\mobilenetv2-7.onnx'
}
if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = Join-Path $scriptRoot '..\assets\representative\cat_image.jpg'
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $scriptRoot '..\out\build\msvc-release\Release'
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $scriptRoot '..\benchmark_results\stage6_optimization.csv'
}
$exe = Join-Path $BuildDir 'Stage6OptimizationBenchmark.exe'
if (-not (Test-Path $exe)) { throw "Release benchmark executable not found: $exe" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
if (-not (Test-Path $Image)) { throw "Image not found: $Image" }
$outputDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Force $outputDir | Out-Null
'run_id,mode,decode_workers,inference_workers,input_capacity,result_capacity,ort_intra,ort_inter,tasks,throughput,e2e_mean,e2e_p50,e2e_p95,e2e_p99,file_read_mean,decode_mean,preprocess_mean,inference_mean,input_queue_wait_mean,result_queue_wait_mean,failed' | Set-Content -Encoding utf8 $Output
$matrix = @(
    @{ Mode='disk'; Decode=1; Infer=4; Capacity=1; Intra=1 },
    @{ Mode='disk'; Decode=2; Infer=4; Capacity=2; Intra=1 },
    @{ Mode='disk'; Decode=4; Infer=4; Capacity=2; Intra=1 },
    @{ Mode='disk'; Decode=4; Infer=4; Capacity=4; Intra=1 },
    @{ Mode='memory'; Decode=1; Infer=4; Capacity=2; Intra=1 },
    @{ Mode='memory'; Decode=2; Infer=4; Capacity=2; Intra=1 },
    @{ Mode='memory'; Decode=4; Infer=4; Capacity=2; Intra=1 },
    @{ Mode='disk'; Decode=4; Infer=2; Capacity=2; Intra=1 },
    @{ Mode='disk'; Decode=4; Infer=2; Capacity=2; Intra=2 },
    @{ Mode='disk'; Decode=4; Infer=4; Capacity=2; Intra=1 },
    @{ Mode='disk'; Decode=4; Infer=4; Capacity=2; Intra=2 }
)
foreach ($run in $matrix) {
    for ($rep = 1; $rep -le 3; $rep++) {
        & $exe --model $Model --image $Image --mode $run.Mode --decode-workers $run.Decode --inference-workers $run.Infer --input-capacity $run.Capacity --result-capacity 4 --ort-intra $run.Intra --ort-inter 1 --repeat $Repeat --warmup $Warmup --csv $Output
        if ($LASTEXITCODE -ne 0) { throw "Stage 6 benchmark failed (mode=$($run.Mode), decode=$($run.Decode), infer=$($run.Infer), intra=$($run.Intra))" }
    }
}
Write-Host "Stage 6 benchmark results written to $Output"
