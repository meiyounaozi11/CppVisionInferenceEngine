[CmdletBinding()]
param(
    [string]$ModelPath = '',
    [string]$ImagePath = ''
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Join-Path $scriptRoot '..\models\representative\mobilenetv2-7.onnx'
}
if ([string]::IsNullOrWhiteSpace($ImagePath)) {
    $ImagePath = Join-Path $scriptRoot '..\assets\representative\cat_image.jpg'
}
$modelUrl = 'https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx'
$imageUrl = 'https://commons.wikimedia.org/wiki/Special:FilePath/Cat_image.jpg'
$expectedModelSha256 = 'C1C513582D56AFCEFF8516C73804E484C81C6A830712AB6D682253F4A3CD042F'
$expectedImageSha256 = 'D91F623700391ABCDC5B73544CF0C6DBEFFED4B925F8D9438AAD93183D3FA1E'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ModelPath), (Split-Path -Parent $ImagePath) | Out-Null
Invoke-WebRequest -Uri $modelUrl -OutFile $ModelPath
Invoke-WebRequest -Uri $imageUrl -OutFile $ImagePath

$modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ModelPath).Hash.ToUpperInvariant()
$imageHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ImagePath).Hash.ToUpperInvariant()
if ($modelHash -ne $expectedModelSha256) {
    throw "MobileNetV2 SHA-256 mismatch: $modelHash"
}
if ($imageHash -ne $expectedImageSha256) {
    throw "Representative image SHA-256 mismatch: $imageHash"
}

Write-Host "Model: $ModelPath ($modelHash)"
Write-Host "Image: $ImagePath ($imageHash)"
