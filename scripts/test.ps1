param(
    [string]$BuildType = "Debug",
    [string]$Compiler = "g++"
)

Write-Host "🧪 Building and running tests..."

$current_dir=$pwd
$project_dir=$pwd
if ($PSScriptRoot -eq $pwd){
    $project_dir = Split-Path -parent $PSScriptRoot
}

.$PSScriptRoot\build.ps1 -BuildType $BuildType -Compiler $Compiler -Tests

if (-not (Test-Path "$project_dir/build")) {
    Write-Host "❌ Build directory missing" -ForegroundColor Red
    exit 1
}

Set-Location "$project_dir/build"

Write-Host "📋 Running ctest..."
ctest --output-on-failure --build-config $BuildType

Set-Location $current_dir
Write-Host "✅ Tests finished!"
