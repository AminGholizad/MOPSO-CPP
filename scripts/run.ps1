param(
    [string]$BuildType = "Debug",
    [string]$Compiler = "g++"
)

Write-Host "🚀 Building and running..."

$current_dir=$pwd
$project_dir=$pwd
if ($PSScriptRoot -eq $pwd){
    $project_dir = Split-Path -parent $PSScriptRoot
}

.$PSScriptRoot\build.ps1 -BuildType $BuildType -Compiler $Compiler

# Detect if project is app (Check if add_library is NOT present, implying it's an executable project)
$rootCMake = Get-Content (Join-Path -Path $project_dir -ChildPath "CMakeLists.txt") -Raw
if ($rootCMake -match "add_library") {
    Write-Host "ℹ️ Project is a library, nothing to run."
    exit 0
} else {
    # Assuming the executable name is the project name
    $exe_name = Split-Path -Leaf $project_dir
    $exe = Join-Path -Path $project_dir -ChildPath "build/bin/$exe_name.exe"
}

if (-not (Test-Path $exe)) {
    Write-Host "❌ Executable not found: $exe" -ForegroundColor Red
    exit 1
}

Write-Host "▶️ Running app..."
& $exe
