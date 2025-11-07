param (
    [string]$BuildType = "Debug"
)

if (-not (Test-Path "build")) {
    New-Item -Path "build" -ItemType Directory | Out-Null
}
Set-Location -Path "build"
cmake .. "-DCMAKE_BUILD_TYPE=$BuildType"
cmake --build .
Set-Location -Path ".."
