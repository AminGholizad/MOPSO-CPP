param(
    [string]$BuildType = "Debug",
    [string]$Compiler = "g++",
    [switch]$Tests
)

Write-Host "♻️  Performing full rebuild..."

.$PSScriptRoot\clean.ps1
.$PSScriptRoot\build.ps1 -BuildType $BuildType -Compiler $Compiler -Tests:$Tests
