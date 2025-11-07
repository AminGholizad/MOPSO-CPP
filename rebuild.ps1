param (
    [string]$BuildType = "Debug"
)

Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
.\build.ps1 -BuildType $BuildType
