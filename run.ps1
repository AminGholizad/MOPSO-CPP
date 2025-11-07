param (
    [string]$BuildType = "Debug"
)

.\build.ps1 -BuildType $BuildType
if ($LASTEXITCODE -eq 0){
.\build\bin\app.exe
}
