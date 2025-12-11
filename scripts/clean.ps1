Write-Host "🧹 Cleaning build artifacts..."

$project_dir=$pwd
if ($PSScriptRoot -eq $pwd){
    $project_dir = Split-Path -parent $PSScriptRoot
}

$paths = @("$project_dir\build", "$project_dir\bin", "$project_dir\external")

foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Host "🗑️  Removing $p ..."
        Remove-Item -Recurse -Force $p
    }
}
Write-Host "✅ Clean complete!"
