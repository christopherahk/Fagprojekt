param(
    [ValidateSet("status", "classic", "loocv")]
    [string]$Mode = "status"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$activeFile = Join-Path $root "streaming.cpp"
$classicDisabledFile = Join-Path $root "streaming_classic.cpp.disabled"
$loocvFile = Join-Path $root "streaming_loocv.cpp"
$loocvDisabledFile = Join-Path $root "streaming_loocv.cpp.disabled"

function Get-ActiveMode {
    if (-not (Test-Path $activeFile)) {
        return "none"
    }

    $head = Get-Content -Path $activeFile -TotalCount 200 -Raw
    if ($head -match "START_FOLD|GET_RESULTS|report_fold_results") {
        return "loocv"
    }

    return "classic"
}

function Rename-InRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    if (-not (Test-Path $SourcePath)) {
        throw "Source file not found: $SourcePath"
    }

    if (Test-Path $TargetPath) {
        throw "Target file already exists: $TargetPath"
    }

    Rename-Item -Path $SourcePath -NewName (Split-Path -Leaf $TargetPath)
}

function Show-Status {
    $activeMode = Get-ActiveMode
    Write-Host "Active mode: $activeMode"
    Write-Host "Files:"
    Write-Host "  streaming.cpp                : $((Test-Path $activeFile))"
    Write-Host "  streaming_classic.cpp.disabled: $((Test-Path $classicDisabledFile))"
    Write-Host "  streaming_loocv.cpp          : $((Test-Path $loocvFile))"
    Write-Host "  streaming_loocv.cpp.disabled : $((Test-Path $loocvDisabledFile))"
}

function Switch-ToLoocv {
    $activeMode = Get-ActiveMode
    if ($activeMode -eq "loocv") {
        Write-Host "Already in loocv mode."
        return
    }

    if ($activeMode -eq "classic") {
        Rename-InRoot -SourcePath $activeFile -TargetPath $classicDisabledFile
    }

    if (Test-Path $loocvFile) {
        Rename-InRoot -SourcePath $loocvFile -TargetPath $activeFile
        Write-Host "Switched to loocv mode."
        return
    }

    if (Test-Path $loocvDisabledFile) {
        Rename-InRoot -SourcePath $loocvDisabledFile -TargetPath $activeFile
        Write-Host "Switched to loocv mode."
        return
    }

    throw "No LOOCV implementation found. Expected streaming_loocv.cpp or streaming_loocv.cpp.disabled"
}

function Switch-ToClassic {
    $activeMode = Get-ActiveMode
    if ($activeMode -eq "classic") {
        Write-Host "Already in classic mode."
        return
    }

    if ($activeMode -eq "loocv") {
        Rename-InRoot -SourcePath $activeFile -TargetPath $loocvDisabledFile
    }

    if (Test-Path $classicDisabledFile) {
        Rename-InRoot -SourcePath $classicDisabledFile -TargetPath $activeFile
        Write-Host "Switched to classic mode."
        return
    }

    throw "No classic implementation found. Expected streaming_classic.cpp.disabled"
}

switch ($Mode) {
    "status" {
        Show-Status
    }
    "classic" {
        Switch-ToClassic
        Show-Status
    }
    "loocv" {
        Switch-ToLoocv
        Show-Status
    }
}
