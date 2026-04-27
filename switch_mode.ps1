param(
    [ValidateSet("status", "classic", "loocv")]
    [string]$Mode = "status"
)

<#
.SYNOPSIS
Switch between classic and LOOCV streaming modes for the Arduino sketch.

.DESCRIPTION
Both streaming implementations (streaming.cpp and streaming_loocv.cpp) are kept in the repository.
They use compile-time guards (#if STREAMING_USE_LOOCV) to ensure only one set of symbols is compiled.

This script helps you understand which mode is active and documents the compile process.

.EXAMPLE
.\switch_mode.ps1 -Mode status          # Show current mode info
.\switch_mode.ps1 -Mode classic         # Prepare instructions for classic build
.\switch_mode.ps1 -Mode loocv           # Prepare instructions for LOOCV build
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$classicFile = Join-Path $root "streaming.cpp"
$loocvFile = Join-Path $root "streaming_loocv.cpp"

function Get-ActiveMode {
    # Check which file's content is "active" by reading guards.
    if (-not (Test-Path $classicFile)) {
        return "unknown"
    }

    $content = Get-Content -Path $classicFile -Raw
    if ($content -match "^\s*#\s*if\s+!defined\s*\(\s*STREAMING_USE_LOOCV\s*\)") {
        return "classic (default)"
    }

    return "unknown"
}

function Show-Status {
    $mode = Get-ActiveMode
    Write-Host ""
    Write-Host "=== Streaming Mode Status ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Current setup: $mode"
    Write-Host ""
    Write-Host "Files present:"
    Write-Host "  streaming.cpp       : $((Test-Path $classicFile))"
    Write-Host "  streaming_loocv.cpp : $((Test-Path $loocvFile))"
    Write-Host ""
    Write-Host "Both files use compile-time guards (#if STREAMING_USE_LOOCV),"
    Write-Host "so only one set of symbols is compiled at a time."
    Write-Host ""
}

function Show-ClassicInstructions {
    Write-Host ""
    Write-Host "=== Building in Classic Mode ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Compile without the LOOCV flag:"
    Write-Host ""
    Write-Host "arduino-cli compile --fqbn arduino:mbed_nano:nano33ble ."
    Write-Host ""
    Write-Host "Or in Arduino IDE: just compile/upload normally."
    Write-Host ""
    Write-Host "This includes streaming.cpp and uses regular streaming flow."
    Write-Host ""
}

function Show-LoocvInstructions {
    Write-Host ""
    Write-Host "=== Building in LOOCV Mode ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Compile WITH the LOOCV flag:"
    Write-Host ""
    Write-Host "arduino-cli compile --fqbn arduino:mbed_nano:nano33ble \"
    Write-Host "  --build-property compiler.cpp.extra_flags=""-DSTREAMING_USE_LOOCV"" ."
    Write-Host ""
    Write-Host "Or in Arduino IDE: you would need to add the flag to your build configuration."
    Write-Host ""
    Write-Host "This activates streaming_loocv.cpp and uses LOOCV test flow."
    Write-Host "After upload, run: python python_files/loocv_coordinator.py"
    Write-Host ""
}

switch ($Mode) {
    "status" {
        Show-Status
    }
    "classic" {
        Show-ClassicInstructions
    }
    "loocv" {
        Show-LoocvInstructions
    }
}
