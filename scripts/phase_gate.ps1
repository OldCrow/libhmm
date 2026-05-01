#Requires -Version 7.0
<#
.SYNOPSIS
    Phase gate: run the required correctness suite before each phase PR.

.DESCRIPTION
    Builds and runs the seven gate tests listed in the plan (Phase D3).
    Exits with code 0 on all-pass, 1 on any failure or build error.

.PARAMETER BuildDir
    Path to the CMake binary directory.  Defaults to <repo-root>/build.

.PARAMETER Config
    CMake build configuration (Release, Debug, ...).  Defaults to Release.

.PARAMETER Rebuild
    If set, rebuild all gate targets before running them.
#>
param(
    [string] $BuildDir  = "",
    [string] $Config    = "Release",
    [switch] $Rebuild
)

Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir

if (-not $BuildDir) {
    $BuildDir = Join-Path $repoRoot "build"
}

if (-not (Test-Path $BuildDir)) {
    Write-Error "Build directory not found: $BuildDir"
    Write-Error "Run cmake -S . -B build first."
    exit 1
}

# Gate tests (plan Phase D, acceptance criteria).
$gateTargets = @(
    "test_canonical_calculators",
    "test_calculator_continuous",
    "test_calculator_edge_cases",
    "test_canonical_training",
    "test_baum_welch_convergence",
    "test_fb_mode_parity",
    "test_bw_parity"
)

# ── Optional rebuild ──────────────────────────────────────────────────────────
if ($Rebuild) {
    Write-Host "Building gate targets ($Config)..." -ForegroundColor Cyan
    $buildArgs = @(
        "--build", $BuildDir,
        "--config", $Config,
        "--target"
    ) + $gateTargets
    cmake @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "PHASE GATE FAILED: build error." -ForegroundColor Red
        exit 1
    }
}

# ── Locate executables ────────────────────────────────────────────────────────
# Multi-config generators (VS, Xcode) put binaries in <build>/tests/<Config>/.
# Single-config generators (Makefiles, Ninja) put them in <build>/tests/.
$testDir = Join-Path $BuildDir "tests"
$candidates = @(
    (Join-Path $testDir $Config),
    $testDir
)

function Find-Exe {
    param([string]$name)
    foreach ($dir in $candidates) {
        $exePath = Join-Path $dir "$name.exe"
        if (Test-Path $exePath) { return $exePath }
        $exePath = Join-Path $dir $name
        if (Test-Path $exePath) { return $exePath }
    }
    return $null
}

# ── Run each gate test ────────────────────────────────────────────────────────
$results  = [ordered]@{}
$anyFail  = $false

Write-Host ""
Write-Host "Phase gate  —  $Config  —  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ("-" * 60)

foreach ($target in $gateTargets) {
    $exe = Find-Exe $target
    if (-not $exe) {
        Write-Host "  SKIP  $target  (executable not found; run with -Rebuild)" -ForegroundColor Yellow
        $results[$target] = "SKIP"
        $anyFail = $true
        continue
    }

    & $exe --gtest_color=no 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  PASS  $target" -ForegroundColor Green
        $results[$target] = "PASS"
    } else {
        Write-Host "  FAIL  $target" -ForegroundColor Red
        $results[$target] = "FAIL"
        $anyFail = $true
        # Re-run with output so the failure is visible.
        & $exe --gtest_color=no
    }
}

Write-Host ("-" * 60)

if ($anyFail) {
    Write-Host "PHASE GATE FAILED" -ForegroundColor Red
    exit 1
} else {
    Write-Host "PHASE GATE PASSED  ($($gateTargets.Count)/$($gateTargets.Count))" -ForegroundColor Green
    exit 0
}
