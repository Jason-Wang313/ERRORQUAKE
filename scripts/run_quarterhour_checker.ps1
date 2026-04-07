$repo = "C:\projects\errorquake"
$progressPath = "C:\projects\errorquake\data\queries\raw\generation_progress.json"
$manifestPath = "C:\projects\errorquake\data\queries\manifest.json"
$logPath = "C:\projects\errorquake\data\queries\logs\quarterhour_checker.log"
$statePath = "C:\projects\errorquake\data\queries\quarterhour_checker_state.json"
$pollSeconds = 900
$lastTotalCandidates = $null
$lastCompletedCells = $null
$checkerStartedAt = (Get-Date).ToUniversalTime().ToString("o")

function Write-Log {
    param([string]$Message)
    Add-Content -Path $logPath -Value ((Get-Date).ToUniversalTime().ToString("o") + " " + $Message)
}

function Save-State {
    param(
        [int]$TotalCandidates,
        [int]$CompletedCells
    )
    @{
        last_total_candidates = $TotalCandidates
        last_completed_cells = $CompletedCells
        started_at = $checkerStartedAt
        last_checked_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $statePath
}

function Get-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    return Get-Content $Path -Raw | ConvertFrom-Json
}

function Get-ProcessCountLike {
    param([string]$Pattern)
    return @(
        Get-CimInstance Win32_Process |
            Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like $Pattern }
    ).Count
}

function Get-HelperSummary {
    return @{
        main_generation = Get-ProcessCountLike "*run_generation.py*"
        prefetch_workers = Get-ProcessCountLike "*run_generation_prefetch.py*"
        verify_sidecars = Get-ProcessCountLike "*run_verification_sidecar.py*"
        prefetch_verifiers = Get-ProcessCountLike "*run_prefetch_verification.py*"
        watchdogs = @(
            Get-CimInstance Win32_Process |
                Where-Object { $_.Name -eq "powershell.exe" -and $_.CommandLine -like "*run_generation_watchdog.ps1*" }
        ).Count
        guardians = @(
            Get-CimInstance Win32_Process |
                Where-Object { $_.Name -eq "powershell.exe" -and $_.CommandLine -like "*run_prefetch_guardian.ps1*" }
        ).Count
    }
}

function Test-RecentErrorSignal {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    $cutoff = (Get-Date).ToUniversalTime().AddMinutes(-15)
    if ((Get-Item $Path).LastWriteTimeUtc -lt $cutoff) {
        return $false
    }
    $recent = Get-Content $Path -Tail 40
    foreach ($line in $recent) {
        if ($line -match '^(?<stamp>\d{4}-\d{2}-\d{2}T[\d:.+-]+)\s+.*(ERROR|GenerationRequestError|RuntimeError)') {
            try {
                $stamp = [datetime]::Parse($Matches.stamp).ToUniversalTime()
                if ($stamp -ge $cutoff) {
                    return $true
                }
            } catch {
                continue
            }
        }
    }
    return $false
}

Write-Log "quarterhour checker started"

while ($true) {
    if (Test-Path $manifestPath) {
        Write-Log "manifest present; checker exiting"
        break
    }

    $progress = Get-JsonFile -Path $progressPath
    $helperSummary = Get-HelperSummary

    if ($null -eq $progress) {
        Write-Log "warning: generation progress file missing"
        Start-Sleep -Seconds $pollSeconds
        continue
    }

    $totalCandidates = [int]$progress.total_candidates
    $completedCells = @($progress.completed_cells).Count
    $currentCell = if ($null -eq $progress.current_cell) { "none" } else { $progress.current_cell -join "_" }
    $hasBaseline = ($null -ne $lastTotalCandidates -and $null -ne $lastCompletedCells)
    $deltaCandidates = if ($hasBaseline) { $totalCandidates - [int]$lastTotalCandidates } else { 0 }
    $deltaCells = if ($hasBaseline) { $completedCells - [int]$lastCompletedCells } else { 0 }

    $warnings = @()
    if ($helperSummary.main_generation -lt 1) { $warnings += "main_generation_missing" }
    if ($helperSummary.verify_sidecars -lt 1) { $warnings += "verify_sidecar_missing" }
    if ($helperSummary.watchdogs -lt 1) { $warnings += "generation_watchdog_missing" }
    if ($helperSummary.guardians -lt 1) { $warnings += "prefetch_guardian_missing" }
    if ($hasBaseline -and $deltaCandidates -le 0 -and $deltaCells -le 0) { $warnings += "no_progress_in_last_15m" }
    if (Test-RecentErrorSignal -Path "C:\projects\errorquake\data\queries\logs\verify_sidecar_stderr.log") { $warnings += "verify_sidecar_error_signal" }
    if (Test-RecentErrorSignal -Path "C:\projects\errorquake\data\queries\logs\prefetch_verify_stderr.log") { $warnings += "prefetch_verify_error_signal" }

    $summary = "status total_candidates=$totalCandidates delta_candidates=$deltaCandidates completed_cells=$completedCells delta_cells=$deltaCells current_cell=$currentCell main_generation=$($helperSummary.main_generation) prefetch_workers=$($helperSummary.prefetch_workers) verify_sidecars=$($helperSummary.verify_sidecars) prefetch_verifiers=$($helperSummary.prefetch_verifiers)"
    if ($warnings.Count -gt 0) {
        Write-Log ($summary + " warnings=" + ($warnings -join ","))
    } else {
        Write-Log ($summary + " warnings=none")
    }

    $lastTotalCandidates = $totalCandidates
    $lastCompletedCells = $completedCells
    Save-State -TotalCandidates $totalCandidates -CompletedCells $completedCells

    Start-Sleep -Seconds $pollSeconds
}
