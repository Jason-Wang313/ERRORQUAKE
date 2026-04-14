$repo = Split-Path -Parent $PSScriptRoot
$queriesDir = Join-Path $repo "data\queries"
$logsDir = Join-Path $queriesDir "logs"
$progressPath = Join-Path $queriesDir "raw\generation_progress.json"
$logPath = Join-Path $logsDir "prefetch_guardian.log"
$stalledMinutes = 12
$pollSeconds = 180
$lastCandidates = $null
$lastChangeAt = Get-Date

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

function Write-Log {
    param([string]$Message)
    Add-Content -Path $logPath -Value ((Get-Date).ToUniversalTime().ToString("o") + " " + $Message)
}

function Get-MainProgress {
    if (-not (Test-Path $progressPath)) {
        return $null
    }
    return Get-Content $progressPath -Raw | ConvertFrom-Json
}

function Get-PrefetchProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -like "*run_generation_prefetch.py*"
        }
}

function Stop-LowPriorityWorker {
    $priorities = @(
        "hist5fast",
        "cultsolo",
        "techsolo",
        "geo"
    )

    foreach ($worker in $priorities) {
        $proc = Get-PrefetchProcesses | Where-Object { $_.CommandLine -like "*--worker $worker*" } | Select-Object -First 1
        if ($null -ne $proc) {
            Stop-Process -Id $proc.ProcessId -Force
            Write-Log ("stalled main run; stopped prefetch worker " + $worker + " pid=" + $proc.ProcessId)
            return $true
        }
    }
    return $false
}

Write-Log "guardian started"

while ($true) {
    $progress = Get-MainProgress
    if ($null -eq $progress) {
        Start-Sleep -Seconds $pollSeconds
        continue
    }

    $currentCandidates = [int]$progress.total_candidates
    $currentCell = $progress.current_cell

    if ($lastCandidates -eq $null -or $currentCandidates -gt $lastCandidates) {
        $lastCandidates = $currentCandidates
        $lastChangeAt = Get-Date
        Write-Log ("progress ok candidates=" + $currentCandidates + " current_cell=" + ($currentCell -join "_"))
    } else {
        $elapsed = (Get-Date) - $lastChangeAt
        if ($elapsed.TotalMinutes -ge $stalledMinutes) {
            $stopped = Stop-LowPriorityWorker
            if (-not $stopped) {
                Write-Log ("stalled main run but no low-priority workers left; candidates=" + $currentCandidates)
            }
            $lastChangeAt = Get-Date
        }
    }

    Start-Sleep -Seconds $pollSeconds
}
