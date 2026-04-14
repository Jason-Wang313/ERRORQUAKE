$repo = Split-Path -Parent $PSScriptRoot
$queriesDir = Join-Path $repo "data\queries"
$logsDir = Join-Path $queriesDir "logs"
$manifest = Join-Path $queriesDir "manifest.json"
$log = Join-Path $logsDir "generation_watchdog.log"
$lastSeenPid = $null

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

function Get-EnvPath {
    if ($env:ERRORQUAKE_ENV_PATH) {
        return $env:ERRORQUAKE_ENV_PATH
    }

    $candidates = @(
        (Join-Path $repo ".env"),
        (Join-Path $repo "MIRROR\.env"),
        (Join-Path (Get-Location) ".env")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Write-Log {
    param([string]$Message)
    Add-Content -Path $log -Value ((Get-Date).ToUniversalTime().ToString("o") + " " + $Message)
}

function Start-GenerationResume {
    $envPath = Get-EnvPath
    if ($envPath -and (Test-Path $envPath)) {
        foreach ($line in Get-Content $envPath) {
            if ($line -match '^NVIDIA_NIM_API_KEY=(.+)$') {
                $env:NVIDIA_API_KEY = $Matches[1].Trim()
            }
        }
    }

    $proc = Start-Process `
        -FilePath python `
        -ArgumentList "scripts/run_generation.py", "--resume", "--rpm", "40" `
        -WorkingDirectory $repo `
        -PassThru
    Write-Log ("restarted generation pid=" + $proc.Id)
}

Write-Log "watchdog started"

while ($true) {
    if (Test-Path $manifest) {
        Write-Log "manifest present; watchdog exiting"
        break
    }

    $proc = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*scripts/run_generation.py*" } |
        Select-Object -First 1

    if ($null -eq $proc) {
        Write-Log "generation process missing; relaunching --resume"
        Start-GenerationResume
        $lastSeenPid = $null
        Start-Sleep -Seconds 30
        continue
    }

    if ($proc.ProcessId -ne $lastSeenPid) {
        Write-Log ("observed generation pid=" + $proc.ProcessId)
        $lastSeenPid = $proc.ProcessId
    }

    Start-Sleep -Seconds 60
}
