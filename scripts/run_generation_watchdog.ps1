$repo = "C:\projects\errorquake"
$manifest = "C:\projects\errorquake\data\queries\manifest.json"
$log = "C:\projects\errorquake\data\queries\logs\generation_watchdog.log"
$envPath = "C:\Users\wangz\MIRROR\.env"
$lastSeenPid = $null

function Write-Log {
    param([string]$Message)
    Add-Content -Path $log -Value ((Get-Date).ToUniversalTime().ToString("o") + " " + $Message)
}

function Start-GenerationResume {
    foreach ($line in Get-Content $envPath) {
        if ($line -match '^NVIDIA_NIM_API_KEY=(.+)$') {
            $env:NVIDIA_API_KEY = $Matches[1].Trim()
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
