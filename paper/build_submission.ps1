$ErrorActionPreference = "Stop"

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args
    )

    Write-Host ("`n==> {0} {1}" -f $Exe, ($Args -join " "))
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1} {2}" -f $LASTEXITCODE, $Exe, ($Args -join " "))
    }
}

Push-Location $paperDir
try {
    Invoke-Step -Exe "pdflatex" -Args @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
    Invoke-Step -Exe "bibtex" -Args @("main")
    Invoke-Step -Exe "pdflatex" -Args @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
    Invoke-Step -Exe "pdflatex" -Args @("-interaction=nonstopmode", "-halt-on-error", "main.tex")
    Write-Host ("`nBuilt submission PDF: {0}" -f (Join-Path $paperDir "main.pdf"))
}
finally {
    Pop-Location
}
