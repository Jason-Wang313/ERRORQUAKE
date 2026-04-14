param(
    [string]$OutDir = (Join-Path $PSScriptRoot "submission_bundle"),
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$paperDir = $PSScriptRoot
$sourceDir = Join-Path $OutDir "source"
$files = @(
    "main.tex",
    "theorems.tex",
    "checklist.tex",
    "references.bib",
    "neurips_2026.sty",
    "build_submission.ps1",
    "main.pdf"
)

if (Test-Path $OutDir) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $sourceDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $sourceDir "figs") | Out-Null

foreach ($file in $files) {
    $src = Join-Path $paperDir $file
    if (-not (Test-Path $src)) {
        throw "Missing required submission file: $src"
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $sourceDir $file)
}

Copy-Item -Path (Join-Path $paperDir "figs\*") -Destination (Join-Path $sourceDir "figs") -Recurse

$notes = @"
Clean submission bundle generated on $(Get-Date -Format o).

Included:
- main.tex
- theorems.tex
- checklist.tex
- references.bib
- neurips_2026.sty
- build_submission.ps1
- main.pdf
- figs/

Excluded on purpose:
- main.aux
- main.log
- main.out
- main.bbl
- main.blg
- self_review.md
- findings_log.md
"@
Set-Content -Path (Join-Path $OutDir "README.txt") -Value $notes -Encoding utf8

if ($Zip) {
    $zipPath = Join-Path $OutDir "ERRORQUAKE_neurips_submission_source.zip"
    if (Test-Path $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $sourceDir "*") -DestinationPath $zipPath -Force
    Write-Host ("Created source archive: {0}" -f $zipPath)
}

Write-Host ("Prepared clean submission bundle: {0}" -f $sourceDir)
