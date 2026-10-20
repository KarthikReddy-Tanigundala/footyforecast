<#
    FootyForecast task runner for Windows / PowerShell (twin of the Makefile).
    Usage:  .\make.ps1 <target>          e.g.  .\make.ps1 backtest
    Targets:
      setup      venv + Python deps (editable install)
      r-setup    renv + pinned R deps (fitzRoy 1.8.0)
      extract    fitzRoy pulls -> data/raw (cached; add -Refresh to re-pull)
      refresh    in-season incremental extract (current season only)
      build      normalise raw extracts -> data/processed/matches.parquet
      tune       grid-search Elo on 2000-2016 + walk-forward re-tuning per season
      backtest   walk-forward 2017-2026 -> results/leaderboard.md
      predict    next-round predictions + predictions/latest.json
      app        run the Streamlit dashboard
      test       pytest
      lint       ruff check + format check
      report     render the R home-ground-advantage report (Quarto)
      all        build -> backtest -> predict
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'r-setup', 'extract', 'refresh', 'build', 'tune', 'backtest', 'predict',
                 'app', 'test', 'lint', 'report', 'all')]
    [string]$Target = 'all',
    [switch]$Refresh
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { $Py = 'python' }

function Find-Tool([string]$Name, [string[]]$Candidates) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($pattern in $Candidates) {
        $hit = Get-ChildItem $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw "$Name not found. Install it or add it to PATH."
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    # Windows PowerShell 5.1 turns a native tool's stderr (Python logging, R messages) into
    # terminating errors under 'Stop', so relax it for the call and rely on the exit code.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

$Rscript = { Find-Tool 'Rscript' @('C:\Program Files\R\R-*\bin\Rscript.exe') }
$Quarto  = { Find-Tool 'quarto'  @('C:\Program Files\Quarto\bin\quarto.exe',
                                   'C:\Program Files\RStudio\resources\app\bin\quarto\bin\quarto.exe') }

switch ($Target) {
    'setup'    { python -m venv .venv; Invoke-Checked $Py @('-m', 'pip', 'install', '-r', 'requirements.txt'); Invoke-Checked $Py @('-m', 'pip', 'install', '-e', '.') }
    'r-setup'  { Invoke-Checked (& $Rscript) @('R/setup_renv.R') }
    'extract'  { $a = @('R/extract_data.R'); if ($Refresh) { $a += '--refresh' }; Invoke-Checked (& $Rscript) $a }
    'refresh'  { Invoke-Checked (& $Rscript) @('R/extract_data.R', '--incremental') }
    'build'    { Invoke-Checked $Py @('-m', 'footyforecast.io') }
    'tune'     { foreach ($m in 'flat', 'venue', 'walkforward') { Invoke-Checked $Py @('-m', 'footyforecast.elo', '--mode', $m) } }
    'backtest' { Invoke-Checked $Py @('-m', 'footyforecast.evaluation.backtest') }
    'predict'  { Invoke-Checked $Py @('-m', 'footyforecast.predict') }
    'app'      { Invoke-Checked $Py @('-m', 'streamlit', 'run', 'app/streamlit_app.py') }
    'test'     { Invoke-Checked $Py @('-m', 'pytest') }
    'lint'     { Invoke-Checked $Py @('-m', 'ruff', 'check', '.'); Invoke-Checked $Py @('-m', 'ruff', 'format', '--check', '.') }
    'report'   { Invoke-Checked (& $Quarto) @('render', 'R/reports/home_ground_advantage.qmd', '--output-dir', '../../results') }
    'all'      { foreach ($t in 'build', 'backtest', 'predict') { & $PSCommandPath $t } }
}
