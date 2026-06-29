$ErrorActionPreference = "Stop"

$TextModel = if ($env:TEXT_MODEL) { $env:TEXT_MODEL } else { "qwen2.5:7b" }
$VisionModel = if ($env:VISION_MODEL) { $env:VISION_MODEL } else { "qwen2.5vl:7b" }
$PythonVersion = if ($env:PYTHON_VERSION) { $env:PYTHON_VERSION } else { "3.12" }

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-WithWinget {
    param(
        [string]$Id,
        [string]$Name
    )

    if (-not (Test-Command winget)) {
        Write-Warning "winget is not available. Install $Name manually, then rerun this script."
        return
    }

    Write-Step "Installing $Name"
    winget install --id $Id --exact --accept-package-agreements --accept-source-agreements
}

Set-Location $PSScriptRoot

if (-not (Test-Command python)) {
    Install-WithWinget -Id "Python.Python.3.12" -Name "Python 3.12"
}

if (-not (Test-Command uv)) {
    Write-Step "Installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Test-Command pdftotext) -or -not (Test-Command pdffonts)) {
    Install-WithWinget -Id "oschwartz10612.Poppler" -Name "Poppler"
}

if (-not (Test-Command ollama)) {
    Install-WithWinget -Id "Ollama.Ollama" -Name "Ollama"
}

Write-Step "Starting Ollama"
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

Write-Step "Installing Python $PythonVersion and project dependencies"
uv python install $PythonVersion
uv sync

Write-Step "Installing spaCy French model"
uv run python -m spacy download fr_core_news_sm

Write-Step "Pulling Ollama text model: $TextModel"
ollama pull $TextModel

Write-Step "Pulling Ollama vision model: $VisionModel"
ollama pull $VisionModel

Write-Step "Installation complete"
Write-Host "Run: uv run python -m scripts.extract"
