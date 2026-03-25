$ErrorActionPreference = "Stop"

function Resolve-ServiceRoot {
  param(
    [string]$RepoRoot
  )

  $candidates = @()

  if ($env:OA_GRADING_SERVICE_ROOT) {
    $candidates += $env:OA_GRADING_SERVICE_ROOT.Trim()
  }
  if ($env:QA_GRADING_SERVICE_ROOT) {
    $candidates += $env:QA_GRADING_SERVICE_ROOT.Trim()
  }

  $candidates += @(
    (Join-Path $RepoRoot "_ext\Subject-Grading"),
    (Join-Path $RepoRoot "_ext\Subject Grading"),
    (Join-Path $RepoRoot "Subject-Grading"),
    (Join-Path $RepoRoot "Subject Grading")
  )

  foreach ($candidate in $candidates) {
    if (-not $candidate) {
      continue
    }

    $resolved = [System.IO.Path]::GetFullPath($candidate)
    $requirements = Join-Path $resolved "requirements.txt"
    $pipelineDir = Join-Path $resolved "oa_main_pipeline"
    if ((Test-Path $requirements) -and (Test-Path $pipelineDir)) {
      return $resolved
    }
  }

  $searched = $candidates | Where-Object { $_ } | ForEach-Object { [System.IO.Path]::GetFullPath($_) }
  $searchedStr = $searched -join "`n - "
  throw "Subject-Grading repository not found. Searched:`n - $searchedStr`nSet OA_GRADING_SERVICE_ROOT if it lives elsewhere."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$serviceRoot = Resolve-ServiceRoot -RepoRoot $repoRoot
$python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN.Trim() } else { "python" }

Push-Location $serviceRoot
try {
  if (-not (Test-Path ".venv")) {
    & $python -m venv .venv
  }

  $venvPython = Join-Path $serviceRoot ".venv\Scripts\python.exe"
  & $venvPython -m pip install --upgrade pip
  & $venvPython -m pip install -r requirements.txt

  $port = if ($env:OA_GRADING_SERVICE_PORT) { $env:OA_GRADING_SERVICE_PORT.Trim() } elseif ($env:QA_GRADING_SERVICE_PORT) { $env:QA_GRADING_SERVICE_PORT.Trim() } else { "8001" }
  & $venvPython -m uvicorn oa_main_pipeline.api:create_app --factory --host 0.0.0.0 --port $port
}
finally {
  Pop-Location
}
