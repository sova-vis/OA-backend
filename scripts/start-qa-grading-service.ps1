$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$serviceRoot = Join-Path $repoRoot "_ext\Subject-Grading"

if (-not (Test-Path $serviceRoot)) {
  throw "Subject-Grading repository not found at $serviceRoot. Clone it first under _ext/Subject-Grading."
}

$python = "python"

Push-Location $serviceRoot
try {
  if (-not (Test-Path ".venv")) {
    & $python -m venv .venv
  }

  $venvPython = Join-Path $serviceRoot ".venv\Scripts\python.exe"
  & $venvPython -m pip install --upgrade pip
  & $venvPython -m pip install -r requirements.txt

  # Uses port 8001 to match QA_GRADING_SERVICE_URL default in backend route.
  & $venvPython -m uvicorn oa_main_pipeline.api:create_app --factory --host 0.0.0.0 --port 8001
}
finally {
  Pop-Location
}
