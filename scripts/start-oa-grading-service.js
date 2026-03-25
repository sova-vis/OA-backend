const fs = require('fs');
const path = require('path');
const { spawnSync, spawn } = require('child_process');

function runOrFail(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    ...options,
  });

  if (result.error) {
    throw result.error;
  }

  if (typeof result.status === 'number' && result.status !== 0) {
    process.exit(result.status);
  }
}

function tryRun(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    ...options,
  });
  return result.status === 0;
}

function resolveBootstrapPython() {
  if (process.env.PYTHON_BIN && process.env.PYTHON_BIN.trim()) {
    return process.env.PYTHON_BIN.trim();
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}

function looksLikeServiceRoot(candidate) {
  if (!candidate || !fs.existsSync(candidate)) {
    return false;
  }

  const requiredPaths = [
    path.join(candidate, 'requirements.txt'),
    path.join(candidate, 'oa_main_pipeline'),
  ];

  return requiredPaths.every((entry) => fs.existsSync(entry));
}

function resolveServiceRoot(repoRoot) {
  const envCandidates = [
    process.env.OA_GRADING_SERVICE_ROOT,
    process.env.QA_GRADING_SERVICE_ROOT,
  ]
    .filter(Boolean)
    .map((entry) => String(entry).trim())
    .filter(Boolean);

  const defaultCandidates = [
    path.join(repoRoot, '_ext', 'Subject-Grading'),
    path.join(repoRoot, '_ext', 'Subject Grading'),
    path.join(repoRoot, 'Subject-Grading'),
    path.join(repoRoot, 'Subject Grading'),
  ];

  const searched = [];
  for (const candidate of [...envCandidates, ...defaultCandidates]) {
    const resolved = path.resolve(candidate);
    searched.push(resolved);
    if (looksLikeServiceRoot(resolved)) {
      return { serviceRoot: resolved, searched };
    }
  }

  return { serviceRoot: null, searched };
}

const repoRoot = path.resolve(__dirname, '..', '..');
const { serviceRoot, searched } = resolveServiceRoot(repoRoot);

if (!serviceRoot) {
  console.error('Subject-Grading repository not found.');
  console.error('Searched:');
  for (const candidate of searched) {
    console.error(`  - ${candidate}`);
  }
  console.error('Set OA_GRADING_SERVICE_ROOT to the service checkout if it lives elsewhere.');
  process.exit(1);
}

const venvPython =
  process.platform === 'win32'
    ? path.join(serviceRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(serviceRoot, '.venv', 'bin', 'python');

if (!fs.existsSync(venvPython)) {
  const bootstrapPython = resolveBootstrapPython();
  console.log(`Creating Python virtual environment using ${bootstrapPython}...`);

  let created = tryRun(bootstrapPython, ['-m', 'venv', '.venv'], { cwd: serviceRoot });

  if (!created && bootstrapPython !== 'python') {
    created = tryRun('python', ['-m', 'venv', '.venv'], { cwd: serviceRoot });
  }

  if (!created) {
    console.error('Failed to create Python virtual environment for OA grading service.');
    process.exit(1);
  }
}

console.log(`Using Subject-Grading service root: ${serviceRoot}`);
console.log('Installing OA grading Python dependencies...');
runOrFail(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip'], { cwd: serviceRoot });
runOrFail(venvPython, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: serviceRoot });

const port = (process.env.OA_GRADING_SERVICE_PORT || process.env.QA_GRADING_SERVICE_PORT || '8001').trim();
console.log(`Starting OA grading service on port ${port}...`);

const child = spawn(
  venvPython,
  ['-m', 'uvicorn', 'oa_main_pipeline.api:create_app', '--factory', '--host', '0.0.0.0', '--port', port],
  {
    cwd: serviceRoot,
    stdio: 'inherit',
  }
);

child.on('exit', (code) => {
  process.exit(typeof code === 'number' ? code : 1);
});

child.on('error', (error) => {
  console.error('Failed to start OA grading service:', error);
  process.exit(1);
});
