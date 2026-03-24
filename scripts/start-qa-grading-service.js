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

  if (process.platform === 'win32') {
    return 'python';
  }

  return 'python3';
}

const repoRoot = path.resolve(__dirname, '..', '..');
const serviceRoot = path.join(repoRoot, '_ext', 'Subject-Grading');

if (!fs.existsSync(serviceRoot)) {
  console.error(`Subject-Grading repository not found at ${serviceRoot}`);
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
    console.error('Failed to create Python virtual environment for Q/A grading service.');
    process.exit(1);
  }
}

console.log('Installing Q/A grading Python dependencies...');
runOrFail(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip'], { cwd: serviceRoot });
runOrFail(venvPython, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { cwd: serviceRoot });

const port = (process.env.QA_GRADING_SERVICE_PORT || '8001').trim();
console.log(`Starting Q/A grading service on port ${port}...`);

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
  console.error('Failed to start Q/A grading service:', error);
  process.exit(1);
});
