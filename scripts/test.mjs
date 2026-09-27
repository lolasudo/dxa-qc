// npm test - все проверки проекта: backend (ruff, pytest) и frontend (eslint, типы, vitest).
import { BACKEND, FRONTEND, NPM, VENV_PYTHON, fail, isReady, log, run } from './lib.mjs';

if (!isReady()) fail('окружение не готово: выполните npm run setup');
log('dev', 'backend: ruff');
run(VENV_PYTHON, ['-m', 'ruff', 'check', 'src', 'tests'], { cwd: BACKEND });
run(VENV_PYTHON, ['-m', 'ruff', 'format', '--check', 'src', 'tests'], { cwd: BACKEND });
log('dev', 'backend: pytest');
run(VENV_PYTHON, ['-m', 'pytest', '-q'], { cwd: BACKEND });
log('dev', 'frontend: eslint, typecheck, vitest');
run(NPM, ['run', 'lint'], { cwd: FRONTEND });
run(NPM, ['run', 'typecheck'], { cwd: FRONTEND });
run(NPM, ['test'], { cwd: FRONTEND });
log('dev', 'все проверки пройдены');
