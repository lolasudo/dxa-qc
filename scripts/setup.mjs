// npm run setup - окружение для разработки: Python-venv с backend-зависимостями и node_modules фронтенда.
// Повторный запуск безопасен. torch: CUDA-сборка, если есть nvidia-smi, иначе CPU;
// переопределение: DXA_QC_TORCH=cpu | cu128.
import { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { BACKEND, FRONTEND, IS_WIN, NPM, VENV_PYTHON, commandOk, fail, log, run } from './lib.mjs';
import { spawnSync } from 'node:child_process';

const TORCH = '2.11.0';
const TORCHVISION = '0.26.0';

function findPython() {
    const candidates = IS_WIN
        ? [['py', ['-3.11']], ['py', ['-3.12']], ['python', []]]
        : [['python3.11', []], ['python3.12', []], ['python3', []]];
    for (const [cmd, pre] of candidates) {
        const res = spawnSync(cmd, [...pre, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], {
            encoding: 'utf8',
        });
        if (res.status === 0 && ['3.11', '3.12'].includes(res.stdout.trim())) return [cmd, pre];
    }
    return null;
}

if (!existsSync(VENV_PYTHON)) {
    const python = findPython();
    if (!python) fail('нужен Python 3.11 или 3.12 (https://www.python.org/downloads/)');
    log('dev', `создаю backend/.venv (${python.flat().join(' ')})`);
    run(python[0], [...python[1], '-m', 'venv', path.join(BACKEND, '.venv')]);
}

const flavour = process.env.DXA_QC_TORCH ?? (commandOk('nvidia-smi', []) ? 'cu128' : 'cpu');
if (!['cpu', 'cu128'].includes(flavour)) fail('DXA_QC_TORCH: cpu или cu128');
log('dev', `устанавливаю backend-зависимости (torch ${TORCH}, ${flavour}) - первый раз это несколько минут`);
const pip = (...args) => run(VENV_PYTHON, ['-m', 'pip', ...args, '--disable-pip-version-check'], { cwd: BACKEND });
pip('install', '--quiet', '--upgrade', 'pip>=26.2', 'setuptools>=81,<82');
pip(
    'install', '--quiet', '--index-url', `https://download.pytorch.org/whl/${flavour}`,
    `torch==${TORCH}`, `torchvision==${TORCHVISION}`,
);
// Точные версии из lock-файла, кроме torch (сборка зависит от платформы), как в Dockerfile.
const lock = readFileSync(path.join(BACKEND, 'requirements-dev.lock'), 'utf8')
    .split(/\r?\n/)
    .filter((l) => l && !/^(#|torch|torchvision)/.test(l));
const tmp = mkdtempSync(path.join(os.tmpdir(), 'dxa-qc-'));
const constraints = path.join(tmp, 'constraints.txt');
writeFileSync(constraints, lock.join('\n'));
try {
    pip('install', '--quiet', '-c', constraints, '-e', '.[ml,api,train,dev]');
} finally {
    rmSync(tmp, { recursive: true, force: true });
}

log('dev', 'устанавливаю зависимости фронтенда (npm ci)');
run(NPM, ['ci', '--no-fund', '--no-audit'], { cwd: FRONTEND });
log('dev', 'готово. Запуск: npm run dev');
