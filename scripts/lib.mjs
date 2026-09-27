// Общие функции для npm-скриптов (Node, без зависимостей): одинаково работают в Windows, Linux, macOS.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const BACKEND = path.join(ROOT, 'backend');
export const FRONTEND = path.join(ROOT, 'frontend');
export const IS_WIN = process.platform === 'win32';
export const VENV_PYTHON = IS_WIN
    ? path.join(BACKEND, '.venv', 'Scripts', 'python.exe')
    : path.join(BACKEND, '.venv', 'bin', 'python');
export const NPM = IS_WIN ? 'npm.cmd' : 'npm';
export const WEIGHTS = ['region_classifier.npz', 'quality_spine.npz', 'quality_hip.npz'].map((f) =>
    path.join(BACKEND, 'weights', f),
);

const COLORS = { api: '\x1b[36m', web: '\x1b[35m', dev: '\x1b[32m', err: '\x1b[31m' };
const RESET = '\x1b[0m';

export function log(tag, message) {
    const color = process.stdout.isTTY ? (COLORS[tag] ?? '') : '';
    console.log(`${color}[${tag}]${color ? RESET : ''} ${message}`);
}

export function fail(message) {
    log('err', message);
    process.exit(1);
}

/** Запуск с выводом в консоль; при ошибке - выход с понятным сообщением. */
export function run(cmd, args, options = {}) {
    const res = spawnSync(cmd, args, { stdio: 'inherit', shell: IS_WIN && cmd.endsWith('.cmd'), ...options });
    if (res.status !== 0) fail(`команда завершилась с ошибкой: ${cmd} ${args.join(' ')}`);
}

export function commandOk(cmd, args) {
    try {
        return spawnSync(cmd, args, { stdio: 'ignore' }).status === 0;
    } catch {
        return false;
    }
}

export function isReady() {
    return existsSync(VENV_PYTHON) && existsSync(path.join(FRONTEND, 'node_modules'));
}

export function missingWeights() {
    return WEIGHTS.filter((f) => !existsSync(f)).map((f) => path.relative(ROOT, f));
}

/** Дочерний процесс с префиксом [tag] у каждой строки вывода. */
export function startPrefixed(tag, cmd, args, options) {
    const child = spawn(cmd, args, {
        ...options,
        shell: IS_WIN && cmd.endsWith('.cmd'),
        stdio: ['ignore', 'pipe', 'pipe'],
    });
    for (const stream of [child.stdout, child.stderr]) {
        let buffer = '';
        stream.setEncoding('utf8');
        stream.on('data', (chunk) => {
            buffer += chunk;
            const lines = buffer.split(/\r?\n/);
            buffer = lines.pop() ?? '';
            for (const line of lines) if (line.trim()) log(tag, line);
        });
    }
    return child;
}

/** Остановить процесс вместе с потомками (uvicorn --reload и vite порождают дочерние процессы). */
export function killTree(child) {
    if (!child || child.exitCode !== null) return;
    if (IS_WIN) {
        spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
    } else {
        child.kill('SIGTERM');
    }
}
