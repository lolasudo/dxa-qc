// npm run dev - backend и frontend одной командой, в режиме разработки:
//   API  http://127.0.0.1:8000 (перезапуск при изменении кода, Swagger /api/v1/docs)
//   сайт http://localhost:5173 (горячая перезагрузка; /api проксируется на API)
// При первом запуске сам вызывает npm run setup. Ctrl+C останавливает оба процесса.
import { spawnSync } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import { BACKEND, FRONTEND, NPM, ROOT, VENV_PYTHON, isReady, killTree, log, missingWeights, startPrefixed } from './lib.mjs';

if (!isReady()) {
    log('dev', 'окружение не готово - выполняю npm run setup');
    const res = spawnSync(process.execPath, [path.join(ROOT, 'scripts', 'setup.mjs')], { stdio: 'inherit' });
    if (res.status !== 0) process.exit(res.status ?? 1);
}

const missing = missingWeights();
if (missing.length) {
    log('dev', `нет весов моделей (${missing.join(', ')}): интерфейс откроется, но анализ будет недоступен.`);
    log('dev', 'Веса: распакуйте архив весов в backend/ или обучите модели (docs/TRAINING_GUIDE.md).');
}

function portFree(port, host) {
    return new Promise((resolve) => {
        const probe = net.createServer();
        probe.once('error', () => resolve(false));
        probe.once('listening', () => probe.close(() => resolve(true)));
        probe.listen(port, host);
    });
}

for (const [port, host, what] of [[8000, '127.0.0.1', 'API'], [5173, 'localhost', 'сайта']]) {
    if (!(await portFree(port, host))) {
        log('err', `порт ${port} для ${what} занят - возможно, npm run dev уже запущен в другом окне.`);
        log('err', 'Закройте его или освободите порт и повторите.');
        process.exit(1);
    }
}

const env = { ...process.env, DXA_QC_API_DOCS: process.env.DXA_QC_API_DOCS ?? '1', PYTHONUNBUFFERED: '1' };
const api = startPrefixed('api', VENV_PYTHON, ['-m', 'dxa_qc.api', '--port', '8000', '--reload'], { cwd: BACKEND, env });
const web = startPrefixed('web', NPM, ['run', 'dev', '--', '--host', 'localhost', '--port', '5173', '--strictPort'], {
    cwd: FRONTEND,
    env,
});
log('dev', 'сайт: http://localhost:5173   API: http://127.0.0.1:8000/api/v1/docs   остановка: Ctrl+C');

let stopping = false;
function stop(code = 0) {
    if (stopping) return;
    stopping = true;
    killTree(api);
    killTree(web);
    process.exit(code);
}

for (const [name, child] of [['api', api], ['web', web]]) {
    child.on('exit', (code) => {
        if (stopping) return;
        log('err', `${name} остановился (код ${code}) - останавливаю остальное`);
        stop(code || 1);
    });
}
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) process.on(signal, () => stop(0));
