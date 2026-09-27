// Сообщение после npm start: адрес сервиса с учётом DXA_QC_PORT (порт или адрес:порт).
const port = process.env.DXA_QC_PORT || '7077';
const url = port.includes(':') ? `http://${port}` : `http://localhost:${port}`;
console.log(`Сервис готов${process.argv[2] === 'gpu' ? ' (GPU)' : ''}: ${url}`);
