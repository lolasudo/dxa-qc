import { Tooltip } from 'antd';
import { useHealth } from '../hooks/useBatch';

export function ServiceStatus() {
    const { data, isError, isPending } = useHealth();

    let tone = 'warn';
    let text = 'Подключение...';
    let hint = 'Проверяем доступность сервиса';

    if (isError) {
        tone = 'down';
        text = 'Сервис недоступен';
        hint = 'Backend не отвечает. Проверьте, что он запущен';
    } else if (!isPending && data) {
        if (data.models_ready) {
            tone = 'ok';
            text = 'Сервис работает';
            hint = 'Модели загружены, можно отправлять исследования';
        } else if (data.models_error) {
            tone = 'down';
            text = 'Модели не загружены';
            hint = data.models_error;
        } else {
            text = 'Загрузка моделей...';
            hint = 'Сервис запускается, это занимает до минуты';
        }
    }

    return (
        <Tooltip title={hint} placement="bottomRight">
            <span className={`status-pill status-pill--${tone}`} role="status" aria-live="polite">
                <span className="status-pill__dot" />
                <span className="status-pill__text">{text}</span>
            </span>
        </Tooltip>
    );
}
