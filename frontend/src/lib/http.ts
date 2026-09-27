import { env } from '../config';

export class ApiError extends Error {
    readonly status: number;
    readonly payload?: unknown;

    constructor(status: number, message: string, payload?: unknown) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.payload = payload;
    }
}

export interface RequestOptions extends Omit<RequestInit, 'body'> {
    timeoutMs?: number;
    body?: BodyInit | null;
}

function buildUrl(path: string): string {
    if (path.startsWith('http://') || path.startsWith('https://')) {
        return path;
    }

    return `${env.apiBaseUrl}${path}`;
}

export function statusMessage(status: number): string {
    if (status === 413) return 'Файл слишком большой';
    if (status === 415) return 'Неподдерживаемый формат файла';
    if (status === 429) return 'Сервис занят, повторите позже';
    if (status >= 500) return `Внутренняя ошибка сервиса (HTTP ${status})`;
    return `Ошибка запроса (HTTP ${status})`;
}

/** FastAPI кладёт текст ошибки в `detail` (строка или список ошибок валидации). */
export function extractMessage(payload: unknown, fallback: string): string {
    if (payload && typeof payload === 'object' && 'detail' in payload) {
        const detail = (payload as { detail?: unknown }).detail;

        if (typeof detail === 'string') {
            return detail;
        }

        if (Array.isArray(detail) && detail.length > 0) {
            const first = detail[0] as { msg?: unknown };
            if (typeof first?.msg === 'string') {
                return first.msg;
            }
        }
    }

    return fallback;
}

async function parseErrorResponse(response: Response): Promise<ApiError> {
    let payload: unknown;
    const text = await response.text().catch(() => '');

    try {
        payload = text ? JSON.parse(text) : undefined;
    } catch {
        payload = text;
    }

    return new ApiError(response.status, extractMessage(payload, statusMessage(response.status)), payload);
}

async function fetchWithTimeout(path: string, options: RequestOptions): Promise<Response> {
    const { timeoutMs = 30_000, signal, ...init } = options;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    if (signal) {
        if (signal.aborted) {
            controller.abort();
        } else {
            signal.addEventListener('abort', () => controller.abort(), { once: true });
        }
    }

    try {
        return await fetch(buildUrl(path), { ...init, signal: controller.signal });
    } catch (error) {
        if (signal?.aborted) {
            throw error; // отмена вызывающей стороной (React Query) - не ошибка сервиса
        }
        if (controller.signal.aborted) {
            throw new ApiError(0, 'Сервер не ответил вовремя');
        }
        throw new ApiError(0, 'Сервер недоступен. Проверьте, что backend запущен');
    } finally {
        clearTimeout(timeout);
    }
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const response = await fetchWithTimeout(path, options);

    if (!response.ok) {
        throw await parseErrorResponse(response);
    }

    if (response.status === 204) {
        return undefined as T;
    }

    const contentType = response.headers.get('content-type') ?? '';

    if (contentType.includes('application/json')) {
        return (await response.json()) as T;
    }

    return (await response.text()) as unknown as T;
}

export async function requestBlob(path: string, options: RequestOptions = {}): Promise<Blob> {
    const response = await fetchWithTimeout(path, options);

    if (!response.ok) {
        throw await parseErrorResponse(response);
    }

    return response.blob();
}

/** XHR вместо fetch: только он даёт прогресс отправки файла. */
export function uploadFileWithProgress<T>(
    path: string,
    file: File,
    onProgress?: (percent: number) => void,
    signal?: AbortSignal,
): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();

        formData.append('file', file);

        xhr.open('POST', buildUrl(path));
        xhr.responseType = 'text';

        xhr.upload.onprogress = (event) => {
            if (!event.lengthComputable) return;
            onProgress?.(Math.round((event.loaded / event.total) * 100));
        };

        xhr.onload = () => {
            let payload: unknown;

            try {
                payload = xhr.responseText ? JSON.parse(xhr.responseText) : undefined;
            } catch {
                payload = xhr.responseText;
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(payload as T);
                return;
            }

            reject(new ApiError(xhr.status, extractMessage(payload, statusMessage(xhr.status)), payload));
        };

        xhr.onerror = () => reject(new ApiError(0, 'Сервер недоступен. Файл не отправлен'));
        xhr.onabort = () => reject(new ApiError(0, 'Загрузка отменена'));

        if (signal) {
            if (signal.aborted) {
                xhr.abort();
                return;
            }

            signal.addEventListener('abort', () => xhr.abort(), { once: true });
        }

        xhr.send(formData);
    });
}
