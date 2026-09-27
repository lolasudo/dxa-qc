import { describe, expect, it, vi } from 'vitest';
import { ApiError, extractMessage, request, statusMessage } from './http';

describe('http', () => {
    it('reads FastAPI detail strings and validation lists', () => {
        expect(extractMessage({ detail: 'Задача не найдена' }, 'x')).toBe('Задача не найдена');
        expect(extractMessage({ detail: [{ msg: 'bad id' }] }, 'x')).toBe('bad id');
        expect(extractMessage('plain', 'fallback')).toBe('fallback');
    });

    it('has Russian messages for common statuses', () => {
        expect(statusMessage(413)).toBe('Файл слишком большой');
        expect(statusMessage(503)).toMatch(/Внутренняя ошибка/);
    });

    it('turns HTTP errors into ApiError with the server message', async () => {
        vi.spyOn(globalThis, 'fetch').mockResolvedValue(
            new Response(JSON.stringify({ detail: 'Задача ещё выполняется' }), { status: 409 }),
        );
        await expect(request('/x')).rejects.toMatchObject({ status: 409, message: 'Задача ещё выполняется' });
    });

    it('reports an unreachable backend clearly', async () => {
        vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
        const error = await request('/health').catch((e: unknown) => e);
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).message).toMatch(/Сервер недоступен/);
    });

    it('parses JSON bodies', async () => {
        vi.spyOn(globalThis, 'fetch').mockResolvedValue(
            new Response(JSON.stringify({ status: 'ok' }), { headers: { 'content-type': 'application/json' } }),
        );
        await expect(request('/health')).resolves.toEqual({ status: 'ok' });
    });
});
