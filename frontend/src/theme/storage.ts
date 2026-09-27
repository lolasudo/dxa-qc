import type { ThemeMode } from './palette';

export const THEME_STORAGE_KEY = 'dxa-qc-theme';

/** Сохранённый выбор пользователя, иначе - системная тема. */
export function initialThemeMode(): ThemeMode {
    try {
        const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
        if (saved === 'light' || saved === 'dark') return saved;
    } catch {
        // localStorage недоступен (приватный режим) - не критично
    }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
