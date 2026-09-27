import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { App as AntApp, ConfigProvider } from 'antd';
// ES-сборка: CJS-обёртка 'antd/locale/ru_RU' отдаёт локаль в .default, и antd молча остаётся на английском.
import ruRU from 'antd/es/locale/ru_RU';
import { buildAntdTheme } from './antdTheme';
import { ThemeContext } from './context';
import { cssVariables, PALETTES, type ThemeMode } from './palette';
import { initialThemeMode, THEME_STORAGE_KEY } from './storage';

export function ThemeProvider({ children }: { children: ReactNode }) {
    const [mode, setMode] = useState<ThemeMode>(initialThemeMode);
    const palette = PALETTES[mode];

    useEffect(() => {
        const root = document.documentElement;
        root.dataset.theme = mode;
        root.style.colorScheme = mode;
        for (const [name, value] of Object.entries(cssVariables(palette))) {
            root.style.setProperty(name, value);
        }
        try {
            window.localStorage.setItem(THEME_STORAGE_KEY, mode);
        } catch {
            // см. initialThemeMode
        }
    }, [mode, palette]);

    const value = useMemo(
        () => ({ mode, palette, setMode, toggle: () => setMode((m) => (m === 'light' ? 'dark' : 'light')) }),
        [mode, palette],
    );
    const antdTheme = useMemo(() => buildAntdTheme(mode, palette), [mode, palette]);

    return (
        <ThemeContext.Provider value={value}>
            <ConfigProvider locale={ruRU} theme={antdTheme}>
                <AntApp>{children}</AntApp>
            </ConfigProvider>
        </ThemeContext.Provider>
    );
}
