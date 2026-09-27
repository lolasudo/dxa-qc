import { createContext, useContext } from 'react';
import { LIGHT, type Palette, type ThemeMode } from './palette';

export interface ThemeContextValue {
    mode: ThemeMode;
    palette: Palette;
    setMode: (mode: ThemeMode) => void;
    toggle: () => void;
}

export const ThemeContext = createContext<ThemeContextValue>({
    mode: 'light',
    palette: LIGHT,
    setMode: () => undefined,
    toggle: () => undefined,
});

export function useTheme(): ThemeContextValue {
    return useContext(ThemeContext);
}
