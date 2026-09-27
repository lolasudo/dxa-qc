// Единственный источник цветов интерфейса: из палитры строятся и тема antd, и CSS-переменные.
// Светлый фон, белые карточки, фирменный синий.

export type ThemeMode = 'light' | 'dark';

export interface Palette {
    primary: string;
    primaryHover: string;
    primarySoft: string;
    accent: string;
    accentSoft: string;
    success: string;
    successSoft: string;
    warning: string;
    warningSoft: string;
    danger: string;
    dangerSoft: string;
    bg: string;
    surface: string;
    surfaceMuted: string;
    border: string;
    text: string;
    textSecondary: string;
    headerBg: string;
    heroBg: string;
    heroText: string;
    heroTextSecondary: string;
    viewerBg: string;
    focus: string;
}

export const LIGHT: Palette = {
    primary: '#1F6BFF',
    primaryHover: '#1557D6',
    primarySoft: '#E6EFFF',
    accent: '#00A3B4',
    accentSoft: '#E0F5F7',
    success: '#12A05C',
    successSoft: '#E4F6EC',
    warning: '#C26A00',
    warningSoft: '#FFF1DC',
    danger: '#E5484D',
    dangerSoft: '#FDECEC',
    bg: '#EEF3F9',
    surface: '#FFFFFF',
    surfaceMuted: '#F5F8FC',
    border: '#DDE5F0',
    text: '#0E1B2E',
    textSecondary: '#56657A',
    headerBg: '#FFFFFF',
    heroBg: '#1F6BFF',
    heroText: '#FFFFFF',
    heroTextSecondary: '#DCE7FF',
    viewerBg: '#0A1220',
    focus: '#1F6BFF',
};

// Тёмная тема для затемнённых кабинетов: глубокий сине-серый вместо чёрного, баннер темнее и
// спокойнее, статусные цвета приглушены при контрасте текста не ниже 4.5:1 к карточкам.
export const DARK: Palette = {
    primary: '#4D8BFF',
    primaryHover: '#6B9FFF',
    primarySoft: '#16284A',
    accent: '#2BC0C9',
    accentSoft: '#0F3036',
    success: '#3DCB84',
    successSoft: '#12301F',
    warning: '#F2A93B',
    warningSoft: '#372A12',
    danger: '#F2666B',
    dangerSoft: '#3A1B1F',
    bg: '#0B1320',
    surface: '#131E2E',
    surfaceMuted: '#182538',
    border: '#243349',
    text: '#E7EEF7',
    textSecondary: '#9DAEC4',
    headerBg: '#0F1A29',
    heroBg: '#16305E',
    heroText: '#FFFFFF',
    heroTextSecondary: '#B9CCF0',
    viewerBg: '#050A12',
    focus: '#6B9FFF',
};

export const PALETTES: Record<ThemeMode, Palette> = {
    light: LIGHT,
    dark: DARK,
};

/** CSS-переменные для собственных стилей (всё, что не покрыто компонентами antd). */
export function cssVariables(p: Palette): Record<string, string> {
    return {
        '--c-primary': p.primary,
        '--c-primary-hover': p.primaryHover,
        '--c-primary-soft': p.primarySoft,
        '--c-accent': p.accent,
        '--c-accent-soft': p.accentSoft,
        '--c-success': p.success,
        '--c-success-soft': p.successSoft,
        '--c-warning': p.warning,
        '--c-warning-soft': p.warningSoft,
        '--c-danger': p.danger,
        '--c-danger-soft': p.dangerSoft,
        '--c-bg': p.bg,
        '--c-surface': p.surface,
        '--c-surface-muted': p.surfaceMuted,
        '--c-border': p.border,
        '--c-text': p.text,
        '--c-text-secondary': p.textSecondary,
        '--c-header-bg': p.headerBg,
        '--c-hero-bg': p.heroBg,
        '--c-hero-text': p.heroText,
        '--c-hero-text-secondary': p.heroTextSecondary,
        '--c-viewer-bg': p.viewerBg,
        '--c-focus': p.focus,
    };
}
