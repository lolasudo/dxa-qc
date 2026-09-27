import { theme, type ThemeConfig } from 'antd';
import type { Palette, ThemeMode } from './palette';

// Golos - гарнитура ParaType для российских госсервисов; встроена в сборку (@fontsource), без CDN.
export const FONT_FAMILY =
    "'Golos Text Variable', 'Golos Text', 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif";

export function buildAntdTheme(mode: ThemeMode, p: Palette): ThemeConfig {
    return {
        algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
            colorPrimary: p.primary,
            colorPrimaryHover: p.primaryHover,
            colorInfo: p.primary,
            colorSuccess: p.success,
            colorWarning: p.warning,
            colorError: p.danger,
            colorLink: p.primary,
            colorBgLayout: p.bg,
            colorBgContainer: p.surface,
            colorBgElevated: p.surface,
            colorBorder: p.border,
            colorBorderSecondary: p.border,
            colorText: p.text,
            colorTextSecondary: p.textSecondary,
            fontFamily: FONT_FAMILY,
            fontSize: 15,
            lineHeight: 1.55,
            borderRadius: 12,
            borderRadiusLG: 20,
            borderRadiusSM: 8,
            controlHeight: 42,
            controlHeightLG: 52,
            boxShadow: 'none',
            boxShadowSecondary: '0 12px 32px rgba(14, 27, 46, 0.14)',
        },
        components: {
            Layout: { headerBg: p.headerBg, bodyBg: p.bg, footerBg: p.bg, headerHeight: 72, headerPadding: '0 32px' },
            Card: { headerFontSize: 18, paddingLG: 28, headerHeight: 64 },
            Button: { fontWeight: 600, primaryShadow: 'none', defaultShadow: 'none', paddingInlineLG: 28 },
            Table: { headerBg: p.surfaceMuted, headerColor: p.textSecondary, rowHoverBg: p.primarySoft, headerBorderRadius: 12 },
            Tag: { defaultBg: p.surfaceMuted },
            Progress: { defaultColor: p.primary, remainingColor: p.surfaceMuted },
            Upload: { colorFillAlter: p.surfaceMuted },
            Steps: { iconSize: 30 },
            Drawer: { paddingLG: 28 },
        },
    };
}
