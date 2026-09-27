import { describe, expect, it, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ThemeToggle } from '../components/ThemeToggle';
import { renderWithProviders } from '../test/render';
import { DARK, LIGHT } from './palette';
import { initialThemeMode, THEME_STORAGE_KEY } from './storage';

describe('theme', () => {
    it('toggles light/dark, applies CSS variables and remembers the choice', async () => {
        window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
        renderWithProviders(<ThemeToggle />);
        const root = document.documentElement;
        expect(root.dataset.theme).toBe('light');
        expect(root.style.getPropertyValue('--c-bg')).toBe(LIGHT.bg);

        await userEvent.click(screen.getByRole('button', { name: 'Тёмная тема' }));
        expect(root.dataset.theme).toBe('dark');
        expect(root.style.getPropertyValue('--c-bg')).toBe(DARK.bg);
        expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
        expect(screen.getByRole('button', { name: 'Светлая тема' })).toBeInTheDocument();
    });

    it('follows the system preference when nothing is saved', () => {
        vi.spyOn(window, 'matchMedia').mockReturnValue({ matches: true } as MediaQueryList);
        expect(initialThemeMode()).toBe('dark');
    });

    it('palettes define the same tokens', () => {
        expect(Object.keys(DARK).sort()).toEqual(Object.keys(LIGHT).sort());
    });
});
