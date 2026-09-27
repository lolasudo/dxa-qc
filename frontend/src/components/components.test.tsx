import { describe, expect, it, vi } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FAILED_ROW, makeRow } from '../test/fixtures';
import { renderWithProviders } from '../test/render';
import { ResultsTable } from './ResultsTable';
import { ServiceStatus } from './ServiceStatus';
import { StudyDetails } from './StudyDetails';

function jsonResponse(body: unknown): Response {
    return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } });
}

describe('StudyDetails', () => {
    it('shows verdict, explanation, criteria and measurements', () => {
        renderWithProviders(<StudyDetails row={makeRow()} />);
        expect(screen.getByText('Есть нарушение')).toBeInTheDocument();
        expect(screen.getByText(/Выявлено: присутствуют посторонние предметы/)).toBeInTheDocument();
        expect(screen.getByText('84%')).toBeInTheDocument();
        expect(screen.getByText('1,8°')).toBeInTheDocument();
        expect(screen.getByAltText('Разметка нарушений')).toHaveAttribute(
            'src',
            '/api/v1/batch/x/images/0/overlay.png',
        );
    });

    it('lets the overlay be hidden', async () => {
        renderWithProviders(<StudyDetails row={makeRow()} />);
        await userEvent.click(screen.getByRole('switch', { name: 'Показать разметку нарушений' }));
        expect(screen.queryByAltText('Разметка нарушений')).not.toBeInTheDocument();
    });

    it('explains a failed file without an image', () => {
        renderWithProviders(<StudyDetails row={FAILED_ROW} />);
        expect(screen.getByText('Не обработано')).toBeInTheDocument();
        expect(screen.getByText('Изображение недоступно')).toBeInTheDocument();
    });
});

describe('ResultsTable', () => {
    it('renders rows with verdicts and opens a study', async () => {
        const onOpen = vi.fn();
        renderWithProviders(<ResultsTable rows={[makeRow(), FAILED_ROW]} onOpen={onOpen} />);
        const row = screen.getByText('CR000001.dcm').closest('tr') as HTMLElement;
        expect(within(row).getByText('Есть нарушение')).toBeInTheDocument();
        expect(within(row).getByText('91%')).toBeInTheDocument();
        expect(screen.getByText('Файл не является корректным DICOM.')).toBeInTheDocument();
        await userEvent.click(within(row).getByRole('button', { name: /Открыть/ }));
        expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ index: 0 }));
    });

    it('filters by file name', async () => {
        renderWithProviders(<ResultsTable rows={[makeRow(), FAILED_ROW]} onOpen={() => undefined} />);
        await userEvent.type(screen.getByPlaceholderText(/Поиск/), 'broken');
        expect(screen.queryByText('CR000001.dcm')).not.toBeInTheDocument();
        expect(screen.getByText('broken.dcm')).toBeInTheDocument();
    });
});

describe('ServiceStatus', () => {
    it('shows ready when models are loaded', async () => {
        vi.spyOn(globalThis, 'fetch').mockResolvedValue(
            jsonResponse({ status: 'ok', models_ready: true, models_error: null }),
        );
        renderWithProviders(<ServiceStatus />);
        await waitFor(() => expect(screen.getByText('Сервис работает')).toBeInTheDocument());
    });

    it('shows unavailable when backend is down', async () => {
        vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
        renderWithProviders(<ServiceStatus />);
        await waitFor(() => expect(screen.getByText('Сервис недоступен')).toBeInTheDocument());
    });
});

describe('ServiceTiles', () => {
    it('switches sections by click and by arrow keys, exposing tab semantics', async () => {
        const { ServiceTiles } = await import('./ServiceTiles');
        const onChange = vi.fn();
        const { rerender } = renderWithProviders(<ServiceTiles value="batch" onChange={onChange} />);
        const batch = screen.getByRole('tab', { name: /Пакетная проверка/ });
        const study = screen.getByRole('tab', { name: /Один снимок/ });
        expect(batch).toHaveAttribute('aria-selected', 'true');
        expect(study).toHaveAttribute('tabindex', '-1');

        await userEvent.click(study);
        expect(onChange).toHaveBeenLastCalledWith('study');

        batch.focus();
        await userEvent.keyboard('{ArrowRight}');
        expect(onChange).toHaveBeenLastCalledWith('study');

        rerender(<ServiceTiles value="study" onChange={onChange} />);
        expect(screen.getByRole('tab', { name: /Один снимок/ })).toHaveAttribute('aria-selected', 'true');
    });
});
