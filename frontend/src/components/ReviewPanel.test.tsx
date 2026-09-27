import { afterEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FAILED_ROW, makeRow } from '../test/fixtures';
import { renderWithProviders } from '../test/render';
import type { StudyResultRow } from '../types/api';
import { ReviewPanel } from './ReviewPanel';

const ROW = makeRow({
    projection: 'Прямая (AP)',
    corrections: [
        {
            violation: 'Присутствуют посторонние предметы',
            action: 'Снять металлические предметы одежды из зоны сканирования',
            amount: null,
            unit: null,
        },
    ],
    sr_url: '/api/v1/batch/x/dicom/0/sr.dcm',
    sc_url: '/api/v1/batch/x/dicom/0/sc.dcm',
});

function reviewed(row: StudyResultRow, decision: 'confirmed' | 'corrected'): StudyResultRow {
    return {
        ...row,
        review: {
            decision,
            quality_class: decision === 'confirmed' ? 1 : 0,
            violation_type: decision === 'confirmed' ? row.violation_type : '',
            comment: decision === 'confirmed' ? '' : 'артефакт вне ROI',
            reviewed_at: '2026-09-26T10:00:00+00:00',
        },
    };
}

function mockFetch(response: StudyResultRow) {
    const fetchMock = vi.fn(async () =>
        new Response(JSON.stringify(response), { headers: { 'content-type': 'application/json' } }),
    );
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe('ReviewPanel', () => {
    it('shows corrections and DICOM downloads', () => {
        renderWithProviders(<ReviewPanel row={ROW} jobId={'a'.repeat(32)} />);
        expect(screen.getByText('Снять металлические предметы одежды из зоны сканирования')).toBeInTheDocument();
        expect(screen.getByRole('link', { name: /DICOM SR/ })).toHaveAttribute('href', ROW.sr_url);
        expect(screen.getByRole('link', { name: /DICOM с разметкой/ })).toHaveAttribute('href', ROW.sc_url);
    });

    it('confirms the result and reports the updated row', async () => {
        const fetchMock = mockFetch(reviewed(ROW, 'confirmed'));
        const onReviewed = vi.fn();
        const jobId = 'b'.repeat(32);
        renderWithProviders(<ReviewPanel row={ROW} jobId={jobId} onReviewed={onReviewed} />);

        await userEvent.click(screen.getByRole('button', { name: /Подтвердить результат/ }));

        await waitFor(() => expect(onReviewed).toHaveBeenCalled());
        const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
        expect(url).toBe(`/api/v1/batch/${jobId}/rows/0/review`);
        expect(init.method).toBe('POST');
        expect(JSON.parse(init.body as string)).toEqual({ decision: 'confirmed', comment: '' });
    });

    it('sends a correction: good image clears the violations', async () => {
        const fetchMock = mockFetch(reviewed(ROW, 'corrected'));
        renderWithProviders(<ReviewPanel row={ROW} jobId={'c'.repeat(32)} />);

        await userEvent.click(screen.getByRole('button', { name: /Исправить/ }));
        await userEvent.click(screen.getByText('Качественное'));
        await userEvent.type(screen.getByLabelText('Комментарий специалиста'), 'артефакт вне ROI');
        await userEvent.click(screen.getByRole('button', { name: /Сохранить исправление/ }));

        await waitFor(() => expect(fetchMock).toHaveBeenCalled());
        const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
        expect(JSON.parse(init.body as string)).toEqual({
            decision: 'corrected',
            quality_class: 0,
            violations: [],
            comment: 'артефакт вне ROI',
        });
    });

    it('shows an existing decision instead of the buttons', () => {
        renderWithProviders(<ReviewPanel row={reviewed(ROW, 'corrected')} jobId={'d'.repeat(32)} />);
        expect(screen.getByText('Исправлено специалистом')).toBeInTheDocument();
        expect(screen.getByText('качественное')).toBeInTheDocument();
        expect(screen.getByText('«артефакт вне ROI»')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /Подтвердить результат/ })).not.toBeInTheDocument();
    });

    it('is not offered for a failed file', () => {
        renderWithProviders(<ReviewPanel row={FAILED_ROW} jobId={'e'.repeat(32)} />);
        expect(screen.queryByText('Решение специалиста')).not.toBeInTheDocument();
        expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });
});
