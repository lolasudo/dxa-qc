import type { StudyResultRow } from '../types/api';

export type Verdict = 'good' | 'bad' | 'failed';

export const VERDICT_LABELS: Record<Verdict, string> = {
    good: 'Качественное',
    bad: 'Есть нарушение',
    failed: 'Не обработано',
};

export function verdictOf(row: Pick<StudyResultRow, 'processing_status' | 'quality_class'>): Verdict {
    if (row.processing_status !== 'Success') return 'failed';
    return row.quality_class === 1 ? 'bad' : 'good';
}
