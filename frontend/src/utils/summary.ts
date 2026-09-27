import type { StudyResultRow } from '../types/api';

export interface ResultsSummary {
    total: number;
    good: number;
    withViolations: number;
    failed: number;
    /** Количество по каждому типу нарушения, по убыванию. */
    byViolation: Array<{ violation: string; count: number }>;
}

export function summarize(rows: readonly StudyResultRow[]): ResultsSummary {
    const counts = new Map<string, number>();
    let good = 0;
    let withViolations = 0;
    let failed = 0;

    for (const row of rows) {
        if (row.processing_status !== 'Success') {
            failed += 1;
            continue;
        }
        if (row.quality_class === 1) withViolations += 1;
        else good += 1;
        for (const c of row.criteria) {
            if (c.violated) counts.set(c.violation, (counts.get(c.violation) ?? 0) + 1);
        }
    }

    const byViolation = [...counts.entries()]
        .map(([violation, count]) => ({ violation, count }))
        .sort((a, b) => b.count - a.count || a.violation.localeCompare(b.violation, 'ru'));

    return { total: rows.length, good, withViolations, failed, byViolation };
}
