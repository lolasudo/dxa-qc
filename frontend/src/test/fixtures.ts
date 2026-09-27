import type { StudyResultRow } from '../types/api';

export function makeRow(overrides: Partial<StudyResultRow> = {}): StudyResultRow {
    return {
        index: 0,
        path_to_study: 'batch.zip/study/CR000001.dcm',
        study_uid: '1.2.3',
        image_uid: '1.2.3.4',
        anatomical_region: 'Поясничный отдел позвоночника',
        quality_class: 1,
        quality_prob: 0.91,
        violation_type: 'Присутствуют посторонние предметы',
        processing_status: 'Success',
        time_of_processing: 0.25,
        explanation: 'Выявлено: присутствуют посторонние предметы (84%).',
        error_reason: null,
        criteria: [
            { violation: 'Некорректная укладка', probability: 0.1, violated: false },
            { violation: 'Не выравнена ось позвоночника', probability: 0.2, violated: false },
            { violation: 'Присутствуют посторонние предметы', probability: 0.84, violated: true },
        ],
        findings: { axis_tilt_deg: -1.84, field_height_mm: 330 },
        preview_url: '/api/v1/batch/x/images/0/preview.png',
        overlay_url: '/api/v1/batch/x/images/0/overlay.png',
        ...overrides,
    };
}

export const FAILED_ROW = makeRow({
    index: 1,
    path_to_study: 'batch.zip/study/broken.dcm',
    anatomical_region: '',
    quality_class: null,
    quality_prob: null,
    violation_type: '',
    processing_status: 'Failure',
    error_reason: 'Файл не является корректным DICOM.',
    explanation: 'Файл не является корректным DICOM.',
    criteria: [],
    findings: {},
    preview_url: undefined,
    overlay_url: undefined,
});
