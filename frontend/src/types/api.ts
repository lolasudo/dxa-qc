// Контракт с backend (backend/src/dxa_qc/api/app.py). Меняется синхронно с ним.

export const ANATOMICAL_REGIONS = [
    'Поясничный отдел позвоночника',
    'Проксимальный отдел бедра',
] as const;

export type AnatomicalRegion = (typeof ANATOMICAL_REGIONS)[number];

export const SPINE_VIOLATIONS = [
    'Некорректная укладка',
    'Не выравнена ось позвоночника',
    'Присутствуют посторонние предметы',
] as const;

export const HIP_VIOLATIONS = [
    'Некорректная укладка',
    'Некорректная область интереса',
] as const;

export type ProcessingStatus = 'Success' | 'Failure';

export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface BatchSubmitResponse {
    job_id: string;
}

export interface BatchStatusResponse {
    job_id: string;
    status: JobStatus;
    /** 0..100 */
    progress: number;
    processed: number;
    /** Неизвестно, пока архив не распакован. */
    total: number | null;
    error: string | null;
    source_name: string;
    created_at: string;
    updated_at: string;
}

export interface CriterionResult {
    violation: string;
    /** Калиброванная вероятность нарушения критерия, 0..1. */
    probability: number | null;
    violated: boolean;
}

/** Измерения, на которых основано решение (углы - градусы, размеры - мм). */
export interface Findings {
    axis_tilt_deg?: number;
    shaft_tilt_deg?: number;
    field_height_mm?: number;
    field_width_mm?: number;
    prosthesis_suspected?: boolean;
    /** Позвоночник: верхние края подвздошных костей попали в нижнюю зону поля. */
    iliac_crests_in_field?: boolean;
}

/** Предложение по коррекции укладки или поля сканирования, требует подтверждения специалистом. */
export interface Correction {
    violation: string;
    action: string;
    /** На сколько изменить - только если это надёжно измеряется по снимку. */
    amount: number | null;
    unit: string | null;
}

export type ReviewDecision = 'confirmed' | 'corrected';

/** Решение специалиста по снимку; после него DICOM SR выпускается как VERIFIED. */
export interface Review {
    decision: ReviewDecision;
    quality_class: 0 | 1;
    /** Нарушения через `;`. */
    violation_type: string;
    comment: string;
    reviewed_at: string;
}

export interface ReviewRequest {
    decision: ReviewDecision;
    /** Обязателен для `corrected`. */
    quality_class?: 0 | 1;
    violations?: string[];
    comment?: string;
}

/** Строка отчёта + поля для интерфейса. */
export interface StudyResultRow {
    index: number;
    path_to_study: string;
    study_uid: string;
    image_uid: string;
    /** Пусто, если обработка не удалась. */
    anatomical_region: AnatomicalRegion | '';
    /** 0 - качественное, 1 - есть нарушение; null при ошибке обработки. */
    quality_class: 0 | 1 | null;
    /** Вероятность нарушения, 0..1; null при ошибке обработки. */
    quality_prob: number | null;
    /** Нарушения через `;`, пусто - нарушений нет. */
    violation_type: string;
    processing_status: ProcessingStatus;
    time_of_processing: number;

    explanation?: string;
    error_reason: string | null;
    /** Проекция и сторона, например «Прямая (AP), правое бедро»; null при ошибке обработки. */
    projection?: string | null;
    criteria: CriterionResult[];
    findings: Findings;
    corrections?: Correction[];
    review?: Review | null;
    preview_url?: string;
    overlay_url?: string;
    /** DICOM SR с заключением. */
    sr_url?: string;
    /** DICOM Secondary Capture со снимком и разметкой нарушений. */
    sc_url?: string;
}

export interface StudyResponse extends StudyResultRow {
    job_id: string;
}

export interface BatchResultsResponse {
    job_id: string;
    rows: StudyResultRow[];
}

export interface HealthResponse {
    status: 'ok';
    models_ready: boolean;
    models_error: string | null;
}
