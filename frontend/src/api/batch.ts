import { request, requestBlob, uploadFileWithProgress } from '../lib/http';
import type {
    BatchResultsResponse,
    BatchStatusResponse,
    BatchSubmitResponse,
    HealthResponse,
    ReviewRequest,
    StudyResponse,
    StudyResultRow,
} from '../types/api';

export function submitBatch(
    file: File,
    onProgress?: (percent: number) => void,
    signal?: AbortSignal,
): Promise<BatchSubmitResponse> {
    return uploadFileWithProgress<BatchSubmitResponse>('/batch', file, onProgress, signal);
}

export function getBatchStatus(jobId: string, signal?: AbortSignal): Promise<BatchStatusResponse> {
    return request<BatchStatusResponse>(`/batch/${encodeURIComponent(jobId)}/status`, { signal });
}

export function getBatchResults(jobId: string, signal?: AbortSignal): Promise<BatchResultsResponse> {
    return request<BatchResultsResponse>(`/batch/${encodeURIComponent(jobId)}/results`, { signal });
}

/** Решение специалиста по одному снимку; возвращает обновлённую строку. */
export function submitReview(jobId: string, index: number, body: ReviewRequest): Promise<StudyResultRow> {
    return request<StudyResultRow>(`/batch/${encodeURIComponent(jobId)}/rows/${index}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
}

/** ZIP: отчёт xlsx/csv, журнал ошибок, объяснения, решения специалиста, PNG и DICOM-серии. */
export function downloadBatchResult(jobId: string): Promise<Blob> {
    return requestBlob(`/batch/${encodeURIComponent(jobId)}/result`, { timeoutMs: 120_000 });
}

/** Одно DICOM-изображение, обрабатывается синхронно (до 3 минут). */
export function analyzeSingleStudy(
    file: File,
    onProgress?: (percent: number) => void,
    signal?: AbortSignal,
): Promise<StudyResponse> {
    return uploadFileWithProgress<StudyResponse>('/study', file, onProgress, signal);
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return request<HealthResponse>('/health', { timeoutMs: 5_000, signal });
}
