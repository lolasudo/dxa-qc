import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
    analyzeSingleStudy,
    downloadBatchResult,
    getBatchResults,
    getBatchStatus,
    getHealth,
    submitBatch,
    submitReview,
} from '../api/batch';
import { env } from '../config';
import type { BatchResultsResponse, JobStatus, ReviewRequest, StudyResultRow } from '../types/api';
import { downloadBlob } from '../utils/download';

export function isJobActive(status?: JobStatus): boolean {
    return !status || status === 'queued' || status === 'running';
}

export function useHealth() {
    return useQuery({
        queryKey: ['health'],
        queryFn: ({ signal }) => getHealth(signal),
        retry: false,
        // Пока модели грузятся, опрашиваем чаще, чтобы статус обновился сразу.
        refetchInterval: (query) => (query.state.data?.models_ready ? 30_000 : 3_000),
        refetchIntervalInBackground: false,
    });
}

export function useSubmitBatch(onProgress?: (percent: number) => void) {
    return useMutation({
        mutationFn: (file: File) => submitBatch(file, onProgress),
    });
}

export function useBatchStatus(jobId: string | null) {
    return useQuery({
        queryKey: ['batch', jobId, 'status'],
        queryFn: ({ signal }) => getBatchStatus(jobId as string, signal),
        enabled: Boolean(jobId),
        refetchInterval: (query) => (isJobActive(query.state.data?.status) ? env.pollIntervalMs : false),
        refetchIntervalInBackground: true,
        retry: 2,
    });
}

export function useBatchResults(jobId: string | null, enabled: boolean) {
    return useQuery({
        queryKey: ['batch', jobId, 'results'],
        queryFn: ({ signal }) => getBatchResults(jobId as string, signal),
        enabled: Boolean(jobId) && enabled,
        staleTime: Infinity,
        retry: 1,
    });
}

export function useDownloadBatchResult(jobId: string | null) {
    return useMutation({
        mutationFn: async () => {
            if (!jobId) {
                throw new Error('Нет задачи для скачивания');
            }

            return downloadBatchResult(jobId);
        },
        onSuccess: (blob) => {
            const stamp = new Date().toISOString().slice(0, 19).replaceAll(':', '-');
            downloadBlob(blob, `dxa_qc_${jobId?.slice(0, 8) ?? 'result'}_${stamp}.zip`);
        },
    });
}

export function useAnalyzeStudy(onProgress?: (percent: number) => void) {
    return useMutation({
        mutationFn: (file: File) => analyzeSingleStudy(file, onProgress),
    });
}

/** Решение специалиста; строка обновляется и в кэше результатов пакета (таблица, сводка). */
export function useSubmitReview(jobId: string | null | undefined, onUpdated?: (row: StudyResultRow) => void) {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ index, body }: { index: number; body: ReviewRequest }) => {
            if (!jobId) {
                throw new Error('Нет задачи для этого снимка');
            }

            return submitReview(jobId, index, body);
        },
        onSuccess: (row) => {
            queryClient.setQueryData<BatchResultsResponse>(['batch', jobId, 'results'], (old) =>
                old ? { ...old, rows: old.rows.map((r) => (r.index === row.index ? row : r)) } : old,
            );
            onUpdated?.(row);
        },
    });
}
