function toNumber(value: string | undefined, fallback: number): number {
    if (!value) return fallback;

    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

export const env = {
    apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? '/api/v1',
    pollIntervalMs: toNumber(import.meta.env.VITE_POLL_INTERVAL_MS, 3000),
    maxUploadSizeMb: toNumber(import.meta.env.VITE_MAX_UPLOAD_SIZE_MB, 2048),
} as const;