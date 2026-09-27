const NUMBER = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1, minimumFractionDigits: 1 });

export function splitViolations(value?: string | null): string[] {
    if (!value) return [];

    return value
        .split(/\s*;\s*/)
        .map((item) => item.trim())
        .filter(Boolean);
}

export function formatProbability(value?: number | null): string {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return '-';
    }

    return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

export function formatSeconds(value?: number | null): string {
    if (typeof value !== 'number' || Number.isNaN(value)) {
        return '-';
    }

    if (value < 1) {
        return `${Math.round(value * 1000)} мс`;
    }

    return `${NUMBER.format(value)} с`;
}

export function formatDegrees(value?: number | null): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${NUMBER.format(Math.abs(value))}°` : '-';
}

/** Миллиметры показываем в сантиметрах: так записаны клинические пороги (3 см / 2 см). */
export function formatCm(mm?: number | null): string {
    return typeof mm === 'number' && Number.isFinite(mm) ? `${NUMBER.format(mm / 10)} см` : '-';
}

export function formatDateTime(value?: string | null): string {
    if (!value) return '-';

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return new Intl.DateTimeFormat('ru-RU', { dateStyle: 'short', timeStyle: 'medium' }).format(date);
}

/** Имя файла без пути внутри архива - для компактного отображения. */
export function fileName(path: string): string {
    const parts = path.split(/[\\/]/).filter(Boolean);
    return parts[parts.length - 1] ?? path;
}
