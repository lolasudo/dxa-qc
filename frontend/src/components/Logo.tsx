/** Знак сервиса: медицинский крест на фирменной синей плашке, внизу - сегменты позвонков. */
export function Logo({ size = 44 }: { size?: number }) {
    return (
        <svg className="brand__mark" width={size} height={size} viewBox="0 0 40 40" role="img" aria-label="DXA QC">
            <rect width="40" height="40" rx="12" fill="var(--c-primary)" />
            <path d="M17 7.5h6v7h7v6h-7v7h-6v-7h-7v-6h7z" fill="#fff" />
            <rect x="15" y="30" width="10" height="2.4" rx="1.2" fill="#fff" opacity="0.8" />
            <rect x="16.5" y="33.8" width="7" height="2.4" rx="1.2" fill="#fff" opacity="0.55" />
        </svg>
    );
}
