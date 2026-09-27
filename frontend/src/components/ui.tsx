import type { ReactNode } from 'react';
import { Progress, Tag } from 'antd';
import { CheckCircleFilled, CloseCircleFilled, ExclamationCircleFilled } from '@ant-design/icons';
import type { StudyResultRow } from '../types/api';
import { formatProbability, splitViolations } from '../utils/format';
import { useTheme } from '../theme/context';
import { VERDICT_LABELS, verdictOf, type Verdict } from '../utils/verdict';

type Tone = 'primary' | 'success' | 'warning' | 'danger' | 'accent';

export function IconBadge({ icon, tone = 'primary', large }: { icon: ReactNode; tone?: Tone; large?: boolean }) {
    const cls = ['icon-badge', tone !== 'primary' && `icon-badge--${tone}`, large && 'icon-badge--lg']
        .filter(Boolean)
        .join(' ');
    return (
        <span className={cls} aria-hidden>
            {icon}
        </span>
    );
}

export function CardTitle({ icon, children, tone }: { icon: ReactNode; children: ReactNode; tone?: Tone }) {
    return (
        <span className="card-title">
            <IconBadge icon={icon} tone={tone} />
            {children}
        </span>
    );
}

const VERDICT_META: Record<Verdict, { color: string; icon: ReactNode }> = {
    good: { color: 'success', icon: <CheckCircleFilled /> },
    bad: { color: 'error', icon: <CloseCircleFilled /> },
    failed: { color: 'warning', icon: <ExclamationCircleFilled /> },
};

export function QualityTag({ row }: { row: Pick<StudyResultRow, 'processing_status' | 'quality_class'> }) {
    const verdict = verdictOf(row);
    const meta = VERDICT_META[verdict];
    return (
        <Tag color={meta.color} icon={meta.icon} bordered={false}>
            {VERDICT_LABELS[verdict]}
        </Tag>
    );
}

/** Вероятность нарушения: цвет растёт от зелёного к красному. */
export function ProbabilityBar({ value }: { value: number | null | undefined }) {
    const { palette } = useTheme();
    if (typeof value !== 'number') return <span>-</span>;
    const color = value >= 0.5 ? palette.danger : value >= 0.3 ? palette.warning : palette.success;
    return (
        <span className="prob-cell">
            <Progress
                percent={Math.round(value * 100)}
                showInfo={false}
                size="small"
                strokeColor={color}
                aria-label={`Вероятность нарушения ${formatProbability(value)}`}
            />
            <span className="prob-cell__value">{formatProbability(value)}</span>
        </span>
    );
}

export function ViolationTags({ value }: { value: string }) {
    const items = splitViolations(value);
    if (!items.length) {
        return <span style={{ color: 'var(--c-text-secondary)' }}>-</span>;
    }
    return (
        <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap' }}>
            {items.map((v) => (
                <Tag key={v} color="volcano" bordered={false} className="violation-tag">
                    {v}
                </Tag>
            ))}
        </span>
    );
}
