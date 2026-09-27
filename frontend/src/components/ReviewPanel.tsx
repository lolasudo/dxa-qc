import { useState } from 'react';
import { App, Button, Checkbox, Input, Radio, Space, Tag, Typography } from 'antd';
import { CheckOutlined, EditOutlined, FileTextOutlined, PictureOutlined, ToolOutlined } from '@ant-design/icons';
import { useSubmitReview } from '../hooks/useBatch';
import type { ReviewRequest, StudyResultRow } from '../types/api';
import { formatDateTime, splitViolations } from '../utils/format';

const COMMENT_MAX = 1000;

function Corrections({ row }: { row: StudyResultRow }) {
    const items = row.corrections ?? [];
    if (!items.length) return null;
    return (
        <div>
            <Typography.Title level={5} style={{ marginTop: 0 }}>
                <ToolOutlined /> Предлагаемая коррекция
            </Typography.Title>
            <ul className="corrections">
                {items.map((c, i) => (
                    <li key={`${c.violation}-${i}`}>
                        <Typography.Text type="secondary">{c.violation}: </Typography.Text>
                        {c.action}
                    </li>
                ))}
            </ul>
            <Typography.Text type="secondary" className="hint">
                Предложение формируется автоматически и применяется только после решения специалиста.
            </Typography.Text>
        </div>
    );
}

function ReviewSummary({ row }: { row: StudyResultRow }) {
    const review = row.review;
    if (!review) return null;
    const violations = splitViolations(review.violation_type);
    return (
        <Space direction="vertical" size={4}>
            <Space wrap>
                <Tag color={review.decision === 'confirmed' ? 'green' : 'gold'}>
                    {review.decision === 'confirmed' ? 'Подтверждено специалистом' : 'Исправлено специалистом'}
                </Tag>
                <Typography.Text type="secondary">{formatDateTime(review.reviewed_at)}</Typography.Text>
            </Space>
            <span>
                Итог: <strong>{review.quality_class === 1 ? 'есть нарушение' : 'качественное'}</strong>
                {violations.length ? ` (${violations.join('; ')})` : ''}
            </span>
            {review.comment ? <Typography.Text italic>«{review.comment}»</Typography.Text> : null}
        </Space>
    );
}

interface ReviewPanelProps {
    row: StudyResultRow;
    /** Задача, к которой относится снимок; без неё решение сохранить нельзя. */
    jobId?: string | null;
    onReviewed?: (row: StudyResultRow) => void;
}

/** Предложения по коррекции, решение специалиста и DICOM-выходы снимка. */
export function ReviewPanel({ row, jobId, onReviewed }: ReviewPanelProps) {
    const { message } = App.useApp();
    const review = useSubmitReview(jobId, onReviewed);
    const [editing, setEditing] = useState(false);
    const [qualityClass, setQualityClass] = useState<0 | 1>(row.quality_class === 0 ? 0 : 1);
    const [violations, setViolations] = useState<string[]>(splitViolations(row.violation_type));
    const [comment, setComment] = useState('');

    if (row.processing_status !== 'Success') return null;

    const send = (body: ReviewRequest) =>
        review.mutate(
            { index: row.index, body },
            {
                onSuccess: () => {
                    setEditing(false);
                    message.success('Решение сохранено, DICOM SR обновлён');
                },
                onError: (error) => message.error(error.message),
            },
        );

    const allowed = row.criteria.map((c) => c.violation);

    return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <Corrections row={row} />

            <div>
                <Typography.Title level={5} style={{ marginTop: 0 }}>
                    Решение специалиста
                </Typography.Title>
                {row.review && !editing ? (
                    <Space direction="vertical" size={10}>
                        <ReviewSummary row={row} />
                        <Button size="small" icon={<EditOutlined />} onClick={() => setEditing(true)} disabled={!jobId}>
                            Изменить решение
                        </Button>
                    </Space>
                ) : editing ? (
                    <Space direction="vertical" size={10} style={{ width: '100%' }}>
                        <Radio.Group
                            value={qualityClass}
                            onChange={(e) => {
                                const value = e.target.value as 0 | 1;
                                setQualityClass(value);
                                if (value === 0) setViolations([]);
                            }}
                            optionType="button"
                            options={[
                                { label: 'Качественное', value: 0 },
                                { label: 'Есть нарушение', value: 1 },
                            ]}
                        />
                        <Checkbox.Group
                            value={violations}
                            onChange={(value) => setViolations(value as string[])}
                            disabled={qualityClass === 0}
                            options={allowed.map((v) => ({ label: v, value: v }))}
                            className="review__violations"
                        />
                        <Input.TextArea
                            value={comment}
                            onChange={(e) => setComment(e.target.value)}
                            maxLength={COMMENT_MAX}
                            showCount
                            autoSize={{ minRows: 2, maxRows: 5 }}
                            placeholder="Комментарий (необязательно)"
                            aria-label="Комментарий специалиста"
                        />
                        <Space wrap>
                            <Button
                                type="primary"
                                loading={review.isPending}
                                onClick={() => send({ decision: 'corrected', quality_class: qualityClass, violations, comment })}
                            >
                                Сохранить исправление
                            </Button>
                            <Button onClick={() => setEditing(false)} disabled={review.isPending}>
                                Отмена
                            </Button>
                        </Space>
                    </Space>
                ) : (
                    <Space wrap>
                        <Button
                            type="primary"
                            icon={<CheckOutlined />}
                            loading={review.isPending}
                            disabled={!jobId}
                            onClick={() => send({ decision: 'confirmed', comment: '' })}
                        >
                            Подтвердить результат
                        </Button>
                        <Button icon={<EditOutlined />} disabled={!jobId || review.isPending} onClick={() => setEditing(true)}>
                            Исправить
                        </Button>
                    </Space>
                )}
            </div>

            {row.sr_url || row.sc_url ? (
                <Space wrap>
                    {row.sr_url ? (
                        <Button icon={<FileTextOutlined />} href={row.sr_url} download>
                            DICOM SR
                        </Button>
                    ) : null}
                    {row.sc_url ? (
                        <Button icon={<PictureOutlined />} href={row.sc_url} download>
                            DICOM с разметкой
                        </Button>
                    ) : null}
                </Space>
            ) : null}
        </Space>
    );
}
