import type { ReactNode } from 'react';
import { Alert, Col, Progress, Row, Space, Tag, Typography } from 'antd';
import {
    CheckCircleFilled,
    CloseCircleFilled,
    ExclamationCircleFilled,
    InfoCircleOutlined,
} from '@ant-design/icons';
import type { StudyResultRow } from '../types/api';
import { useTheme } from '../theme/context';
import { fileName, formatCm, formatDegrees, formatProbability, formatSeconds } from '../utils/format';
import { VERDICT_LABELS, verdictOf, type Verdict } from '../utils/verdict';
import { ImageViewer } from './ImageViewer';
import { ReviewPanel } from './ReviewPanel';
import { IconBadge } from './ui';

const VERDICT_VIEW: Record<Verdict, { icon: ReactNode; tone: 'success' | 'danger' | 'warning' }> = {
    good: { icon: <CheckCircleFilled />, tone: 'success' },
    bad: { icon: <CloseCircleFilled />, tone: 'danger' },
    failed: { icon: <ExclamationCircleFilled />, tone: 'warning' },
};

function Fact({ label, value }: { label: string; value: ReactNode }) {
    return (
        <div className="fact">
            <div className="fact__label">{label}</div>
            <div className="fact__value">{value}</div>
        </div>
    );
}

function Criteria({ row }: { row: StudyResultRow }) {
    const { palette } = useTheme();
    if (!row.criteria.length) return null;
    return (
        <div>
            <Typography.Title level={5} style={{ marginTop: 0 }}>
                Критерии качества
            </Typography.Title>
            {row.criteria.map((c) => (
                <div className="criterion" key={c.violation}>
                    <div className="criterion__head">
                        <span>
                            {c.violated ? (
                                <CloseCircleFilled style={{ color: palette.danger, marginRight: 8 }} />
                            ) : (
                                <CheckCircleFilled style={{ color: palette.success, marginRight: 8 }} />
                            )}
                            {c.violation}
                        </span>
                        <Typography.Text type={c.violated ? 'danger' : 'secondary'} strong={c.violated}>
                            {formatProbability(c.probability)}
                        </Typography.Text>
                    </div>
                    <Progress
                        percent={Math.round((c.probability ?? 0) * 100)}
                        showInfo={false}
                        size="small"
                        strokeColor={c.violated ? palette.danger : palette.primary}
                        style={{ margin: '6px 0 0' }}
                    />
                </div>
            ))}
        </div>
    );
}

interface StudyDetailsProps {
    row: StudyResultRow;
    /** Задача снимка: нужна для решения специалиста. */
    jobId?: string | null;
    onReviewed?: (row: StudyResultRow) => void;
}

export function StudyDetails({ row, jobId, onReviewed }: StudyDetailsProps) {
    const verdict = verdictOf(row);
    const view = VERDICT_VIEW[verdict];
    const f = row.findings ?? {};

    return (
        <Row gutter={[24, 24]}>
            <Col xs={24} lg={13}>
                <ImageViewer
                    previewUrl={row.preview_url}
                    overlayUrl={row.overlay_url}
                    region={row.anatomical_region}
                />
            </Col>
            <Col xs={24} lg={11}>
                <Space direction="vertical" size={18} style={{ width: '100%' }}>
                    <div className={`verdict verdict--${verdict}`}>
                        <IconBadge icon={view.icon} tone={view.tone} large />
                        <div>
                            <div className="verdict__title">{VERDICT_LABELS[verdict]}</div>
                            <div className="verdict__subtitle">
                                {verdict === 'failed'
                                    ? 'Файл не удалось проанализировать'
                                    : `${row.projection ? `${row.anatomical_region}, ${row.projection}` : row.anatomical_region}, вероятность нарушения ${formatProbability(row.quality_prob)}`}
                            </div>
                        </div>
                    </div>

                    {row.explanation ? (
                        <Alert
                            type={verdict === 'good' ? 'success' : verdict === 'bad' ? 'error' : 'warning'}
                            icon={<InfoCircleOutlined />}
                            showIcon
                            message="Заключение"
                            description={row.explanation}
                        />
                    ) : null}

                    <Criteria row={row} />

                    <ReviewPanel key={row.index} row={row} jobId={jobId} onReviewed={onReviewed} />

                    {verdict !== 'failed' ? (
                        <div className="facts">
                            {f.axis_tilt_deg !== undefined ? (
                                <Fact label="Наклон оси позвоночника" value={formatDegrees(f.axis_tilt_deg)} />
                            ) : null}
                            {f.shaft_tilt_deg !== undefined ? (
                                <Fact label="Наклон оси бедра" value={formatDegrees(f.shaft_tilt_deg)} />
                            ) : null}
                            {f.field_width_mm !== undefined ? (
                                <Fact label="Ширина поля" value={formatCm(f.field_width_mm)} />
                            ) : null}
                            {f.field_height_mm !== undefined ? (
                                <Fact label="Высота поля" value={formatCm(f.field_height_mm)} />
                            ) : null}
                            {f.prosthesis_suspected ? (
                                <Fact label="Металл" value={<Tag color="magenta">эндопротез?</Tag>} />
                            ) : null}
                            <Fact label="Время анализа" value={formatSeconds(row.time_of_processing)} />
                        </div>
                    ) : null}

                    <Space direction="vertical" size={2}>
                        <Typography.Text type="secondary">Файл</Typography.Text>
                        <span className="mono" title={row.path_to_study} translate="no">
                            {fileName(row.path_to_study)}
                        </span>
                        {row.study_uid ? (
                            <>
                                <Typography.Text type="secondary" style={{ marginTop: 6 }}>
                                    StudyInstanceUID / SOPInstanceUID
                                </Typography.Text>
                                <span className="mono" translate="no">{row.study_uid}</span>
                                <span className="mono" translate="no">{row.image_uid}</span>
                            </>
                        ) : null}
                    </Space>
                </Space>
            </Col>
        </Row>
    );
}
