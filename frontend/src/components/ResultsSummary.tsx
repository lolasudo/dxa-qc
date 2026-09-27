import type { ReactNode } from 'react';
import { Card, Space, Tag, Typography } from 'antd';
import {
    CheckCircleOutlined,
    ExclamationCircleOutlined,
    FileSearchOutlined,
    WarningOutlined,
} from '@ant-design/icons';
import type { ResultsSummary as Summary } from '../utils/summary';
import { IconBadge } from './ui';

function Stat({ icon, tone, value, label }: { icon: ReactNode; tone: 'primary' | 'success' | 'danger' | 'warning'; value: number; label: string }) {
    return (
        <div className="stat">
            <IconBadge icon={icon} tone={tone} large />
            <div>
                <div className="stat__value">{value}</div>
                <div className="stat__label">{label}</div>
            </div>
        </div>
    );
}

export function ResultsSummary({ summary }: { summary: Summary }) {
    return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <div className="stats">
                <Stat icon={<FileSearchOutlined />} tone="primary" value={summary.total} label="Всего снимков" />
                <Stat icon={<CheckCircleOutlined />} tone="success" value={summary.good} label="Качественные" />
                <Stat icon={<WarningOutlined />} tone="danger" value={summary.withViolations} label="С нарушениями" />
                <Stat icon={<ExclamationCircleOutlined />} tone="warning" value={summary.failed} label="Не обработаны" />
            </div>
            {summary.byViolation.length ? (
                <Card className="card" size="small">
                    <Space wrap size={[8, 8]}>
                        <Typography.Text type="secondary">Найденные нарушения:</Typography.Text>
                        {summary.byViolation.map(({ violation, count }) => (
                            <Tag key={violation} color="volcano" bordered={false}>
                                {violation}: {count}
                            </Tag>
                        ))}
                    </Space>
                </Card>
            ) : null}
        </Space>
    );
}
