import { Button, Input, Space, Table, Tooltip, Typography } from 'antd';
import { EyeOutlined, SearchOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useMemo, useState } from 'react';
import { ANATOMICAL_REGIONS, HIP_VIOLATIONS, SPINE_VIOLATIONS, type StudyResultRow } from '../types/api';
import { fileName, formatSeconds, splitViolations } from '../utils/format';
import { verdictOf, VERDICT_LABELS, type Verdict } from '../utils/verdict';
import { ProbabilityBar, QualityTag, ViolationTags } from './ui';

interface ResultsTableProps {
    rows?: StudyResultRow[];
    loading?: boolean;
    onOpen: (row: StudyResultRow) => void;
}

const ALL_VIOLATIONS = [...new Set([...SPINE_VIOLATIONS, ...HIP_VIOLATIONS])];

export function ResultsTable({ rows, loading, onOpen }: ResultsTableProps) {
    const [search, setSearch] = useState('');

    const filtered = useMemo(() => {
        const q = search.trim().toLowerCase();
        if (!q) return rows;
        return rows?.filter((r) => r.path_to_study.toLowerCase().includes(q) || r.study_uid.toLowerCase().includes(q));
    }, [rows, search]);

    const columns = useMemo<ColumnsType<StudyResultRow>>(
        () => [
            {
                title: 'Файл',
                dataIndex: 'path_to_study',
                key: 'file',
                width: 180,
                ellipsis: { showTitle: false },
                render: (value: string) => (
                    <Tooltip placement="topLeft" title={value}>
                        <Typography.Text strong translate="no">
                            {fileName(value)}
                        </Typography.Text>
                    </Tooltip>
                ),
            },
            {
                title: 'Область',
                dataIndex: 'anatomical_region',
                key: 'region',
                width: 180,
                filters: ANATOMICAL_REGIONS.map((region) => ({ text: region, value: region })),
                onFilter: (value, record) => record.anatomical_region === value,
                render: (value: string) => value || '-',
            },
            {
                title: 'Заключение',
                key: 'verdict',
                width: 150,
                filters: (Object.keys(VERDICT_LABELS) as Verdict[]).map((v) => ({ text: VERDICT_LABELS[v], value: v })),
                onFilter: (value, record) => verdictOf(record) === value,
                render: (_, record) => (
                    <Space direction="vertical" size={2}>
                        <QualityTag row={record} />
                        {record.review ? (
                            <Typography.Text type="secondary" className="hint">
                                {record.review.decision === 'confirmed' ? 'подтверждено' : 'исправлено'} специалистом
                            </Typography.Text>
                        ) : null}
                    </Space>
                ),
            },
            {
                title: 'Вероятность нарушения',
                dataIndex: 'quality_prob',
                key: 'prob',
                width: 170,
                defaultSortOrder: 'descend',
                sorter: (a, b) => (a.quality_prob ?? -1) - (b.quality_prob ?? -1),
                render: (value: number | null) => <ProbabilityBar value={value} />,
            },
            {
                title: 'Нарушения',
                dataIndex: 'violation_type',
                key: 'violations',
                width: 220,
                filters: ALL_VIOLATIONS.map((v) => ({ text: v, value: v })),
                onFilter: (value, record) => splitViolations(record.violation_type).includes(String(value)),
                render: (value: string, record) =>
                    record.processing_status === 'Success' ? (
                        <ViolationTags value={value} />
                    ) : (
                        <Typography.Text type="warning">{record.error_reason ?? 'Ошибка обработки'}</Typography.Text>
                    ),
            },
            {
                title: 'Время',
                dataIndex: 'time_of_processing',
                key: 'time',
                width: 80,
                sorter: (a, b) => a.time_of_processing - b.time_of_processing,
                render: (value: number) => <span className="num">{formatSeconds(value)}</span>,
            },
            {
                title: '',
                key: 'actions',
                width: 112,
                fixed: 'right',
                render: (_, record) => (
                    <Button size="small" icon={<EyeOutlined />} onClick={() => onOpen(record)}>
                        Открыть
                    </Button>
                ),
            },
        ],
        [onOpen],
    );

    return (
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Input
                allowClear
                prefix={<SearchOutlined />}
                name="results-search"
                autoComplete="off"
                spellCheck={false}
                aria-label="Поиск по имени файла или StudyInstanceUID"
                placeholder="Поиск по имени файла или StudyInstanceUID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                style={{ maxWidth: 420 }}
            />
            <Table<StudyResultRow>
                rowKey={(row) => row.index}
                columns={columns}
                dataSource={filtered}
                loading={loading}
                onRow={(record) => ({ onDoubleClick: () => onOpen(record) })}
                pagination={{
                    pageSize: 20,
                    showSizeChanger: true,
                    pageSizeOptions: ['10', '20', '50', '100'],
                    showTotal: (total) => `Снимков: ${total}`,
                }}
                scroll={{ x: 1092 }} // сумма ширин колонок: на узком экране таблица прокручивается, а не сжимает имя файла
                locale={{ emptyText: 'Нет данных для отображения' }}
            />
        </Space>
    );
}
