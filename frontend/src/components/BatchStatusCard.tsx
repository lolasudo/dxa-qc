import { Alert, Button, Card, Progress, Space, Steps, Typography } from 'antd';
import { DownloadOutlined, SyncOutlined } from '@ant-design/icons';
import type { BatchStatusResponse } from '../types/api';
import { useDownloadBatchResult } from '../hooks/useBatch';
import { formatDateTime } from '../utils/format';
import { CardTitle } from './ui';

interface BatchStatusCardProps {
    status: BatchStatusResponse;
}

const STEP_INDEX = { queued: 0, running: 1, done: 2, failed: 1 } as const;

export function BatchStatusCard({ status }: BatchStatusCardProps) {
    const download = useDownloadBatchResult(status.job_id);
    const failed = status.status === 'failed';
    const done = status.status === 'done';

    return (
        <Card className="card" title={<CardTitle icon={<SyncOutlined spin={!done && !failed} />}>Ход обработки</CardTitle>}>
            <Space direction="vertical" size={20} style={{ width: '100%' }}>
                <Steps
                    current={STEP_INDEX[status.status]}
                    status={failed ? 'error' : done ? 'finish' : 'process'}
                    items={[
                        { title: 'В очереди' },
                        {
                            title: 'Анализ снимков',
                            description:
                                status.total !== null ? `${status.processed} из ${status.total}` : 'Распаковка архива',
                        },
                        { title: 'Готово' },
                    ]}
                />

                <Progress
                    percent={Math.round(status.progress)}
                    status={failed ? 'exception' : done ? 'success' : 'active'}
                />

                <Typography.Text type="secondary">
                    Файл <b>{status.source_name}</b>, загружен {formatDateTime(status.created_at)}
                    {done ? `, готов ${formatDateTime(status.updated_at)}` : ''}
                </Typography.Text>

                {failed ? (
                    <Alert type="error" showIcon message="Обработка не выполнена" description={status.error} />
                ) : null}

                {done ? (
                    <Space wrap>
                        <Button
                            type="primary"
                            icon={<DownloadOutlined />}
                            loading={download.isPending}
                            onClick={() => download.mutate()}
                        >
                            Скачать отчёт (ZIP)
                        </Button>
                        <Typography.Text type="secondary">
                            Таблица XLSX/CSV, журнал ошибок, заключения и визуализации
                        </Typography.Text>
                    </Space>
                ) : null}

                {download.isError ? (
                    <Alert type="error" showIcon message="Не удалось скачать отчёт" description={download.error.message} />
                ) : null}
            </Space>
        </Card>
    );
}
