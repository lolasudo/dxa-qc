import { useCallback, useMemo, useState } from 'react';
import { Alert, App, Button, Card, Col, Drawer, Row, Space, Typography } from 'antd';
import { FolderOpenOutlined, PlusOutlined, TableOutlined } from '@ant-design/icons';
import { BatchStatusCard } from '../components/BatchStatusCard';
import { FileDropZone } from '../components/FileDropZone';
import { ResultsSummary } from '../components/ResultsSummary';
import { ResultsTable } from '../components/ResultsTable';
import { StudyDetails } from '../components/StudyDetails';
import { CardTitle } from '../components/ui';
import { useBatchResults, useBatchStatus, useSubmitBatch } from '../hooks/useBatch';
import type { StudyResultRow } from '../types/api';
import { fileName } from '../utils/format';
import { summarize } from '../utils/summary';

export function BatchPage() {
    const { message } = App.useApp();
    const [jobId, setJobId] = useState<string | null>(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [opened, setOpened] = useState<StudyResultRow | null>(null);

    const submit = useSubmitBatch(setUploadProgress);
    const statusQuery = useBatchStatus(jobId);
    const status = statusQuery.data;
    const resultsQuery = useBatchResults(jobId, status?.status === 'done');
    const rows = resultsQuery.data?.rows;
    const summary = useMemo(() => (rows ? summarize(rows) : null), [rows]);

    const handleSubmit = (file: File) => {
        setUploadProgress(0);
        submit.mutate(file, {
            onSuccess: (data) => {
                message.success('Файл принят, начат анализ');
                setJobId(data.job_id);
            },
            onError: (error) => message.error(error.message),
        });
    };

    const handleOpen = useCallback((row: StudyResultRow) => setOpened(row), []);

    return (
        <Space direction="vertical" size={20} style={{ width: '100%' }}>
            <Row gutter={[20, 20]}>
                <Col xs={24} lg={jobId ? 10 : 24}>
                    <Card
                        className="card"
                        title={<CardTitle icon={<FolderOpenOutlined />}>Загрузка исследований</CardTitle>}
                        extra={
                            jobId ? (
                                <Button icon={<PlusOutlined />} onClick={() => setJobId(null)} disabled={submit.isPending}>
                                    Новая загрузка
                                </Button>
                            ) : null
                        }
                    >
                        <FileDropZone
                            key={jobId ?? 'new'} // после отправки зона сбрасывается: нельзя случайно отправить тот же файл повторно
                            kind="zip-or-dicom"
                            busy={submit.isPending}
                            progress={uploadProgress}
                            submitLabel="Начать анализ"
                            hint="ZIP-архив с DICOM-файлами (папки исследований внутри допускаются) или отдельный DICOM-файл. Файлы анализируются на локальном сервере и никуда не передаются."
                            onSubmit={handleSubmit}
                        />
                    </Card>
                </Col>
                {jobId ? (
                    <Col xs={24} lg={14}>
                        {status ? (
                            <BatchStatusCard status={status} />
                        ) : statusQuery.isError ? (
                            <Alert type="error" showIcon message="Не удалось получить статус" description={statusQuery.error.message} />
                        ) : (
                            <Card className="card" loading />
                        )}
                    </Col>
                ) : null}
            </Row>

            {summary ? <ResultsSummary summary={summary} /> : null}

            {status?.status === 'done' ? (
                <Card className="card" title={<CardTitle icon={<TableOutlined />}>Результаты</CardTitle>}>
                    {resultsQuery.isError ? (
                        <Alert type="error" showIcon message="Не удалось загрузить результаты" description={resultsQuery.error.message} />
                    ) : (
                        <ResultsTable rows={rows} loading={resultsQuery.isPending} onOpen={handleOpen} />
                    )}
                </Card>
            ) : null}

            <Drawer
                open={Boolean(opened)}
                onClose={() => setOpened(null)}
                width="min(1180px, 100vw)"
                title={
                    opened ? (
                        <Typography.Text strong ellipsis style={{ maxWidth: '70vw' }}>
                            {fileName(opened.path_to_study)}
                        </Typography.Text>
                    ) : null
                }
                destroyOnClose
            >
                {opened ? <StudyDetails row={opened} jobId={jobId} onReviewed={setOpened} /> : null}
            </Drawer>
        </Space>
    );
}
