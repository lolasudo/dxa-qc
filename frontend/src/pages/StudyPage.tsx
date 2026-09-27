import { useState } from 'react';
import type { StudyResultRow } from '../types/api';
import { Alert, App, Card, Col, Empty, Row } from 'antd';
import { FileImageOutlined, MedicineBoxOutlined } from '@ant-design/icons';
import { FileDropZone } from '../components/FileDropZone';
import { StudyDetails } from '../components/StudyDetails';
import { CardTitle } from '../components/ui';
import { useAnalyzeStudy } from '../hooks/useBatch';

export function StudyPage() {
    const { message } = App.useApp();
    const [progress, setProgress] = useState(0);
    const analyze = useAnalyzeStudy(setProgress);
    // Строка после решения специалиста; сбрасывается при новом анализе.
    const [reviewed, setReviewed] = useState<StudyResultRow | null>(null);

    const handleSubmit = (file: File) => {
        setProgress(0);
        setReviewed(null);
        analyze.mutate(file, { onError: (error) => message.error(error.message) });
    };

    return (
        <Row gutter={[20, 20]}>
            <Col xs={24} xl={7}>
                <Card className="card" title={<CardTitle icon={<FileImageOutlined />}>Снимок</CardTitle>}>
                    <FileDropZone
                        kind="dicom"
                        busy={analyze.isPending}
                        progress={progress}
                        submitLabel="Проверить снимок"
                        hint="Один DICOM-файл поясничного отдела позвоночника или проксимального отдела бедра. Область определяется автоматически."
                        onSubmit={handleSubmit}
                    />
                </Card>
            </Col>
            <Col xs={24} xl={17}>
                <Card className="card" title={<CardTitle icon={<MedicineBoxOutlined />} tone="accent">Результат проверки</CardTitle>}>
                    {analyze.isError ? (
                        <Alert type="error" showIcon message="Снимок не проанализирован" description={analyze.error.message} />
                    ) : analyze.data ? (
                        <StudyDetails
                            row={reviewed ?? analyze.data}
                            jobId={analyze.data.job_id}
                            onReviewed={setReviewed}
                        />
                    ) : (
                        <Empty
                            image={Empty.PRESENTED_IMAGE_SIMPLE}
                            description="Загрузите снимок - здесь появятся заключение, разметка и измерения"
                        />
                    )}
                </Card>
            </Col>
        </Row>
    );
}
