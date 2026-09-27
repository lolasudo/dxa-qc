import { useState } from 'react';
import { CloudUploadOutlined, DeleteOutlined, FileOutlined } from '@ant-design/icons';
import { App, Button, Progress, Space, Typography, Upload, type UploadProps } from 'antd';
import { env } from '../config';
import { validateFile, type AcceptKind } from '../utils/files';

interface FileDropZoneProps {
    kind: AcceptKind;
    busy: boolean;
    progress: number;
    submitLabel: string;
    hint: string;
    onSubmit: (file: File) => void;
}

function formatSize(bytes: number): string {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
    return `${(bytes / 1024 / 1024).toFixed(1).replace('.', ',')} МБ`;
}

export function FileDropZone({ kind, busy, progress, submitLabel, hint, onSubmit }: FileDropZoneProps) {
    const { message } = App.useApp();
    const [file, setFile] = useState<File | null>(null);
    const maxBytes = env.maxUploadSizeMb * 1024 * 1024;

    const beforeUpload: UploadProps['beforeUpload'] = (candidate) => {
        const error = validateFile(candidate, kind, maxBytes);
        if (error) {
            message.error(error);
            return Upload.LIST_IGNORE;
        }
        setFile(candidate);
        return false; // отправкой управляем сами
    };

    return (
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
            {file ? (
                <div className="fact" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <FileOutlined style={{ fontSize: 22, color: 'var(--c-primary)' }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="fact__value" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {file.name}
                        </div>
                        <div className="fact__label">{formatSize(file.size)}</div>
                    </div>
                    <Button
                        type="text"
                        icon={<DeleteOutlined />}
                        disabled={busy}
                        onClick={() => setFile(null)}
                        aria-label="Убрать файл"
                    />
                </div>
            ) : (
                <Upload.Dragger
                    name="file"
                    multiple={false}
                    showUploadList={false}
                    accept={kind === 'dicom' ? '.dcm,application/dicom,*/*' : '.zip,.dcm,application/zip,application/dicom'}
                    beforeUpload={beforeUpload}
                    disabled={busy}
                >
                    <p className="ant-upload-drag-icon" style={{ marginBottom: 8 }}>
                        <CloudUploadOutlined style={{ color: 'var(--c-primary)' }} />
                    </p>
                    <p className="ant-upload-text" style={{ fontWeight: 600 }}>
                        Перетащите файл сюда или нажмите, чтобы выбрать
                    </p>
                    <Typography.Paragraph type="secondary" style={{ margin: '4px 16px 0' }}>
                        {hint}
                    </Typography.Paragraph>
                </Upload.Dragger>
            )}

            {busy ? (
                <Progress
                    percent={progress}
                    status="active"
                    format={(p) => (p === 100 ? 'Анализ...' : `${p}%`)}
                />
            ) : null}

            <Button
                type="primary"
                size="large"
                block
                icon={<CloudUploadOutlined />}
                loading={busy}
                disabled={!file}
                onClick={() => file && onSubmit(file)}
            >
                {submitLabel}
            </Button>
        </Space>
    );
}
