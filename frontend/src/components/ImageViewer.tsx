import { useState } from 'react';
import { Alert, Space, Switch, Typography } from 'antd';

interface ImageViewerProps {
    previewUrl?: string;
    overlayUrl?: string;
    region?: string;
}

// Цвета совпадают с backend/src/dxa_qc/visualization/overlay.py.
const LEGEND = {
    spine: [
        ['#2ECC71', 'Ось позвоночника (в норме)'],
        ['#EF4444', 'Нарушение / посторонний предмет'],
        ['#F59E0B', 'Кандидат в посторонние предметы'],
        ['#38BDF8', 'Центральная линия позвонков'],
    ],
    hip: [
        ['#2ECC71', 'Ось бедренной кости (в норме)'],
        ['#EF4444', 'Нарушение укладки / поля'],
        ['#D946EF', 'Металл (эндопротез)'],
    ],
} as const;

export function ImageViewer({ previewUrl, overlayUrl, region }: ImageViewerProps) {
    const [showOverlay, setShowOverlay] = useState(true);
    const [failed, setFailed] = useState(false);
    const legend = region?.includes('бедра') ? LEGEND.hip : LEGEND.spine;

    if (!previewUrl || failed) {
        return (
            <Alert
                type="warning"
                showIcon
                message="Изображение недоступно"
                description={
                    failed
                        ? 'Изображение удалено по сроку хранения или сервис недоступен.'
                        : 'Для этого файла изображение не построено (файл не обработан).'
                }
            />
        );
    }

    return (
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <div className="viewer">
                <div className="viewer__stack">
                    <img
                        className="viewer__image"
                        src={previewUrl}
                        alt="Снимок исследования"
                        onError={() => setFailed(true)}
                    />
                    {overlayUrl && showOverlay ? (
                        <img className="viewer__overlay" src={overlayUrl} alt="Разметка нарушений" />
                    ) : null}
                </div>
            </div>
            {overlayUrl ? (
                <Space wrap size={[16, 8]} style={{ justifyContent: 'space-between', width: '100%' }}>
                    <Space size={8}>
                        <Switch
                            checked={showOverlay}
                            onChange={setShowOverlay}
                            aria-label="Показать разметку нарушений"
                        />
                        <Typography.Text>Разметка нарушений</Typography.Text>
                    </Space>
                    <div className="legend">
                        {legend.map(([color, label]) => (
                            <span key={label}>
                                <span className="legend__swatch" style={{ background: color }} />
                                {label}
                            </span>
                        ))}
                    </div>
                </Space>
            ) : null}
        </Space>
    );
}
