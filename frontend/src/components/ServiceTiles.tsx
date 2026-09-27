import type { KeyboardEvent, ReactNode } from 'react';
import { AppstoreOutlined, FileImageOutlined } from '@ant-design/icons';

export type Section = 'batch' | 'study';

interface Tile {
    value: Section;
    icon: ReactNode;
    title: string;
    text: string;
}

const TILES: Tile[] = [
    {
        value: 'batch',
        icon: <AppstoreOutlined />,
        title: 'Пакетная проверка',
        text: 'ZIP-архив с исследованиями: таблица результатов, отчёт XLSX/CSV и разметка снимков',
    },
    {
        value: 'study',
        icon: <FileImageOutlined />,
        title: 'Один снимок',
        text: 'Проверка одного DICOM-файла прямо в кабинете: вердикт, заключение и измерения',
    },
];

/** Плашки разделов (роль tablist). */
export function ServiceTiles({ value, onChange }: { value: Section; onChange: (s: Section) => void }) {
    const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        e.preventDefault();
        const next = TILES[(TILES.findIndex((t) => t.value === value) + 1) % TILES.length];
        onChange(next.value);
        document.getElementById(`tab-${next.value}`)?.focus();
    };

    return (
        <div className="tiles" role="tablist" aria-label="Разделы сервиса" onKeyDown={onKeyDown}>
            {TILES.map((tile) => {
                const active = tile.value === value;
                return (
                    <button
                        key={tile.value}
                        id={`tab-${tile.value}`}
                        type="button"
                        role="tab"
                        aria-selected={active}
                        aria-controls={`panel-${tile.value}`}
                        tabIndex={active ? 0 : -1}
                        className={active ? 'tile tile--active' : 'tile'}
                        onClick={() => onChange(tile.value)}
                    >
                        <span className="tile__icon" aria-hidden>
                            {tile.icon}
                        </span>
                        <span className="tile__body">
                            <span className="tile__title">{tile.title}</span>
                            <span className="tile__text">{tile.text}</span>
                        </span>
                    </button>
                );
            })}
        </div>
    );
}
