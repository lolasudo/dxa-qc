export type AcceptKind = 'zip-or-dicom' | 'dicom';

const ZIP_TYPES = ['application/zip', 'application/x-zip-compressed'];

/** Файлы DICOM часто без расширения, поэтому окончательную проверку делает сервер по сигнатуре. */
export function validateFile(file: File, kind: AcceptKind, maxBytes: number): string | null {
    if (file.size === 0) return 'Файл пустой';
    if (file.size > maxBytes) return `Размер файла превышает ${Math.round(maxBytes / 1024 / 1024)} МБ`;
    const isZip = file.name.toLowerCase().endsWith('.zip') || ZIP_TYPES.includes(file.type);
    if (kind === 'dicom' && isZip) return 'Для одного исследования загрузите DICOM-файл, а не архив';
    return null;
}
