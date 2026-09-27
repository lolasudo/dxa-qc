import { describe, expect, it } from 'vitest';
import { FAILED_ROW, makeRow } from '../test/fixtures';
import { validateFile } from './files';
import { fileName, formatCm, formatDegrees, formatProbability, formatSeconds, splitViolations } from './format';
import { summarize } from './summary';
import { verdictOf } from './verdict';

const file = (name: string, size: number, type = '') => new File([new Uint8Array(size)], name, { type });

describe('format', () => {
    it('splits violations with and without spaces', () => {
        expect(splitViolations('A;B ; C')).toEqual(['A', 'B', 'C']);
        expect(splitViolations('')).toEqual([]);
    });

    it('formats numbers the Russian way', () => {
        expect(formatProbability(0.914)).toBe('91%');
        expect(formatProbability(null)).toBe('-');
        expect(formatSeconds(0.25)).toBe('250 мс');
        expect(formatSeconds(3.21)).toBe('3,2 с');
        expect(formatDegrees(-1.84)).toBe('1,8°');
        expect(formatCm(305)).toBe('30,5 см');
    });

    it('extracts file name from both path separators', () => {
        expect(fileName('a.zip/study/x.dcm')).toBe('x.dcm');
        expect(fileName('C:\\data\\y.dcm')).toBe('y.dcm');
    });
});

describe('verdict and summary', () => {
    it('classifies rows', () => {
        expect(verdictOf(makeRow())).toBe('bad');
        expect(verdictOf(makeRow({ quality_class: 0 }))).toBe('good');
        expect(verdictOf(FAILED_ROW)).toBe('failed');
    });

    it('counts rows and violations', () => {
        const s = summarize([makeRow(), makeRow({ index: 2, quality_class: 0, criteria: [] }), FAILED_ROW]);
        expect(s).toMatchObject({ total: 3, good: 1, withViolations: 1, failed: 1 });
        expect(s.byViolation).toEqual([{ violation: 'Присутствуют посторонние предметы', count: 1 }]);
    });
});

describe('validateFile', () => {
    const max = 1000;

    it('rejects empty, too large and zip-for-study', () => {
        expect(validateFile(file('a.zip', 0), 'zip-or-dicom', max)).toMatch(/пустой/);
        expect(validateFile(file('a.zip', 2000), 'zip-or-dicom', max)).toMatch(/превышает/);
        expect(validateFile(file('a.zip', 10), 'dicom', max)).toMatch(/DICOM/);
    });

    it('accepts DICOM without extension', () => {
        expect(validateFile(file('CR000001', 10), 'dicom', max)).toBeNull();
        expect(validateFile(file('batch.zip', 10, 'application/zip'), 'zip-or-dicom', max)).toBeNull();
    });
});
