import '@testing-library/jest-dom/vitest';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    window.localStorage.clear();
});

// jsdom не реализует API, которые использует antd.
Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
    }),
});

class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}

if (!('ResizeObserver' in window)) {
    Object.defineProperty(window, 'ResizeObserver', { writable: true, value: ResizeObserverStub });
}
