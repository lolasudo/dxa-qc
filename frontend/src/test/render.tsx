import type { ReactElement } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import { ThemeProvider } from '../theme/ThemeProvider';

export function renderWithProviders(ui: ReactElement) {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
        <QueryClientProvider client={client}>
            <ThemeProvider>{ui}</ThemeProvider>
        </QueryClientProvider>,
    );
}
