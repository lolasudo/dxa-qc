import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Alert, Button } from 'antd';

interface ErrorBoundaryProps {
    children: ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    message?: string;
}

export class ErrorBoundary extends Component<
    ErrorBoundaryProps,
    ErrorBoundaryState
> {
    state: ErrorBoundaryState = {
        hasError: false,
    };

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return {
            hasError: true,
            message: error.message,
        };
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
        // Локальный лог. Внешние сервисы не используем из-за офлайн-требования.
        console.error('Frontend error:', error, errorInfo);
    }

    handleReset = () => {
        this.setState({ hasError: false, message: undefined });
    };

    render() {
        if (this.state.hasError) {
            return (
                <Alert
                    type="error"
                    showIcon
                    message="Ошибка интерфейса"
                    description={this.state.message ?? 'Неизвестная ошибка'}
                    action={
                        <Button size="small" onClick={this.handleReset}>
                            Повторить
                        </Button>
                    }
                />
            );
        }

        return this.props.children;
    }
}