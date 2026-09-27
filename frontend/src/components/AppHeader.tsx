import { Layout } from 'antd';
import { Logo } from './Logo';
import { ServiceStatus } from './ServiceStatus';
import { ThemeToggle } from './ThemeToggle';

export function AppHeader() {
    return (
        <Layout.Header className="app-header">
            <div className="app-header__inner">
                <div className="brand">
                    <Logo />
                    <div>
                        <div className="brand__title">Контроль качества DXA</div>
                        <div className="brand__subtitle">Помощник рентгенолаборанта</div>
                    </div>
                </div>
                <div className="header-actions">
                    <ServiceStatus />
                    <ThemeToggle />
                </div>
            </div>
        </Layout.Header>
    );
}
