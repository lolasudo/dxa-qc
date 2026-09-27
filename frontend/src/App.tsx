import { useEffect, useState } from 'react';
import { Layout } from 'antd';
import { AppHeader } from './components/AppHeader';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Hero } from './components/Hero';
import { ServiceTiles, type Section } from './components/ServiceTiles';
import { BatchPage } from './pages/BatchPage';
import { StudyPage } from './pages/StudyPage';

// Раздел живёт в адресе (#study), чтобы ссылка открывала нужный раздел, а «назад» работал.
function sectionFromHash(): Section {
    return window.location.hash === '#study' ? 'study' : 'batch';
}

export default function App() {
    const [section, setSection] = useState<Section>(sectionFromHash);

    useEffect(() => {
        const onHash = () => setSection(sectionFromHash());
        window.addEventListener('hashchange', onHash);
        return () => window.removeEventListener('hashchange', onHash);
    }, []);

    const changeSection = (next: Section) => {
        setSection(next);
        const hash = next === 'study' ? '#study' : '#batch';
        if (window.location.hash !== hash) window.history.pushState(null, '', hash);
    };

    return (
        <Layout className="app">
            <a className="skip-link" href="#main">
                Перейти к содержимому
            </a>
            <AppHeader />
            <Layout.Content id="main" tabIndex={-1}>
                <div className="page">
                    <Hero />
                    <ServiceTiles value={section} onChange={changeSection} />
                    <div className="workspace">
                        <ErrorBoundary>
                            {/* Обе страницы остаются смонтированными: переключение не теряет загруженные результаты. */}
                            <div hidden={section !== 'batch'} role="tabpanel" id="panel-batch" aria-labelledby="tab-batch">
                                <BatchPage />
                            </div>
                            <div hidden={section !== 'study'} role="tabpanel" id="panel-study" aria-labelledby="tab-study">
                                <StudyPage />
                            </div>
                        </ErrorBoundary>
                    </div>
                </div>
            </Layout.Content>
            <Layout.Footer className="app-footer">
                <div className="page app-footer__inner">
                    <p>
                        Результаты носят вспомогательный характер: решение о повторном исследовании принимает
                        специалист.
                    </p>
                    <p>Исследования обрабатываются на локальном сервере и не передаются во внешние системы.</p>
                </div>
            </Layout.Footer>
        </Layout>
    );
}
