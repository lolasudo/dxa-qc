/** Синий баннер вверху страницы; справа схема поясничного отдела и проксимального отдела бедра. */
export function Hero() {
    return (
        <section className="hero" aria-labelledby="hero-title">
            <div className="hero__text">
                <h1 id="hero-title" className="hero__title">
                    Проверка качества денситометрии
                </h1>
                <p className="hero__lead">
                    Сервис смотрит снимки поясничного отдела позвоночника и проксимального отдела бедра сразу
                    после исследования: проверяет укладку, ось, посторонние предметы и область интереса, пока
                    пациент ещё в кабинете.
                </p>
                <ul className="hero__facts">
                    <li>Работает без интернета</li>
                    <li>Около секунды на снимок</li>
                    <li>Отчёт в формате XLSX и CSV</li>
                </ul>
            </div>
            <HeroArt />
        </section>
    );
}

function HeroArt() {
    const vertebrae = [0, 1, 2, 3, 4];
    return (
        <svg className="hero__art" viewBox="0 0 260 220" aria-hidden focusable="false">
            <g fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                {/* поясничный отдел: пять позвонков и ось */}
                {vertebrae.map((i) => (
                    <rect key={i} x="36" y={22 + i * 34} width="56" height="26" rx="9" opacity={0.9 - i * 0.08} />
                ))}
                <path d="M64 12v196" strokeDasharray="4 8" opacity="0.6" />
                {/* проксимальный отдел бедра: головка, шейка, вертелы, диафиз */}
                <circle cx="176" cy="58" r="24" />
                <path d="M194 72c14 10 24 22 30 38M190 84c-2 22 2 44 8 62l2 66M226 112c6 12 6 26 0 40l-2 60" />
                <path d="M222 106c10-2 18 2 22 10" opacity="0.7" />
                {/* рамка области интереса */}
                <rect x="140" y="18" width="112" height="120" rx="10" strokeDasharray="6 7" opacity="0.55" />
            </g>
        </svg>
    );
}
