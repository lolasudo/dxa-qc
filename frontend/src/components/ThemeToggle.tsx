import { Button, Tooltip } from 'antd';
import { MoonOutlined, SunOutlined } from '@ant-design/icons';
import { useTheme } from '../theme/context';

export function ThemeToggle() {
    const { mode, toggle } = useTheme();
    const label = mode === 'dark' ? 'Светлая тема' : 'Тёмная тема';
    return (
        <Tooltip title={label}>
            <Button
                shape="circle"
                size="large"
                icon={mode === 'dark' ? <SunOutlined /> : <MoonOutlined />}
                onClick={toggle}
                aria-label={label}
            />
        </Tooltip>
    );
}
