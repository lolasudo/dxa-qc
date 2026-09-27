/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        // 127.0.0.1, не localhost: на Windows localhost может уйти в IPv6 (::1), а API слушает IPv4.
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2020',
    sourcemap: false,
    rollupOptions: {
      output: {
        // ВАЖНО: для Vite 8 + Rolldown manualChunks должна быть функцией
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined;
          }

          if (id.includes('/react/') || id.includes('/react-dom/')) {
            return 'vendor-react';
          }

          if (id.includes('@tanstack')) {
            return 'vendor-query';
          }

          if (id.includes('/antd/') || id.includes('@ant-design')) {
            return 'vendor-antd';
          }

          return 'vendor-other';
        },
      },
    },
  },
});