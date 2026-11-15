import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './test',
  fullyParallel: false,
  use: { baseURL: 'http://127.0.0.1:5178', viewport: { width: 1440, height: 900 }, headless: true, launchOptions: { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] } },
  webServer: { command: 'npm run dev -- --port 5178 --strictPort', url: 'http://127.0.0.1:5178', reuseExistingServer: !process.env.CI },
});
