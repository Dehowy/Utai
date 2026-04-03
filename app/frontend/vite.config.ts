import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      usePolling: true,
      interval: 1000,
    },
    proxy: {
      '/api': {
        // host.docker.internal for Docker, localhost for native dev
        target: process.env.VITE_API_TARGET || 'http://localhost:8765',
        ws: true,
      },
      '/ws': {
        target: process.env.VITE_WS_TARGET || 'ws://localhost:8765',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
