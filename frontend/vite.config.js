import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Ensure react() is correctly used inside plugins
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5000", // Proxy requests to Flask
    },
  },
});
