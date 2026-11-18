import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The browser only ever talks to `/api/*`. In dev, Vite forwards that to the FastAPI server (stripping the prefix);
// in Docker, nginx does the same. No CORS, and the API URL is never baked into the bundle.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
