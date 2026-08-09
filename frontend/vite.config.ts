import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // The dev-server harness assigns a free port via PORT; 5173 is the
    // standalone default.
    port: Number((globalThis as { process?: { env?: Record<string, string> } }).process?.env?.PORT) || 5173,
  },
});
