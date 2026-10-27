import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base: the built site works from any subpath, e.g. GitHub Pages' /comp432-study-guide/.
export default defineConfig({ base: "./", plugins: [react()] });
