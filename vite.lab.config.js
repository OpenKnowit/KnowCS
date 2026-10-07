// KnowCS Lab: multi-page prototype build served at /lab/ next to the single-file main site.
// Every lab/*.html is its own page; they share one entry (src/lab/main.tsx) that lazy-loads the page.
import { readdirSync } from "node:fs"
import { resolve } from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { katexFontSlim } from "./scripts/vite-plugins.mjs"

const pages = readdirSync("lab").filter((f) => f.endsWith(".html"))

export default defineConfig({
  plugins: [katexFontSlim(), react()],
  build: {
    outDir: "dist-lab",
    emptyOutDir: true,
    assetsDir: "lab/assets",
    rollupOptions: {
      input: Object.fromEntries(pages.map((f) => [f.replace(/\.html$/, ""), resolve("lab", f)])),
    },
  },
  // only crawl lab pages (the main site's index.html needs the markdown plugins of vite.config.js)
  optimizeDeps: { entries: ["lab/*.html"] },
  server: {
    port: 5175,
    open: "/lab/index.html",
  },
})
