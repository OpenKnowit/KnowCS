// KnowCS: a multi-page site. Every page in src/lib/sitemap.ts gets its own HTML file, generated into site/
// (gitignored) when the config loads; all pages share one entry script that renders the right page.
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { katexFontSlim, markdownHtml, rawHk } from "./scripts/vite-plugins.mjs"
import { PAGES } from "./src/lib/sitemap.ts"

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"))
const ROOT = resolve("site")
const FAVICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0' stop-color='%233b82f6'/%3E%3Cstop offset='1' stop-color='%234f46e5'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='14' fill='url(%23g)'/%3E%3Cg stroke='%23fff' stroke-width='2.8' opacity='0.8'%3E%3Cline x1='16' y1='20' x2='32' y2='14'/%3E%3Cline x1='16' y1='20' x2='32' y2='32'/%3E%3Cline x1='16' y1='20' x2='32' y2='50'/%3E%3Cline x1='16' y1='44' x2='32' y2='14'/%3E%3Cline x1='16' y1='44' x2='32' y2='32'/%3E%3Cline x1='16' y1='44' x2='32' y2='50'/%3E%3Cline x1='32' y1='14' x2='48' y2='32'/%3E%3Cline x1='32' y1='32' x2='48' y2='32'/%3E%3Cline x1='32' y1='50' x2='48' y2='32'/%3E%3C/g%3E%3Cg fill='%23fff'%3E%3Ccircle cx='16' cy='20' r='6'/%3E%3Ccircle cx='16' cy='44' r='6'/%3E%3Ccircle cx='32' cy='14' r='6'/%3E%3Ccircle cx='32' cy='32' r='6'/%3E%3Ccircle cx='32' cy='50' r='6'/%3E%3Ccircle cx='48' cy='32' r='6'/%3E%3C/g%3E%3C/svg%3E"
const DESCRIPTION = "KnowCS: interactive visual lab for HKUST COMP2211 Machine Learning: Naive Bayes, KNN, K-Means, perceptrons, backpropagation, convolution, CNNs and alpha-beta pruning. English / 简体中文 / 繁體中文."

const escape = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;")

function pageHtml(page) {
  const up = "../".repeat(page.path.split("/").length)
  const head = `<meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#2563eb" />
    <link rel="icon" type="image/svg+xml" href="${FAVICON}" />`
  if (page.kind === "redirect") {
    return `<!doctype html>
<html lang="en">
  <head>
    ${head}
    <meta http-equiv="refresh" content="0; url=${page.to}" />
    <title>KnowCS</title>
  </head>
  <body>
    <script>location.replace(${JSON.stringify(page.to)} + location.hash)</script>
    <p>This page moved to <a href="${page.to}">${page.to}</a>.</p>
  </body>
</html>
`
  }
  return `<!doctype html>
<html lang="en">
  <head>
    ${head}
    <meta name="description" content="${escape(DESCRIPTION)}" />
    <meta property="og:title" content="KnowCS · COMP2211 Interactive ML Lab" />
    <meta property="og:description" content="Drag sliders, click canvases and watch machine-learning algorithms work step by step." />
    <title>KnowCS · COMP2211 Interactive ML Lab</title>
  </head>
  <body data-page="${page.kind}" data-id="${page.id ?? ""}">
    <div id="root"></div>
    <script type="module" src="${up}src/site/main.tsx"></script>
  </body>
</html>
`
}

// (re)generate the HTML entries
rmSync(ROOT, { recursive: true, force: true })
for (const page of PAGES) {
  const file = resolve(ROOT, page.path)
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, pageHtml(page))
}

export default defineConfig({
  root: ROOT,
  plugins: [katexFontSlim(), markdownHtml(), rawHk(), react()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  build: {
    outDir: resolve("dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: Object.fromEntries(PAGES.map((p) => [p.path.replace(/\/?index\.html$|\.html$/, "") || "home", resolve(ROOT, p.path)])),
    },
  },
  server: {
    port: 5174,
  },
})
