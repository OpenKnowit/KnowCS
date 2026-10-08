// KnowCS: a multi-page site. Every page in src/lib/sitemap.ts gets its own HTML file, generated into site/
// (gitignored) when the config loads; all pages share one entry script that renders the right page.
import { createHash } from "node:crypto"
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { katexFontSlim, localeSplit, markdownHtml, rawHk } from "./scripts/vite-plugins.mjs"
import { PAGES } from "./src/lib/sitemap.ts"

const ROOT = resolve("site")
const FAVICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0' stop-color='%233b82f6'/%3E%3Cstop offset='1' stop-color='%234f46e5'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='14' fill='url(%23g)'/%3E%3Cg stroke='%23fff' stroke-width='2.8' opacity='0.8'%3E%3Cline x1='16' y1='20' x2='32' y2='14'/%3E%3Cline x1='16' y1='20' x2='32' y2='32'/%3E%3Cline x1='16' y1='20' x2='32' y2='50'/%3E%3Cline x1='16' y1='44' x2='32' y2='14'/%3E%3Cline x1='16' y1='44' x2='32' y2='32'/%3E%3Cline x1='16' y1='44' x2='32' y2='50'/%3E%3Cline x1='32' y1='14' x2='48' y2='32'/%3E%3Cline x1='32' y1='32' x2='48' y2='32'/%3E%3Cline x1='32' y1='50' x2='48' y2='32'/%3E%3C/g%3E%3Cg fill='%23fff'%3E%3Ccircle cx='16' cy='20' r='6'/%3E%3Ccircle cx='16' cy='44' r='6'/%3E%3Ccircle cx='32' cy='14' r='6'/%3E%3Ccircle cx='32' cy='32' r='6'/%3E%3Ccircle cx='32' cy='50' r='6'/%3E%3Ccircle cx='48' cy='32' r='6'/%3E%3C/g%3E%3C/svg%3E"
const DESCRIPTION = "KnowCS: interactive visual lab for HKUST COMP2211 Machine Learning: Naive Bayes, KNN, K-Means, perceptrons, backpropagation, convolution, CNNs and alpha-beta pruning. English / 简体中文 / 繁體中文."

const escape = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;")
const SITE = "https://knowcs.online"
const EN = JSON.parse(readFileSync(new URL("./src/locales/en.json", import.meta.url), "utf8"))
const ZH = JSON.parse(readFileSync(new URL("./src/locales/zh.json", import.meta.url), "utf8"))
const get = (o, path) => path.split(".").reduce((a, k) => (a && typeof a === "object" ? a[k] : undefined), o)

/** Static title / description for link previews and search engines (the app sets the live title per language). */
function pageMeta(page) {
  const pick = (titleKey, descKey) => {
    const en = get(EN, titleKey)
    const zh = get(ZH, titleKey)
    return { title: en && zh && en !== zh ? `${en} · ${zh}` : en ?? "KnowCS", description: (descKey && get(EN, descKey)) || DESCRIPTION }
  }
  switch (page.kind) {
    case "module":
      return pick(`site.modules.${page.id}.title`, `site.modules.${page.id}.blurb`)
    case "watch":
      return pick("watch.ui.gallery_title", "watch.ui.gallery_lead")
    case "watch-item":
      return pick(`watch.${page.id}.title`, `watch.${page.id}.sub`)
    case "drill":
      return pick("drill.title", "drill.lead")
    case "formulas":
      return pick("formulas.title", "formulas.lead")
    case "papers":
      return pick("papers.title", "papers.lead")
    case "notfound":
      return pick("notfound.title", "notfound.lead")
    case "notes":
      return pick("app.section.package.title", "app.section.package.subtitle")
    case "note":
      return pick(`package.notes.${page.id}`, "app.section.package.subtitle")
    case "extend":
      return pick("app.section.extend.title", "app.section.extend.subtitle")
    case "extend-item":
      return pick(`extend.items.${page.id}.title`, "app.section.extend.subtitle")
    default:
      return { title: "COMP2211 Interactive ML Lab · 交互式机器学习实验室", description: DESCRIPTION }
  }
}
const urlOf = (page) => `${SITE}/${page.path.replace(/index\.html$/, "")}`

function pageHtml(page) {
  const up = "../".repeat(page.path.split("/").length)
  const head = `<meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#2563eb" />
    <link rel="icon" type="image/svg+xml" href="${FAVICON}" />
    <link rel="manifest" href="/manifest.json" />
    <link rel="apple-touch-icon" href="/apple-touch-icon.png" />`
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
  const meta = pageMeta(page)
  return `<!doctype html>
<html lang="en">
  <head>
    ${head}
    <meta name="description" content="${escape(meta.description)}" />
    <link rel="canonical" href="${urlOf(page)}" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="KnowCS" />
    <meta property="og:url" content="${urlOf(page)}" />
    <meta property="og:title" content="${escape(meta.title)}" />
    <meta property="og:description" content="${escape(meta.description)}" />
    <meta name="twitter:card" content="summary" />
    <title>${escape(meta.title)} · KnowCS</title>
  </head>
  <body data-page="${page.kind}" data-id="${page.id ?? ""}">
    <noscript>KnowCS needs JavaScript: every page is an interactive visualisation. · 本站的互动页面需要启用 JavaScript。</noscript>
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
// sitemap.xml and robots.txt, served from the site root
mkdirSync(resolve(ROOT, "public"), { recursive: true })
writeFileSync(
  resolve(ROOT, "public/sitemap.xml"),
  `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${PAGES.filter((p) => p.kind !== "redirect" && p.kind !== "notfound").map((p) => `  <url><loc>${urlOf(p)}</loc></url>`).join("\n")}
</urlset>
`,
)
copyFileSync(resolve("src/sw/sw.js"), resolve(ROOT, "public/sw.js"))
// installable as an app: the favicon as a real file, plus a manifest
writeFileSync(resolve(ROOT, "public/icon.svg"), decodeURIComponent(FAVICON.replace("data:image/svg+xml,", "")))
for (const f of ["icon-192.png", "icon-512.png", "apple-touch-icon.png"]) copyFileSync(resolve("src/assets/icons", f), resolve(ROOT, "public", f))
writeFileSync(
  resolve(ROOT, "public/manifest.json"),
  JSON.stringify({
    name: "KnowCS · COMP2211 Interactive ML Lab",
    short_name: "KnowCS",
    start_url: "/",
    display: "standalone",
    background_color: "#f1f5f9",
    theme_color: "#2563eb",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
    ],
  }),
)
writeFileSync(resolve(ROOT, "public/robots.txt"), `User-agent: *\nAllow: /\nSitemap: ${SITE}/sitemap.xml\n`)

/**
 * offline.json: every page and hashed asset of this build, for "save for offline" (see src/sw/sw.js). Its version
 * and size also go into every page as <meta name="knowcs-offline">, so the footer button needs no request.
 */
function offlineManifest() {
  return {
    name: "knowcs:offline-manifest",
    apply: "build",
    generateBundle: {
      order: "post",
      handler(_, bundle) {
        const files = Object.values(bundle).filter((f) => f.fileName.startsWith("assets/"))
        const size = (f) => (f.type === "chunk" ? Buffer.byteLength(f.code) : typeof f.source === "string" ? Buffer.byteLength(f.source) : f.source.length)
        const assets = [...files.map((f) => `/${f.fileName}`).sort(), "/icon.svg", "/icon-192.png", "/apple-touch-icon.png", "/manifest.json"]
        const pages = PAGES.filter((p) => p.kind !== "redirect" && p.kind !== "notfound").map((p) => `/${p.path.replace(/index\.html$/, "")}`)
        const version = createHash("sha256").update(assets.join()).digest("hex").slice(0, 12)
        const bytes = files.reduce((n, f) => n + size(f), 0)
        this.emitFile({ type: "asset", fileName: "offline.json", source: JSON.stringify({ version, bytes, pages, assets }) })
        const meta = `<meta name="knowcs-offline" content="${version} ${bytes}" />`
        for (const f of Object.values(bundle))
          if (f.type === "asset" && f.fileName.endsWith(".html") && typeof f.source === "string") f.source = f.source.replace("</head>", `    ${meta}\n  </head>`)
      },
    },
  }
}

/**
 * The core strings are only requested once the entry script has run. A tiny inline script picks the language the
 * way i18n.ts does (?lang=, then the saved choice, then the browser) and preloads that file right away.
 */
function preloadStrings() {
  return {
    name: "knowcs:preload-strings",
    apply: "build",
    transformIndexHtml: {
      order: "post",
      handler(html, ctx) {
        if (!ctx.bundle) return html
        const files = {}
        for (const f of Object.values(ctx.bundle)) {
          const m = f.type === "chunk" && f.facadeModuleId && /locales\/(en|zh|zh-HK)\.json\?core$/.exec(f.facadeModuleId)
          if (m) files[m[1]] = `/${f.fileName}`
        }
        if (Object.keys(files).length !== 3) return html
        const script = `<script>(function(){try{var f=${JSON.stringify(files)};var t=new URLSearchParams(location.search).get("lang")||localStorage.getItem("knowcs-lang")||(navigator.languages&&navigator.languages[0])||navigator.language||"";t=t.toLowerCase().replace(/_/g,"-");var l=t.indexOf("zh")!==0?"en":/(^|-)(hant|hk|tw|mo)(-|$)/.test(t)?"zh-HK":"zh";var a=document.createElement("link");a.rel="modulepreload";a.href=f[l];document.head.appendChild(a)}catch(e){}})()</script>`
        return html.replace("</head>", `    ${script}\n  </head>`)
      },
    },
  }
}

export default defineConfig({
  root: ROOT,
  plugins: [katexFontSlim(), markdownHtml(), rawHk(), localeSplit(), offlineManifest(), preloadStrings(), react()],
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
