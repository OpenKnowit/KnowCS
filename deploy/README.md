# KnowCS server deployment

Production: https://knowcs.online · SSH alias: `pastpaper`.

Initial release: `20261005-2158` (2026-10-05). DNS apex A record points to `129.226.210.66`, TTL 600. HTTPS uses Let’s Encrypt with scheduled renewal.

GitHub Actions only validates the project. PinMe deployment has been removed.

## Publish a new build

Run `npm run lint`, `npm test`, `npm run typecheck` and `npm run build` first.
The build is multi-page: `dist/` holds `index.html`, one `<module>/index.html` per page,
`notes/`, `extend/`, redirect stubs at `lab/*.html` (old prototype links) and hashed `assets/`.
Upload the whole `dist/` contents into a new timestamped directory under
`/opt/1panel/www/sites/knowcs/releases/` (e.g. `tar` locally, `scp`, extract). Compare local and
remote SHA-256 of every file (`find . -type f | sort | xargs shasum -a 256`).
Nginx serves `/<module>/` from `<module>/index.html` via `try_files $uri $uri/`; no config change is needed.
Create a temporary symlink targeting that release, then atomically replace
`/opt/1panel/www/sites/knowcs/current` using `sudo mv -Tf`.
Do not remove prior releases. Static updates do not require an OpenResty reload.

Keep the previous release's recent chunks in the new one before switching `current`:
`find current/assets -type f -mtime -14 -exec cp -pn {} releases/<timestamp>/assets/ \;`. The service worker serves
pages a visitor has seen from its cache and refreshes them in the background (stale-while-revalidate, `src/sw/sw.js`),
so right after a deploy an old page may still ask for the old hashed chunks. File names are content hashes, so nothing
is overwritten; chunks older than two weeks drop out on their own.

The playground's "Run in real Python" card downloads Pyodide from jsDelivr by default
(`https://cdn.jsdelivr.net/pyodide/v314.0.7/full/`, see `src/lib/pyodide/config.ts`): about 6 MB over the wire, fetched
ahead of the first run only on pages that have a playground, and not on a mobile, slow or Save-Data connection
(`useWarmEngine` in `src/site/playground/useEngine.ts`). To serve it
from elsewhere (a self-hosted copy of the Pyodide `full/` directory, or COS + CDN), build with
`VITE_PYODIDE_BASE=https://…/full/ npm run build`. The host must send CORS headers for `.mjs`, `.wasm` and wheels.

## Narration audio (/audio/)

The explainer voice-overs are not part of `dist/` and not in Git. They live in one shared directory,
`/opt/1panel/www/sites/knowcs/audio/<episode>/<lang>.json` + `<lang>.<hash>.m4a`, generated locally into
`work/dubbing/out/audio/` by `scripts/tts/narrate.py` (see its docstring). Every release links to it:
`ln -s ../../audio releases/<timestamp>/audio` (create the link before switching `current`, and leave it out of the
SHA-256 comparison). To publish new or changed tracks, upload the episode folders into `audio/` (old `.m4a` files can
stay; the manifests name the current ones). An episode or language without a manifest simply plays silent. The service
worker does not cache `/audio/`; Nginx serves it under `location /` with Range support and `no-cache`.

## HTTPS and configuration

The virtual host is `/opt/1panel/www/conf.d/knowcs.online.conf` (template in this directory).
Certbot uses webroot `/opt/1panel/www/sites/knowcs/acme`.
The deploy hook `/etc/letsencrypt/renewal-hooks/deploy/knowcs.sh` copies the renewed
certificate into the container's mounted `ssl/` directory, validates Nginx and reloads it.
Back up the virtual host before configuration changes; always run `nginx -t` before reload.

## Verify and roll back

Check HTTPS 200 (home, a module page, `/notes/numpy/`), HTTP-to-HTTPS redirect, gzip, browser console,
the served HTML SHA-256 and that `/lab/home-a.html` redirects. Confirm existing sites still respond.
To roll back content, atomically repoint `current` to the retained prior release.

## Preview site: dev.knowcs.online

The `dev` branch is published to https://dev.knowcs.online (same server, same OpenResty container), first release
`20261102-1403`. DNS: an A record `dev` → `129.226.210.66`.

- Files: `/opt/1panel/www/sites/knowcs-dev/` with the same `releases/<timestamp>/` + `current` layout as production;
  each release links `audio` to production's narration audio (`ln -s ../../../knowcs/audio releases/<ts>/audio`).
- Virtual host: `/opt/1panel/www/conf.d/dev.knowcs.online.conf` (template: `dev.knowcs.online.conf` here). It adds
  `X-Robots-Tag: noindex, nofollow`, so the preview stays out of search results.
- HTTPS: Certbot webroot `/opt/1panel/www/sites/knowcs-dev/acme`, certificate `/etc/letsencrypt/live/dev.knowcs.online/`;
  the deploy hook `/etc/letsencrypt/renewal-hooks/deploy/knowcs-dev.sh` (template: `knowcs-dev-renew.sh`) copies it into
  `ssl/` and reloads OpenResty.
- Publishing is the same as production with `knowcs-dev` in place of `knowcs`. SSH to the server drops now and then:
  run long commands (Certbot) with `nohup … &` and poll their log.

## Admin API: /admin/ and /api/

`/admin/` edits the Resources list (add, edit, delete, reorder, upload a picture). The page talks to a small Node
service, `server/admin.mjs` (no framework; `opencc-js` derives the Traditional Chinese text), which the Resources page
also reads (`GET /api/resources`); without the service the page shows its built-in list. On the server:

- Code: `/opt/knowcs-admin/` (`admin.mjs`, `package.json`, `resources-seed.json`, `npm install --omit=dev`), owned by root.
- Data: `/var/lib/knowcs-admin/` (`resources.json` + `.backup`, `media/` uploads, `session.key`, `stats.json`), owned by the system
  user `knowcs-admin`. The first start copies `resources-seed.json` in; after that the JSON file is the source of truth.
- Settings: `/etc/knowcs-admin.env` (root only, not in Git): `ADMIN_PASSPHRASE`, `DATA_DIR=/var/lib/knowcs-admin`,
  `PORT=3021`, `HOST=127.0.0.1`, `COOKIE_SECURE=true`, `TRUST_PROXY=true`. Change the passphrase there and
  `systemctl restart knowcs-admin`.
- Service: `knowcs-admin.service` (template here), `systemctl status knowcs-admin`, logs with `journalctl -u knowcs-admin`.
- OpenResty: the vhost's `location /api/` proxies to `127.0.0.1:3021` (the container uses host networking), on both
  `knowcs.online.conf` and `dev.knowcs.online.conf`. The two sites share one service, so they show the same list.

Sign-in sets an HttpOnly, SameSite=Strict, signed cookie for 12 hours; five wrong passphrases per minute per IP are
refused; changes are accepted only from the site's own pages (Origin check). Sign-ins, refused ones too, are written
to the journal. Update the code with the same three files and `systemctl restart knowcs-admin`.

Visits: every page sends one `POST /api/hit` when it is opened (`src/site/visit.ts`: path, language, the site that
linked to it), and `/admin/` shows the counts per day (`GET /api/stats`, signed in). Only counts are kept, in
`stats.json`, for 400 days and per site (`STATS_HOSTS`, by default `knowcs.online,dev.knowcs.online`): no cookie, and
no address or browser is stored. A visitor is told apart within one day by a hash under a key that stays in memory,
so a restart counts that day's returning visitors again. Crawlers, the admin and 404 pages, and browsers that send
Do Not Track or Global Privacy Control are left out. The counts are written every 30 seconds and when the service stops.

The API has its own tests, which start the real server on a spare port with an empty data folder: run
`npm run test:server` after changing `server/` (they are not part of `npm test`).
