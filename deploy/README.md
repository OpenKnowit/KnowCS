# KnowCS server deployment

Production: https://knowcs.online · SSH alias: `pastpaper`.

Initial release: `20261005-2158` (2026-10-05). DNS apex A record points to `129.226.210.66`, TTL 600. HTTPS uses Let’s Encrypt with scheduled renewal.

GitHub Actions only validates the project. PinMe deployment has been removed.

## Publish a new build

Run `npm run lint`, `npm test`, `npm run typecheck` and `npm run build` first.
Upload `dist/index.html` into a new timestamped directory under
`/opt/1panel/www/sites/knowcs/releases/`. Compare local and remote SHA-256.
Create a temporary symlink targeting that release, then atomically replace
`/opt/1panel/www/sites/knowcs/current` using `sudo mv -Tf`.
Do not remove prior releases. Static updates do not require an OpenResty reload.

## HTTPS and configuration

The virtual host is `/opt/1panel/www/conf.d/knowcs.online.conf` (template in this directory).
Certbot uses webroot `/opt/1panel/www/sites/knowcs/acme`.
The deploy hook `/etc/letsencrypt/renewal-hooks/deploy/knowcs.sh` copies the renewed
certificate into the container's mounted `ssl/` directory, validates Nginx and reloads it.
Back up the virtual host before configuration changes; always run `nginx -t` before reload.

## Verify and roll back

Check HTTPS 200, HTTP-to-HTTPS redirect, gzip, browser console, and the served HTML
SHA-256. Confirm existing sites still respond.
To roll back content, atomically repoint `current` to the retained prior release.
