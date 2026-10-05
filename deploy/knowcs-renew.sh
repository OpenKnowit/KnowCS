#!/bin/sh
set -eu

# Certbot runs deploy hooks for every certificate; only handle KnowCS here.
[ "${RENEWED_LINEAGE:-}" = "/etc/letsencrypt/live/knowcs.online" ] || exit 0
install -m 644 "$RENEWED_LINEAGE/fullchain.pem" /opt/1panel/www/sites/knowcs/ssl/fullchain.pem
install -m 600 "$RENEWED_LINEAGE/privkey.pem" /opt/1panel/www/sites/knowcs/ssl/privkey.pem
docker exec 1Panel-openresty-QDQf nginx -t
docker exec 1Panel-openresty-QDQf nginx -s reload
