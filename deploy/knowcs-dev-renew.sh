#!/bin/sh
set -eu

# Certbot runs deploy hooks for every certificate; only handle the KnowCS dev site here.
[ "${RENEWED_LINEAGE:-}" = "/etc/letsencrypt/live/dev.knowcs.online" ] || exit 0
install -m 644 "$RENEWED_LINEAGE/fullchain.pem" /opt/1panel/www/sites/knowcs-dev/ssl/fullchain.pem
install -m 600 "$RENEWED_LINEAGE/privkey.pem" /opt/1panel/www/sites/knowcs-dev/ssl/privkey.pem
docker exec 1Panel-openresty-QDQf nginx -t
docker exec 1Panel-openresty-QDQf nginx -s reload
