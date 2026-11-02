#!/usr/bin/env bash
# Re-sync pencil/Knowcs/Knowcs.pen from a running KnowCS site (default: production).
#   scripts/pencil-sync/sync.sh                      # from https://knowcs.online
#   BASE=http://localhost:5173 scripts/pencil-sync/sync.sh
# Close pencil/Knowcs/Knowcs.pen in Pencil first, or Pencil may save its open copy over the new file.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${BASE:-https://knowcs.online}
CAP=$(mktemp -d)
trap 'rm -rf "$CAP"' EXIT

SITE="home= playground=playground/ watch=watch/ watch-episode=watch/bayes/ drill=drill/ formulas=formulas/ papers=papers/ notes=notes/ note=notes/numpy/ extend=extend/ extend-item=extend/attention/ 404=404.html"
MODS="numpy bayes-basics bayes-virus naive-bayes gaussian-nb knn evaluation cross-validation kmeans kmeans-table perceptron backprop xor-mlp kernel convolution otsu affine cnn-shapes pytorch alphabeta"

node scripts/pencil-sync/capture.mjs "$CAP" "$BASE" $SITE $(for m in $MODS; do printf '%s=%s/ ' "$m" "$m"; done)
rm -rf pencil/Knowcs/assets
SITE_KEYS=$(for s in $SITE; do printf '%s,' "${s%%=*}"; done)
MOD_KEYS=${MODS// /,}
node scripts/pencil-sync/assemble.mjs "$CAP" pencil/Knowcs/Knowcs.pen "Site:${SITE_KEYS%,}" "Modules:$MOD_KEYS"
