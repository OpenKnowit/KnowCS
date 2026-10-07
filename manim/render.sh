#!/usr/bin/env bash
# Render the KnowCS Manim explainers.
#   ./render.sh            # all five, 1080p60
#   ./render.sh l 1 3      # quick 480p preview of topics 1 and 3
#   MANIM=/path/to/manim ./render.sh
set -euo pipefail
cd "$(dirname "$0")"

MANIM=${MANIM:-manim}
Q=${1:-h}
shift || true

SCENES=(
  "1 s1_xor.py XorHiddenLayer"
  "2 s2_broadcast.py BroadcastDistances"
  "3 s3_bayes.py BayesBaseRate"
  "4 s4_backprop.py BackpropXor"
  "5 s5_cnn.py ConvToCnn"
)

want=("$@")
for entry in "${SCENES[@]}"; do
  read -r n file scene <<<"$entry"
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $n " ]]; then continue; fi
  echo "== $n $scene"
  "$MANIM" -q"$Q" --disable_caching --progress_bar none -v WARNING "$file" "$scene"
done
