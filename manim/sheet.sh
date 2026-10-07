#!/usr/bin/env bash
# Contact sheet of a rendered video for quick review:  ./sheet.sh <video.mp4> [seconds-per-frame]
set -euo pipefail
cd "$(dirname "$0")"
v=$1
every=${2:-4}
mkdir -p media/check
out="media/check/$(basename "$v" .mp4).png"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$v")
rows=$(python3 -c "import math; print(max(1, math.ceil($dur / $every / 4)))")
ffmpeg -v error -y -i "$v" -vf "fps=1/$every,scale=480:-1,tile=4x$rows" -frames:v 1 "$out"
echo "$out (${dur}s)"
