#!/bin/bash

set -euo pipefail

SLIDE_DIR="$1"
AUDIO_DIR="$2"
OUTPUT_DIR="$3"

num_video="${4:?num_video missing at \$4}"
VIDEO_PATH="${5:?output video path missing at \$5}"
ref_img="${6:-}"

mkdir -p "$OUTPUT_DIR"
: > list.txt

AEXT=("wav" "mp3" "m4a" "aac" "flac" "ogg" "opus")

find_audio() {
  local idx="$1"
  local n=$((idx - 1))
  local cand=""
  for ext in "${AEXT[@]}"; do
    cand="$AUDIO_DIR/${n}.${ext}"
    if [[ -f "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

for i in $(seq 1 "$num_video"); do
  slide_path="$SLIDE_DIR/$i.png"
  if [[ ! -f "$slide_path" ]]; then
    echo "❌ Skip $i: slide not found $slide_path"
    continue
  fi

  audio_path="$(find_audio "$i" || true)"
  if [[ -z "$audio_path" ]]; then
    echo "⚠️  Skip $i: no audio found for index $((i-1)) in $AUDIO_DIR"
    continue
  fi

  duration=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$audio_path" | awk '{printf "%.3f", $0}')
  if [[ -z "$duration" ]]; then
    echo "⚠️  Skip $i: cannot read audio duration"
    continue
  fi

  output_path="$OUTPUT_DIR/page_$(printf "%03d" "$i").mp4"
  echo "✅ Processing $i (slide: $(basename "$slide_path"), audio: $(basename "$audio_path"), ${duration}s)..."

  ffmpeg -y \
    -loop 1 -t "$duration" -i "$slide_path" \
    -i "$audio_path" \
    -map 0:v -map 1:a \
    -c:v libx264 -pix_fmt yuv420p -r 30 -preset fast -crf 23 \
    -c:a aac -b:a 192k -ar 44100 -ac 2 \
    -shortest "$output_path"

  echo "file '$output_path'" >> list.txt
done

if [[ -s list.txt ]]; then
  ffmpeg -y -f concat -safe 0 -i list.txt \
    -c:v libx264 -pix_fmt yuv420p -r 30 -preset fast -crf 23 \
    -c:a aac -b:a 192k -ar 44100 -ac 2 \
    "$VIDEO_PATH"
  echo "🎬 All done! -> $VIDEO_PATH"
else
  echo "⚠️ No valid clips to concatenate"
fi
