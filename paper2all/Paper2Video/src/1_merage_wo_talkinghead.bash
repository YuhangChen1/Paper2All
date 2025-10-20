#!/usr/bin/env bash
set -euo pipefail

SLIDE_DIR="${1:-}"; AUDIO_DIR="${2:-}"; OUTPUT_MP4="${3:-}"
FPS="${4:-25}"
SIZE="${5:-keep}"

if [[ -z "${SLIDE_DIR}" || -z "${AUDIO_DIR}" || -z "${OUTPUT_MP4}" ]]; then
  echo "usage: $0 <SLIDE_DIR> <AUDIO_DIR> <OUTPUT_MP4> [FPS=30] [SIZE=keep]"
  exit 1
fi

command -v ffmpeg >/dev/null 2>&1 || { echo "need ffmpeg"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "need ffprobe"; exit 1; }

TMP_DIR="$(mktemp -d -t slides_audio_to_video-XXXXXXXX)"
SEG_DIR="$TMP_DIR/segments"
LIST_FILE="$TMP_DIR/list.txt"
mkdir -p "$SEG_DIR"
: > "$LIST_FILE"

if [[ "$SIZE" == "keep" ]]; then
  VF="scale=trunc(iw/2)*2:trunc(ih/2)*2"
else
  VF="scale=${SIZE}"
fi

echo "merge slides and audio"
i=1
made_any=false
while true; do
  IMG="$SLIDE_DIR/$i.png"
  WAV_IDX=$((i-1))
  WAV="$AUDIO_DIR/$WAV_IDX.wav"

  if [[ ! -f "$IMG" ]]; then
    break
  fi

  if [[ ! -f "$WAV" ]]; then
    echo "⚠️  skip page $i, no $WAV"
    i=$((i+1))
    continue
  fi

  if ! ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 "$WAV" >/dev/null; then
    echo "⚠️  skip page $i, no $WAV"
    i=$((i+1))
    continue
  fi

  SEG="$SEG_DIR/page_$(printf "%03d" "$i").mp4"
  echo "  • generate clip $IMG  +  $WAV  →  $(basename "$SEG")"

  ffmpeg -y -hide_banner -loglevel error \
    -r "$FPS" -loop 1 -i "$IMG" -i "$WAV" \
    -tune stillimage -shortest \
    -vf "$VF,format=yuv420p" \
    -c:v libx264 -preset medium -crf 18 -r "$FPS" \
    -c:a aac -b:a 192k \
    -movflags +faststart \
    -map 0:v:0 -map 1:a:0 \
    "$SEG"

  echo "file '$SEG'" >> "$LIST_FILE"
  made_any=true
  i=$((i+1))
done

if [[ "$made_any" != true ]]; then
  echo "❌ unsuccessful"
  rm -rf "$TMP_DIR"
  exit 1
fi

echo "▶︎ merge all ..."
ffmpeg -y -hide_banner -loglevel error \
  -f concat -safe 0 -i "$LIST_FILE" \
  -c copy \
  "$OUTPUT_MP4"

echo "✅ finished $OUTPUT_MP4"
rm -rf "$TMP_DIR"
