#!/bin/bash
SLIDE_DIR="$1"
VIDEO_DIR="$2"
OUTPUT_DIR="$3"
width="$4"
height="$5"
num_video="$6"
VIDEO_PATH="$7"
ref_img="$8"
mkdir -p "$OUTPUT_DIR"
> list.txt

for i in $(seq 1 "$num_video"); do
  slide_path="$SLIDE_DIR/$i.png"
  video_path="$VIDEO_DIR/$((i-1))/"$ref_img"/merge_video.mp4"
  output_path="$OUTPUT_DIR/page_$(printf "%03d" "$i").mp4"

  if [[ ! -f "$slide_path" || ! -f "$video_path" ]]; then
    echo "❌ Skip $i: missing slide or video"
    continue
  fi

  has_audio=$(ffprobe -v error -select_streams a -show_entries stream=codec_type \
    -of default=noprint_wrappers=1:nokey=1 "$video_path")

  if [[ -z "$has_audio" ]]; then
    echo "⚠️ Skip $i: no audio stream"
    continue
  fi

  duration=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$video_path")

  echo "✅ Processing $i..."

  ffmpeg -y \
    -loop 1 -t "$duration" -i "$slide_path" \
    -i "$video_path" \
    -filter_complex "[1:v]scale="$width":"$height"[avatar];[0:v][avatar]overlay=W-w-10:10[outv]" \
    -map "[outv]" -map 1:a -c:v libx264 -c:a aac -preset fast -crf 23 \
    -shortest "$output_path"

  echo "file '$output_path'" >> list.txt
done

if [[ -s list.txt ]]; then
  ffmpeg -f concat -safe 0 -i list.txt -c copy $VIDEO_PATH -y
  echo "🎬 All done!"
else
  echo "⚠️ No valid clips to concatenate"
fi
