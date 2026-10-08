#!/bin/bash
# Normalize interview clips to the format the Qwen3-Omni decoders handle reliably:
#   H.264 + AAC in MP4, short side 480px, constant 25fps, mono audio.
#
#   bash tools/normalize_videos.sh raw/demo01 videos/demo01
#
# Resolution and fps here are about the DECODER being happy and the file being small.
# The sampling that drives token count happens at inference time (FPS / VIDEO_MAX_PIXELS).
set -euo pipefail

SRC=${1:?usage: normalize_videos.sh <src_dir> <dst_dir>}
DST=${2:?usage: normalize_videos.sh <src_dir> <dst_dir>}

# Check dependencies first. Without this, a missing ffprobe makes the audio-stream
# test below fail open and report "NO AUDIO STREAM" for perfectly good files.
missing=()
for cmd in ffmpeg ffprobe; do
  command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "error: not found on PATH: ${missing[*]}" >&2
  echo "  macOS:  brew install ffmpeg" >&2
  echo "  Torch:  ffmpeg is already in the Apptainer env built by setup_env.sh" >&2
  exit 1
fi

[ -d "$SRC" ] || { echo "error: no such directory: $SRC" >&2; exit 1; }
mkdir -p "$DST"

shopt -s nullglob nocaseglob
inputs=("$SRC"/*.{mp4,mov,m4v,mkv,webm,avi})
if [ ${#inputs[@]} -eq 0 ]; then
  echo "error: no video files found in $SRC" >&2
  exit 1
fi
echo "normalizing ${#inputs[@]} file(s) from $SRC -> $DST"

for f in "${inputs[@]}"; do
  base=$(basename "${f%.*}")
  out="$DST/$base.mp4"

  if ! ffprobe -v error -select_streams a -show_entries stream=codec_type \
       -of csv=p=0 "$f" | grep -q audio; then
    echo "!! $base has NO AUDIO STREAM — fix the source before scoring" >&2
  fi

  ffmpeg -y -loglevel error -i "$f" \
    -vf "scale='min(854,iw)':'min(480,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -r 25 -fps_mode cfr \
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
    -c:a aac -b:a 128k -ar 48000 -ac 1 \
    -movflags +faststart \
    "$out"

  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")
  printf "%-20s -> %s  (%.1fs)\n" "$base" "$out" "$dur"
done

echo
echo "Now trim each clip to the candidate's answer only — if the interviewer's"
echo "question is audible, the model will hear and score it. For example:"
echo "  ffmpeg -i $DST/q1.mp4 -ss 00:00:03 -to 00:00:48 -c copy $DST/q1_trimmed.mp4"
