#!/bin/bash
# VoD dataset bootstrap on remote compute node.
# Run after view_of_delft_PUBLIC_2.zip finishes uploading.
# Usage: bash bootstrap_vod.sh
set -euo pipefail

VOD=/root/autodl-tmp/datasets/vod
ZIP="$VOD/view_of_delft_PUBLIC_2.zip"
EXPECTED=8683

echo "== [1/6] check disk =="
df -h "$VOD"

echo "== [2/6] verify zip integrity (CRC) =="
unzip -tq "$ZIP" > /tmp/unzip_test.log 2>&1 && tail -n 1 /tmp/unzip_test.log || { echo "ZIP CRC FAILED"; tail -n 20 /tmp/unzip_test.log; exit 1; }

echo "== [3/6] extract =="
cd "$VOD"
unzip -oq "$ZIP"

echo "== [4/6] repair symlink stubs =="
cd "$VOD/view_of_delft_PUBLIC"
# Each stub is a small text file whose content is a relative target path.
# Replace it with a real symlink. Targets may use backslashes or trailing slashes.
find . -type f -size -200c -print0 | while IFS= read -r -d '' f; do
  content=$(tr -d '\r\n' < "$f" | sed 's/\\$//')
  # normalise windows backslashes to forward slashes
  target=$(echo "$content" | tr '\\' '/')
  case "$target" in
    ../*|./*|..\\*)
      rm -f "$f"
      ln -s "$target" "$f"
      echo "linked: $f -> $target"
      ;;
  esac
done

echo "== [5/6] verify structure =="
cnt=$(ls radar_5frames/training/velodyne | wc -l)
echo "radar_5frames/training/velodyne count = $cnt (expected $EXPECTED)"
[ "$cnt" -eq "$EXPECTED" ] || echo "WARN: count mismatch"

# confirm the label_2 symlink resolves through to real files
if [ -d radar_5frames/training/label_2 ]; then
  echo "label_2 resolves: $(ls radar_5frames/training/label_2 | wc -l) files"
else
  echo "WARN: radar_5frames/training/label_2 not a directory/symlink-to-dir"
fi

echo "== [6/6] project data symlink =="
mkdir -p /root/PRISM-Pillars/data/VoD
ln -sfn "$VOD/view_of_delft_PUBLIC" /root/PRISM-Pillars/data/VoD/view_of_delft_PUBLIC
ls -l /root/PRISM-Pillars/data/VoD/

echo "VOD_BOOTSTRAP_DONE"
