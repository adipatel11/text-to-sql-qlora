#!/usr/bin/env bash
# Download and unpack the Spider 1.0 dataset into data/spider.
#
# Spider is distributed as a Google Drive zip, and Drive file IDs occasionally
# rot. If the default ID below stops working, grab the current link from
#   https://yale-lily.github.io/spider
# and either set SPIDER_GDRIVE_ID=<id> or download manually (see bottom).
#
# The end state we need (used by textsql.data.prepare_spider):
#   data/spider/train_spider.json
#   data/spider/train_others.json
#   data/spider/dev.json
#   data/spider/tables.json
#   data/spider/database/<db_id>/<db_id>.sqlite   <- the executable DBs
set -euo pipefail

DEST="${1:-data/spider}"
SPIDER_GDRIVE_ID="${SPIDER_GDRIVE_ID:-1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m}"
TMP_ZIP="$(mktemp -t spider_XXXX).zip"

mkdir -p "$DEST"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown (Google Drive downloader)..."
  pip install --quiet gdown
fi

echo "Downloading Spider (gdrive id: $SPIDER_GDRIVE_ID) ..."
if ! gdown "$SPIDER_GDRIVE_ID" -O "$TMP_ZIP"; then
  cat <<'EOF'
[!] gdown download failed. The Drive ID has probably changed.

Manual fallback:
  1. Open https://yale-lily.github.io/spider and download "spider.zip".
  2. unzip spider.zip
  3. Move the contents so you have:
        data/spider/train_spider.json
        data/spider/dev.json
        data/spider/database/<db_id>/<db_id>.sqlite
  4. Re-run: python -m textsql.data.prepare_spider
EOF
  exit 1
fi

echo "Unzipping ..."
unzip -q -o "$TMP_ZIP" -d "$DEST/_unzipped"
rm -f "$TMP_ZIP"

# The zip usually nests everything under a top-level "spider/" or "spider_data/"
# folder. Flatten so files land directly under $DEST.
INNER="$(find "$DEST/_unzipped" -maxdepth 2 -name dev.json -print -quit || true)"
if [ -n "$INNER" ]; then
  SRC_DIR="$(dirname "$INNER")"
  echo "Flattening $SRC_DIR -> $DEST"
  cp -R "$SRC_DIR"/. "$DEST"/
fi
rm -rf "$DEST/_unzipped"

echo "Done. Sanity check:"
ls -1 "$DEST" | sed 's/^/  /'
echo
echo "Next: python -m textsql.data.prepare_spider --spider-dir $DEST"
