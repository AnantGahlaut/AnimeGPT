#!/usr/bin/env bash
# Build data/corpus.txt. Extra args pass through, e.g.:  scripts/build_corpus.sh --limit 3
# Override auto-detected paths with NARUTO_DIR / DBZ_DIR / OP_CSV if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
NARUTO_DIR=${NARUTO_DIR:-$(find data/raw/narutosubtitle -type d -name Subtitles | head -1)}
DBZ_DIR=${DBZ_DIR:-$(find data/raw/anime-subtitles -type d -path "*Extracted/Extracted" | head -1)}
OP_CSV=${OP_CSV:-$(find data/raw/one-piece-transcripts-with-character-names-382-777 -name "*.csv" | head -1)}
echo "naruto: $NARUTO_DIR | dbz: $DBZ_DIR | one piece: $OP_CSV"
python3 data_pipeline.py \
  --sub "Naruto=$NARUTO_DIR" \
  --sub "Dragon Ball Z=$DBZ_DIR::(^|\W)dbz\W" \
  --csv "One Piece=$OP_CSV" \
  --label --out data/corpus.txt "$@"
