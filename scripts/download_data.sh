#!/usr/bin/env bash
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> Create New Token)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for slug in \
  rakibulhasanshaon69/narutosubtitle \
  jef1056/anime-subtitles \
  ramazanturann/one-piece-transcripts-with-character-names-382-777
do
  kaggle datasets download -d "$slug" -p "data/raw/${slug##*/}" --unzip
done
echo "done. next: scripts/build_corpus.sh --limit 3"
