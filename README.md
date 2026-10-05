# anime-gpt

A small GPT trained from scratch on speaker-labeled anime dialogue (Naruto, Dragon Ball Z, One Piece),
so you can type `Sasuke: NARUTO!!!!` and have the model answer as `Naruto:`.

Built to learn how decoder-only transformers actually work, and to compare tokenizers
(character-level vs word-level vs BPE) on the same data and model size.

## Status

- [x] Data pipeline: sources, 12 documented cleaning steps, per-rule drop counts, 16 unit tests
- [ ] Speaker labeling run (Naruto, DBZ via Claude; One Piece has human labels) + hand-checked accuracy
- [ ] Char-level GPT trained (`gpt.py`, written, not yet run end to end)
- [ ] Word-level tokenizer, then BPE
- [ ] Tokenizer comparison table (below)

## Results (to fill in)

| Tokenizer | Vocab | Params | Val loss | Perplexity | Train time |
|---|---|---|---|---|---|
| char | | | | | |
| word | | | | | |
| BPE | | | | | |

Label quality: _N_ random lines hand-checked per show, _X_% correct (see `work/review_sample.txt` after a run).

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # for LLM speaker labels
scripts/download_data.sh                # needs ~/.kaggle/kaggle.json
scripts/build_corpus.sh --limit 3       # cheap test run; read work/report.txt + work/review_sample.txt
scripts/build_corpus.sh                 # full run -> data/corpus.txt
python3 gpt.py train --data data/corpus.txt --small --iters 200   # smoke test
python3 gpt.py train --data data/corpus.txt                        # real run (GPU)
python3 gpt.py chat --ckpt ckpt.pt --show Naruto
pytest                                  # data-cleaning tests
```

Chat commands: `/show NAME`, `/reply NAME|auto`, `/n K`, `/temp T`, `/me NAME`, `/reset`, `/quit`.

## Data

Nothing from the datasets is committed. Download them yourself:

| Show | Source | Speakers |
|---|---|---|
| Naruto (220 eps) | Kaggle `rakibulhasanshaon69/narutosubtitle` | LLM-labeled |
| Dragon Ball Z (eps 1-100) | Kaggle `jef1056/anime-subtitles` (`Extracted/Extracted`) | LLM-labeled |
| One Piece (eps 382-777) | Kaggle `ramazanturann/one-piece-transcripts-with-character-names-382-777` | from fansub files (human) |

The subtitles are fan-made and copyrighted. This repo is for personal/educational use; do not redistribute the corpus.

### Cleaning steps

Every step is counted and written to `work/report.txt`: parse (skip typeset signs/songs), strip markup, normalize to ASCII,
drop noise (music lines, "(sighs)"), dedupe, remove theme-song/recap runs repeated at episode edges, drop broken episodes,
label speakers, merge split cues, normalize names, validate. Full list in the `data_pipeline.py` docstring.

Known limits: LLM labels will slip on fast back-and-forth between minor characters; the model matches style, not meaning;
the corpus is a few MB, so the model overfits quickly (validation holds out every 20th episode).

## Layout

```
data_pipeline.py   sources + cleaning + labeling -> corpus
gpt.py             model, training, interactive chat
scripts/           download + build wrappers
tests/             cleaning-step tests
data/raw/          downloaded datasets (gitignored)
work/              reports, label cache (gitignored)
```
