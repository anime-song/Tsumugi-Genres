# Tsumugi-Genres

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Tsumugi--Genres-yellow)](https://huggingface.co/anime-song/Tsumugi-Genres)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)

## Updates

### v0.2.0 (2026-11-22)

- **Electronic genres are more accurate.**  Songs whose genres were checked by hand, many of them game and anime
  electronic music, are now part of training (before, they were only used to test).  In 5-fold cross-validation
  on the same songs, the v0.1.0 recipe → v0.2.0:
  - hand-checked electronic genres: mAP 0.865 → 0.910
  - songs checked as *not* chiptune that still scored 0.5 or more for chiptune: 52 % → 7 %
    (kawaii future bass 69 % → 31 %, denpa 64 % → 36 %)
  - songs checked as dubstep that scored 0.5 or more: 44 % → 76 % (drum and bass 62 % → 81 %, house 40 % → 60 %)
- **Two new genres:** eurobeat and jersey club (137 genres in all).

Genres of whole songs: a score for each of 137 genres, so a song can have several (funk + soul + disco, or
pop + shibuya-kei + jazz).  Small genre heads (6 × 3.4 MB) read the frozen encoder of
[SheetSage2-Refined](https://github.com/anime-song/SheetSage2-Refined).  Details and scores are on the
[model card](https://huggingface.co/anime-song/Tsumugi-Genres).

## Install

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/anime-song/Tsumugi-Genres
```

This installs SheetSage2-Refined too.  The weights are downloaded from Hugging Face on first use: this model
(21 MB) and SheetSage2-Refined's encoder (2.7 GB).  A CUDA GPU is recommended; the CPU works too.

## Usage

```bash
tsumugi-genres song.mp3                  # prints the top 10 genres, writes outputs/genres/song.genres.json
tsumugi-genres *.mp3 --top 5 --device cpu
```

```python
from tsumugi_genres import HeadGenres

model = HeadGenres.load()
for genre in model.predict("song.mp3")[:5]:
    print(genre["id"], genre["name"], round(genre["score"], 3))    # e.g. 24 funk 0.998
```

`predict` returns every genre with its `id`, `name` and `score` (0–1, each genre on its own), highest first.
The genre ids and names are in the model's `config.json`.  If you already run SheetSage2-Refined's encoder,
pass its windows (`sheetsage2_refined.beats.encode_windows`) to `model.scores_from_windows(...)` to skip a second
encoder pass.

Options: `--top`, `--model` (a Hugging Face repo id or a local model folder), `--revision`, `--encoder-model`
(another SheetSage2-Refined folder or repo), `--device`, `--out-dir`, `--memory-fraction`.

## License

The code is MIT ([LICENSE](LICENSE)).  The weights, this model's and SheetSage2-Refined's, are
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only.
