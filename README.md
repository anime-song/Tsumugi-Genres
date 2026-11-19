# Tsumugi-Genres

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Tsumugi--Genres-yellow)](https://huggingface.co/anime-song/Tsumugi-Genres)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)

Genres of whole songs: a score for each of 135 genres, so a song can have several (funk + soul + disco, or
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
