---
license: cc-by-nc-4.0
tags:
- audio
- music
- music-genre-classification
- multi-label
---

# Tsumugi-Genres

Whole songs in, genres out: a score for each of **137 genres**, several per song (funk + soul + disco, or
pop + shibuya-kei + jazz).  The 137 include 13 broad genres (pop, rock, electronic, jazz, metal, hip hop, ...)
next to the finer ones.

Tsumugi-Genres is six small genre heads (3.4 MB each) on the frozen encoder of
[SheetSage2-Refined](https://huggingface.co/anime-song/SheetSage2-Refined), which it loads as a dependency.

- Code (MIT): https://github.com/anime-song/Tsumugi-Genres
- Encoder: [anime-song/SheetSage2-Refined](https://huggingface.co/anime-song/SheetSage2-Refined) v0.5.2
- Weights: CC BY-NC 4.0, non-commercial use only

## Quick start

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/anime-song/Tsumugi-Genres
tsumugi-genres song.mp3                 # top 10 genres; every genre in outputs/genres/song.genres.json
```

```python
from tsumugi_genres import HeadGenres

model = HeadGenres.load()
for genre in model.predict("song.mp3")[:5]:
    print(genre["id"], genre["name"], round(genre["score"], 3))
```

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

## Model

- **Input:** the whole song (no length limit), through SheetSage2's encoder in 300 s windows.  The states of
  layers 10, 12, 14 and 16 are averaged per second.
- **Head:** one learnt query per genre attends over the song's seconds (one attention layer, width 256), and
  reads that genre's score.
- **Ensemble:** six heads (two dropout settings × three seeds, each trained on a different 90 % of the songs);
  the released score is the mean of their probabilities.
- **Output:** each genre's probability on its own (sigmoid), not a distribution over genres.  The scores are best
  read as a ranking; 0.5 is a rough cut for "this genre is there".

## Training data

About 9,400 songs, many of them Japanese pop, anime and game songs.  Genres are defined by sound: a song made from
several styles carries all of them.  Labels come from genre references checked song by song, and from hand checks (since v0.2.0 the hand-checked songs
are trained on too).

## Results

5-fold cross-validation of one head of the same recipe on the same songs (the released model averages six):

| | mAP | ROC-AUC |
|---|---|---|
| Fine and broad genres with at least 5 test songs (123) | 0.793 | 0.993 |
| Broad genres (13) | 0.950 | 0.994 |
| Hand-checked songs (56 genres) | 0.545 | |

v0.1.0 reported 0.800 over 121 genres; the test songs have changed since (two genres and about 200 songs, many of
them hard electronic ones), so the two are not directly comparable.  On v0.2.0's songs, v0.1.0's recipe scores
0.793 and 0.487 (hand-checked).

## Genres

acid jazz, afrobeats, alternative rock, ambient, ballad, baroque, bebop, big band, black metal, bluegrass, blues,
blues rock, bolero, boogie, boom bap, bossa nova, breakbeat, breakcore, bubblegum pop, celtic, celtic fantasy,
chanson, chiptune, city pop, classical, country, cumbia, dance pop, dancehall, dark ambient, death metal, denpa,
disco, doo-wop, doom metal, dream pop, drill, drone, drum and bass, dub, dubstep, easy listening, electro house,
electro swing, electronic, emo, emo rap, enka, eurobeat, eurodance, flamenco, folk, folk metal, folk rock, free jazz,
french pop, funk, funk carioca, future bass, garage rock, gospel, grunge, hard rock, hardcore punk,
hardcore techno, hardstyle, hip hop, house, hyperpop, indie folk, indie pop, indie rock, industrial metal, jazz,
jazz fusion, jazz pop, jersey club, jump blues, kawaii future bass, kawaii metal, kayokyoku, latin, latin jazz, latin pop,
math rock, merengue, metal, metalcore, neo city pop, neo soul, neo-acoustic, new wave, nu metal, opera,
orchestral pop, phonk, polka, pop, pop punk, pop rock, post-hardcore, post-punk, post-rock, power metal,
progressive metal, progressive rock, psychedelic rock, psytrance, punk, r&b, ragtime, reggae, reggaeton, rock,
salsa, samba, shibuya-kei, shoegaze, show tunes, ska, ska punk, soft rock, sophisti-pop, soul, sunshine pop,
symphonic metal, synthpop, synthwave, tango, techno, thrash metal, trance, trap, trip hop, uk garage, vaporwave,
vocal jazz

The ids and the output order are in `config.json`.

## License

The weights (`genres.safetensors`) are CC BY-NC 4.0; SheetSage2-Refined's weights, which the model loads, are
under their own license (CC BY-NC 4.0).  The code is MIT.
