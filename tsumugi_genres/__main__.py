"""Genres of audio files: python -m tsumugi_genres AUDIO [AUDIO ...] prints each song's top genres and writes
OUT_DIR/<stem>.genres.json (every genre with its id, name and score, highest first)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .genres import DEFAULT_MODEL, HeadGenres


def main() -> None:
    parser = argparse.ArgumentParser(prog="tsumugi-genres", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=10, help="how many genres to print")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local model folder")
    parser.add_argument("--revision", help="the Hugging Face repo's branch, tag or commit")
    parser.add_argument("--encoder-model", help="SheetSage2-Refined folder or repo for the encoder (default: the one config.json names)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "genres")
    parser.add_argument("--memory-fraction", type=float, default=0.6, help="cap on this process's share of GPU memory")
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.device(args.device).index or 0)
    model = HeadGenres.load(args.model, args.device, args.revision, args.encoder_model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for audio in args.audio:
        started = time.perf_counter()
        genres = model.predict(audio)
        (args.out_dir / f"{audio.stem}.genres.json").write_text(
            json.dumps({"audio": str(audio), "genres": genres}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{audio.name} ({time.perf_counter() - started:.1f} s)", flush=True)
        for g in genres[: args.top]:
            print(f"  {g['score']:.3f}  {g['name']}", flush=True)


if __name__ == "__main__":
    main()
