"""Genres of whole songs from SheetSage2-Refined's frozen encoder.

    from tsumugi_genres import HeadGenres

    model = HeadGenres.load()                 # this model + SheetSage2-Refined's encoder, from Hugging Face
    for g in model.predict("song.mp3")[:10]:  # every genre, highest score first
        print(g["id"], g["name"], round(g["score"], 3))

The song goes through the encoder in 300 s windows (``sheetsage2_refined.beats.encode_windows``, as for its heads);
the states of a few layers are averaged per second and read by an ensemble of genre heads (``head.GenreHead``),
whose probabilities are averaged.  A song can have several genres: each score is that genre's own probability.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .head import GenreHead

DEFAULT_MODEL = "anime-song/Tsumugi-Genres"
FRAME_RATE = 25  # the encoder's frames per second


def model_folder(model: Path | str, revision: str | None = None) -> Path:
    """A local model folder as it is, or the Hugging Face model repo (downloaded into the Hub cache)."""
    if Path(model).is_dir():
        return Path(model)
    if isinstance(model, Path) or str(model).count("/") != 1:
        raise FileNotFoundError(f"no model folder {model} (a Hugging Face repo id looks like owner/name)")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model), revision=revision, allow_patterns=["config.json", "*.safetensors"]))


def encoder_file(model: str, revision: str | None) -> Path:
    """SheetSage2's weights in a SheetSage2-Refined folder or Hugging Face repo (only that file is downloaded)."""
    if Path(model).is_dir():
        config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
        return Path(model) / config["base_weights"]
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(model, "config.json", revision=revision)).read_text(encoding="utf-8"))
    return Path(hf_hub_download(model, config["base_weights"], revision=revision))


def per_second(windows, layers: list[int]) -> torch.Tensor:
    """Encoder windows (``(states, keep)`` as ``encode_windows`` yields them) -> ``[layers, seconds, 1024]``: each
    layer's frames averaged per second (the last second over the frames it has), rounded to float16 as in training."""
    kept = [[states[i][0, :keep] for i in layers] for states, keep in windows]
    out = []
    for j in range(len(layers)):
        x = torch.cat([w[j] for w in kept]).float()  # [frames, 1024]
        frames = x.shape[0]
        seconds = math.ceil(frames / FRAME_RATE)
        pad = seconds * FRAME_RATE - frames
        weight = torch.ones(frames, device=x.device)
        if pad:
            x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
            weight = torch.cat([weight, weight.new_zeros(pad)])
        weight = weight.view(seconds, FRAME_RATE)
        out.append((x.view(seconds, FRAME_RATE, -1) * weight[..., None]).sum(1) / weight.sum(1)[:, None])
    return torch.stack(out).half().float()


class HeadGenres:
    def __init__(self, encoder, heads: list[GenreHead], config: dict, device: torch.device):
        self.encoder, self.heads, self.config, self.device = encoder, heads, config, device
        self.layers = config["layers"]
        self.genres = config["genres"]  # [{"id", "name"}] in output order

    @classmethod
    def load(cls, model: Path | str = DEFAULT_MODEL, device: str | None = None, revision: str | None = None,
             encoder_model: str | None = None, encoder_revision: str | None = None) -> "HeadGenres":
        """This model (a local folder or Hugging Face repo id, at ``revision``) and SheetSage2-Refined's encoder
        (``encoder_model``: another folder or repo; default: the one ``config.json`` names) on ``device``
        (default: CUDA if there is one)."""
        from sheetsage2_refined.beats import load_encoder

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        folder = model_folder(model, revision)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        enc = config["encoder"]
        if encoder_model is None:
            encoder_model, encoder_revision = enc["model"], encoder_revision or enc.get("revision")
        encoder = load_encoder(encoder_file(encoder_model, encoder_revision), device)
        state = load_file(str(folder / config["weights"]), device="cpu")
        heads = []
        for i in range(config["members"]):
            head = GenreHead(len(config["layers"]), len(config["genres"]), d=config["d"], heads=config["heads"])
            prefix = f"member{i}."
            head.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
            heads.append(head.to(device).eval())
        return cls(encoder, heads, config, device)

    @torch.no_grad()
    def scores_from_windows(self, windows) -> np.ndarray:
        """Each genre's probability ``[genres]`` from encoder windows already computed (``(states, keep)`` as
        ``sheetsage2_refined.beats.encode_windows`` yields them; e.g. to share one encoder pass with its heads)."""
        x = per_second(windows, self.layers)[None].to(self.device)
        return np.mean([torch.sigmoid(h(x).float())[0].cpu().numpy() for h in self.heads], 0)

    @torch.no_grad()
    def scores(self, audio: Path | str | torch.Tensor) -> np.ndarray:
        """Each genre's probability ``[genres]`` for an audio file or a mono 24 kHz waveform."""
        from sheetsage2_refined.beats import encode_windows
        from sheetsage2_refined.transcribe import load_waveform

        wave = audio if isinstance(audio, torch.Tensor) else load_waveform(Path(audio))
        return self.scores_from_windows(encode_windows(self.encoder, wave, self.device))

    def predict(self, audio: Path | str | torch.Tensor) -> list[dict]:
        """Every genre as ``{"id", "name", "score"}``, highest score first."""
        s = self.scores(audio)
        return [dict(self.genres[i], score=float(s[i])) for i in np.argsort(-s)]
