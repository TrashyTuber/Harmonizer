"""Load a trained harmonizer from a checkpoint written by train.py.

Checkpoints are {"config": {"arch": ..., "model": {...}, ...}, "state_dict": ...}.
The original BiLSTM checkpoint (artifacts/model.pt) predates that format and
is a bare state_dict; it is loaded with the hyperparameters it was trained with.
"""

import torch

from model import build_model

LEGACY_CONFIG = {
    "arch": "lstm",
    "model": {"pitch_embed_dim": 128, "dur_embed_dim": 64, "hidden_dim": 256},
}


def load_model(path, tokenizer, device):
    """Return (model in eval mode, config dict) for the checkpoint at path."""
    ckpt = torch.load(path, map_location=device)
    if "state_dict" in ckpt:
        config, state_dict = ckpt["config"], ckpt["state_dict"]
    else:
        config, state_dict = LEGACY_CONFIG, ckpt

    model = build_model(config["arch"], config["model"], tokenizer).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config
