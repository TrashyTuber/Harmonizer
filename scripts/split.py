"""Split parsed songs into train/val/test, then transpose the training split.

Songs are shuffled with a fixed seed and split 80/10/10 BEFORE any
transposition, so no melody appears in two splits in different keys.
The training split is then augmented into all 12 keys (shifts of -5..+6
semitones, keeping melodies centered in their original register). Val and
test stay in their original keys so metrics describe real songs and stay
comparable across augmentation ablations.

Outputs: data/processed/{train,val,test}.jsonl
"""

import argparse
import json
import random
from pathlib import Path

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_INDEX = {name: i for i, name in enumerate(PITCH_NAMES)}


def transpose_chord(label: str, k: int) -> str:
    if label == "NC":
        return label
    root, quality = label.split(":")
    return f"{PITCH_NAMES[(PITCH_INDEX[root] + k) % 12]}:{quality}"


def transpose_song(song: dict, k: int) -> dict:
    notes = [
        dict(n, pitch=n["pitch"] + k, chord=transpose_chord(n["chord"], k))
        for n in song["notes"]
    ]
    return {"file": song["file"], "transposition": k, "notes": notes}


def write_jsonl(songs: list, path: Path) -> None:
    with open(path, "w") as f:
        for song in songs:
            f.write(json.dumps(song) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    songs = [json.loads(line) for line in open(args.pairs)]
    random.Random(args.seed).shuffle(songs)

    n = len(songs)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    splits = {
        "train": songs[:n_train],
        "val": songs[n_train : n_train + n_val],
        "test": songs[n_train + n_val :],
    }

    splits["train"] = [
        transpose_song(song, k) for song in splits["train"] for k in range(-5, 7)
    ]

    for name, split_songs in splits.items():
        write_jsonl(split_songs, args.out_dir / f"{name}.jsonl")
        n_notes = sum(len(s["notes"]) for s in split_songs)
        pitches = [n["pitch"] for s in split_songs for n in s["notes"]]
        print(
            f"{name:<5} {len(split_songs):>5} sequences  {n_notes:>7} notes  "
            f"pitch range {min(pitches)}-{max(pitches)}"
        )


if __name__ == "__main__":
    main()
