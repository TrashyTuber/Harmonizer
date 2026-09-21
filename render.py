"""Render a song's melody plus predicted and ground-truth chords as MIDI.

Writes two files to artifacts/renders/: <song>_pred.mid and <song>_truth.mid,
each containing the melody line over sustained accompaniment chords, so the
two harmonizations can be compared by ear.

Usage:
    uv run python render.py                  # first song in the val split
    uv run python render.py jigs102.mid      # a specific song
    uv run python render.py --list           # show available songs
"""

import argparse
import json
from pathlib import Path

import pretty_midi
import torch

from decode import viterbi_decode
from model import LSTMHarmonizer
from tokenizer import Tokenizer

TEMPO = 110  # BPM; folk tunes sit comfortably here
MELODY_PROGRAM = 73  # flute
CHORD_PROGRAM = 24  # nylon guitar
CHORD_LOWEST_ROOT = 45  # roots voiced in A2..G#3, below the melody

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_TO_PC = {r: i for i, r in enumerate(ROOTS)}

# Sevenths use shell voicings (root/3rd/7th, no 5th) to stay out of the
# melody's way; triads stay closed but low.
QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "dom7": (0, 4, 10),
    "min7": (0, 3, 10),
    "maj7": (0, 4, 11),
    "hdim7": (0, 3, 10),
    "dim7": (0, 3, 6, 9),
    "oth": (0, 7),  # unknown quality: play a bare root + fifth
}


def chord_pitches(label: str) -> list:
    if label in ("NC", "<PAD>"):
        return []
    root, quality = label.split(":")
    # place the root in the octave starting at CHORD_LOWEST_ROOT, preserving
    # its pitch class (root_pitch % 12 == ROOT_TO_PC[root])
    root_pitch = CHORD_LOWEST_ROOT + (ROOT_TO_PC[root] - CHORD_LOWEST_ROOT) % 12
    return [root_pitch + iv for iv in QUALITY_INTERVALS[quality]]


def chord_spans(notes: list, labels: list) -> list:
    """Merge consecutive same-label notes into (start_beat, end_beat, label)."""
    spans = []
    for note, label in zip(notes, labels):
        if spans and spans[-1][2] == label:
            spans[-1][1] = note["offset"] + note["dur"]
        else:
            if spans:
                spans[-1][1] = note["offset"]  # previous span ends where this begins
            spans.append([note["offset"], note["offset"] + note["dur"], label])
    return spans


def build_midi(notes: list, labels: list) -> pretty_midi.PrettyMIDI:
    sec = 60.0 / TEMPO  # seconds per quarter note
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO)

    melody = pretty_midi.Instrument(program=MELODY_PROGRAM, name="melody")
    for n in notes:
        melody.notes.append(
            pretty_midi.Note(
                velocity=95,
                pitch=n["pitch"],
                start=n["offset"] * sec,
                end=(n["offset"] + n["dur"]) * sec,
            )
        )

    chords = pretty_midi.Instrument(program=CHORD_PROGRAM, name="chords")
    for start, end, label in chord_spans(notes, labels):
        for pitch in chord_pitches(label):
            chords.notes.append(
                pretty_midi.Note(velocity=70, pitch=pitch, start=start * sec, end=end * sec)
            )

    pm.instruments.extend([melody, chords])
    return pm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=None, help="song filename within the split")
    ap.add_argument("--split", type=Path, default=Path("data/processed/val.jsonl"))
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/renders"))
    ap.add_argument("--list", action="store_true", help="list songs in the split and exit")
    ap.add_argument(
        "--switch-penalty",
        type=float,
        default=None,
        help="decode with Viterbi smoothing at this penalty instead of per-note argmax",
    )
    args = ap.parse_args()

    songs = [json.loads(l) for l in open(args.split)]
    if args.list:
        for s in songs:
            print(s["file"])
        return

    song = songs[0] if args.file is None else next(
        (s for s in songs if s["file"] == args.file), None
    )
    if song is None:
        raise SystemExit(f"{args.file} not found in {args.split} (try --list)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = Tokenizer.load("artifacts/tokenizer.json")
    model = LSTMHarmonizer(
        pitch_vocab_size=len(tokenizer.pitch_stoi),
        dur_vocab_size=len(tokenizer.duration_stoi),
        chord_vocab_size=len(tokenizer.chord_stoi),
        pitch_embed_dim=128,
        dur_embed_dim=64,
        hidden_dim=256,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    pitch_ids, dur_ids, _ = tokenizer.encode(song)
    with torch.no_grad():
        logits = model(
            torch.tensor(pitch_ids).unsqueeze(0).to(device),
            torch.tensor(dur_ids).unsqueeze(0).to(device),
        ).squeeze(0).cpu()
    if args.switch_penalty is not None:
        pred_ids = viterbi_decode(logits, args.switch_penalty)
        pred_name = f"{song['file'].removesuffix('.mid')}_pred_viterbi.mid"
    else:
        pred_ids = logits.argmax(-1).tolist()
        pred_name = f"{song['file'].removesuffix('.mid')}_pred.mid"
    pred_labels = tokenizer.decode_chords(pred_ids)
    true_labels = [n["chord"] for n in song["notes"]]

    stem = song["file"].removesuffix(".mid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_midi(song["notes"], pred_labels).write(str(args.out_dir / pred_name))
    build_midi(song["notes"], true_labels).write(str(args.out_dir / f"{stem}_truth.mid"))

    print(f"{song['file']} — model vs truth by chord segment:\n")
    print(f"{'beat':>6}  {'truth':<8} {'pred':<8}")
    true_spans = chord_spans(song["notes"], true_labels)
    pred_at = {n["offset"]: p for n, p in zip(song["notes"], pred_labels)}
    for start, _, label in true_spans:
        pred = pred_at.get(start, "?")
        marker = "" if pred == label else "  <-- differs"
        print(f"{start:>6g}  {label:<8} {pred:<8}{marker}")

    print(f"\nWrote {args.out_dir}/{pred_name} and {stem}_truth.mid")


if __name__ == "__main__":
    main()
