"""Print a parsed song as chord segments with melody note names.

Usage: uv run python scripts/show.py ashover1.mid [--jsonl data/processed/pairs.jsonl]
"""

import argparse
import json
from itertools import groupby
from pathlib import Path

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="MIDI filename as stored in the jsonl, e.g. ashover1.mid")
    ap.add_argument("--jsonl", type=Path, default=Path("data/processed/pairs.jsonl"))
    args = ap.parse_args()

    song = next(
        (json.loads(l) for l in open(args.jsonl) if json.loads(l)["file"] == args.file),
        None,
    )
    if song is None:
        raise SystemExit(f"{args.file} not found in {args.jsonl}")

    print(f"{song['file']} — {len(song['notes'])} notes\n")
    print(f"{'beat':>6}  {'chord':<8} melody (name/duration)")
    for chord, group in groupby(song["notes"], key=lambda n: n["chord"]):
        group = list(group)
        melody = "  ".join(f"{note_name(n['pitch'])}/{n['dur']:g}" for n in group)
        print(f"{group[0]['offset']:>6g}  {chord:<8} {melody}")


if __name__ == "__main__":
    main()
