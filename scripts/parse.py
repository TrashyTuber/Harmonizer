"""Parse Nottingham MIDI files into aligned (melody note, chord label) pairs.

Each MIDI file has two parts: part 0 is the melody (single notes), part 1 is
the accompaniment (block chords). For every melody note we record the chord
sounding at its onset, using the lead-sheet convention that a chord stays in
effect until the next one starts. Notes before the first chord get "NC".

Chord labels are root pitch-class + quality (e.g. "G:maj", "A:min", "D:dom7"),
matched against interval templates. Chords that fit no template become "oth".

Output: one JSON line per song in data/processed/pairs.jsonl.
"""

import argparse
import json
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

from music21 import converter, note as m21note, chord as m21chord

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Interval sets relative to the root, in semitones.
QUALITY_TEMPLATES = {
    (0, 4, 7): "maj",
    (0, 3, 7): "min",
    (0, 3, 6): "dim",
    (0, 4, 8): "aug",
    (0, 4, 7, 10): "dom7",
    (0, 3, 7, 10): "min7",
    (0, 4, 7, 11): "maj7",
    (0, 3, 6, 10): "hdim7",
    (0, 3, 6, 9): "dim7",
}


def chord_label(ch: m21chord.Chord) -> str:
    """Name a chord by trying every pitch class as a candidate root."""
    pcs = frozenset(p.pitchClass for p in ch.pitches)
    for root in pcs:
        intervals = tuple(sorted((pc - root) % 12 for pc in pcs))
        quality = QUALITY_TEMPLATES.get(intervals)
        if quality:
            return f"{PITCH_NAMES[root]}:{quality}"
    return f"{PITCH_NAMES[ch.root().pitchClass]}:oth"


def parse_file(path: Path) -> dict:
    try:
        score = converter.parse(path)
    except Exception as e:
        return {"file": path.name, "error": f"parse failure: {e}"}

    if len(score.parts) != 2:
        return {"file": path.name, "error": f"expected 2 parts, got {len(score.parts)}"}

    melody_part, chord_part = score.parts

    # Build the chord timeline: (onset, label), sorted by onset.
    timeline = []
    for el in chord_part.flatten().notes:
        if isinstance(el, m21chord.Chord) and len(el.pitches) >= 3:
            timeline.append((float(el.offset), chord_label(el)))
    timeline.sort()
    if not timeline:
        return {"file": path.name, "error": "no chords found in part 1"}
    onsets = [t for t, _ in timeline]

    notes = []
    for el in melody_part.flatten().notes:
        if isinstance(el, m21chord.Chord):
            el = el.notes[-1]  # rare double-stop in the melody: keep the top note
        if not isinstance(el, m21note.Note):
            continue
        t = float(el.offset)
        i = bisect_right(onsets, t) - 1  # most recent chord at or before t
        chord = timeline[i][1] if i >= 0 else "NC"
        notes.append(
            {
                "pitch": el.pitch.midi,
                "offset": t,
                "dur": float(el.quarterLength),
                "chord": chord,
            }
        )

    if not notes:
        return {"file": path.name, "error": "no melody notes found in part 0"}
    return {"file": path.name, "notes": notes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-dir", type=Path, default=Path("data/nottingham/MIDI"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="only parse N files (for testing)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = sorted(args.midi_dir.glob("*.mid"))[: args.limit]
    print(f"Parsing {len(paths)} files with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(parse_file, paths, chunksize=16))

    songs = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for song in songs:
            f.write(json.dumps(song) + "\n")

    label_counts = Counter(n["chord"] for s in songs for n in s["notes"])
    total_notes = sum(label_counts.values())

    print(f"\nParsed OK: {len(songs)}  Failed: {len(failures)}")
    for r in failures[:10]:
        print(f"  {r['file']}: {r['error']}")
    print(f"\nTotal melody notes: {total_notes}")
    print(f"Unique chord labels: {len(label_counts)}")
    print("\nTop 20 labels:")
    for label, count in label_counts.most_common(20):
        print(f"  {label:<8} {count:>7}  ({100 * count / total_notes:.1f}%)")


if __name__ == "__main__":
    main()
