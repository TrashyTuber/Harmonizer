"""Gradio demo: type or upload a melody and hear it harmonized by the BiLSTM.

Run locally with: uv run python app.py
"""

import tempfile
from fractions import Fraction
from pathlib import Path

import gradio as gr
import numpy as np
import pretty_midi
import torch

from checkpoint import load_model
from decode import transition_viterbi_decode
from render import build_midi, chord_spans
from tokenizer import Tokenizer

ARTIFACTS = Path(__file__).parent / "artifacts"
LAM = 0.05
MAX_NOTES = 2000
ONSET_GRID = 12  # uploaded onsets snap to 1/12 beat, which covers 16ths and triplets
SAMPLE_RATE = 22050

SMOOTHED = "Smoothed (learned chord transitions)"
RAW = "Raw (best chord for each note)"

DEVICE = torch.device("cpu")
TOKENIZER = Tokenizer.load(ARTIFACTS / "tokenizer.json")
MODEL, _ = load_model(ARTIFACTS / "lstm.pt", TOKENIZER, DEVICE)
LOG_T = torch.load(ARTIFACTS / "transition_matrix.pt")
PAD_ID = TOKENIZER.chord_stoi["<PAD>"]

KNOWN_DURS = [d for d in TOKENIZER.duration_stoi if isinstance(d, float)]
PITCHES = sorted(p for p in TOKENIZER.pitch_stoi if isinstance(p, int))
LOWEST, HIGHEST = PITCHES[0], PITCHES[-1]

FLAT_NAMES = {"A#": "B♭", "D#": "E♭", "G#": "A♭"}
QUALITY_SUFFIX = {
    "maj": "", "min": "m", "dim": "°", "aug": "+", "dom7": "7",
    "min7": "m7", "maj7": "maj7", "hdim7": "ø7", "dim7": "°7", "oth": "(?)",
}

EXAMPLES = [
    ["C4 C4 G4 G4 | A4 A4 G4:2 | F4 F4 E4 E4 | D4 D4 C4:2 | "
     "G4 G4 F4 F4 | E4 E4 D4:2 | G4 G4 F4 F4 | E4 E4 D4:2 | "
     "C4 C4 G4 G4 | A4 A4 G4:2 | F4 F4 E4 E4 | D4 D4 C4:2"],
    ["B4 A4 G4 A4 | B4 B4 B4:2 | A4 A4 A4:2 | B4 D5 D5:2 | "
     "B4 A4 G4 A4 | B4 B4 B4 B4 | A4 A4 B4 A4 | G4:4"],
    ["F#4 F#4 G4 A4 | A4 G4 F#4 E4 | D4 D4 E4 F#4 | F#4:1.5 E4:0.5 E4:2 | "
     "F#4 F#4 G4 A4 | A4 G4 F#4 E4 | D4 D4 E4 F#4 | E4:1.5 D4:0.5 D4:2"],
]
EXAMPLE_LABELS = [
    "Twinkle, Twinkle, Little Star (C major)",
    "Mary Had a Little Lamb (G major)",
    "Ode to Joy (D major)",
]

INTRO = """
# Harmonizer

Give it a melody and a neural network writes the chords. The model is a
bidirectional LSTM trained on about 800 British and Irish folk tunes, so it
harmonizes in a folk style. It was trained on every tune in all 12 keys, so
your melody can be in any key.
"""

NOTATION_HELP = """
Write notes separated by spaces: pitch, octave, and an optional length in
beats after a colon (default 1 beat). `C4` is middle C. Use `#` or `b` for
accidentals, `R` for a rest, and `|` for bar lines (they're ignored).
Lengths can be decimals or fractions, so `C4:1/3` is an eighth-note triplet.
"""


def snap_duration(dur):
    """Snap to the nearest length the model knows; leave far-off values as rare."""
    nearest = min(KNOWN_DURS, key=lambda d: abs(d - dur))
    return nearest if abs(nearest - dur) <= 0.06 else dur


def chord_name(label):
    if label == "NC":
        return "N.C."
    root, quality = label.split(":")
    return FLAT_NAMES.get(root, root) + QUALITY_SUFFIX[quality]


def note_name(pitch):
    return pretty_midi.note_number_to_name(pitch)


def parse_text(text):
    notes, beat = [], 0.0
    for token in text.replace("|", " ").split():
        name, _, length = token.partition(":")
        try:
            dur = float(Fraction(length)) if length else 1.0
        except (ValueError, ZeroDivisionError):
            raise gr.Error(f"Couldn't read the length in '{token}'. Try something like C4:0.5 or C4:1/3.")
        if dur <= 0:
            raise gr.Error(f"'{token}' has a length of zero or less.")
        dur = snap_duration(dur)
        if name.upper() in ("R", "REST"):
            beat += dur
            continue
        try:
            pitch = pretty_midi.note_name_to_number(name)
        except Exception:
            raise gr.Error(f"Couldn't read the note '{token}'. Notes look like C4, F#4 or Bb3.")
        notes.append({"pitch": pitch, "offset": beat, "dur": dur, "chord": "NC"})
        beat += dur
    return notes


def parse_midi(path):
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        raise gr.Error("That file couldn't be read as MIDI.")
    track = next((i for i in pm.instruments if not i.is_drum and i.notes), None)
    if track is None:
        raise gr.Error("The MIDI file has no melody notes.")

    def to_beats(seconds):
        return pm.time_to_tick(seconds) / pm.resolution

    # Keep the highest note at each onset, so chords in the track reduce to a melody line.
    top = {}
    for n in track.notes:
        onset = round(to_beats(n.start) * ONSET_GRID) / ONSET_GRID
        if onset not in top or n.pitch > top[onset].pitch:
            top[onset] = n
    onsets = sorted(top)
    first_bar = np.floor(onsets[0])

    notes = []
    for i, onset in enumerate(onsets):
        length = to_beats(top[onset].end) - onset
        if i + 1 < len(onsets):
            length = min(length, onsets[i + 1] - onset)
        if length <= 0:
            continue
        notes.append({
            "pitch": top[onset].pitch,
            "offset": float(onset - first_bar),
            "dur": snap_duration(length),
            "chord": "NC",
        })
    return notes


def octave_shift(notes):
    """Octave shift that moves the melody into the pitch range the model was trained on."""
    low = min(n["pitch"] for n in notes)
    high = max(n["pitch"] for n in notes)
    for shift in sorted(range(-48, 49, 12), key=abs):
        if low + shift >= LOWEST and high + shift <= HIGHEST:
            return shift
    raise gr.Error(
        f"The melody spans {note_name(low)} to {note_name(high)}, which is too wide. "
        f"The model handles melodies within {note_name(LOWEST)} to {note_name(HIGHEST)}."
    )


def fill_opening_rest(labels):
    """Give the opening N.C. notes the first real chord.

    Most training tunes start with a pickup labeled N.C., and the model has no
    beat-position input to tell a pickup from a downbeat start, so it predicts
    N.C. at the start of nearly every melody.
    """
    first_chord = next((label for label in labels if label != "NC"), None)
    if first_chord is None:
        return labels
    opening = next(i for i, label in enumerate(labels) if label != "NC")
    return [first_chord] * opening + labels[opening:]


def chord_chart(notes, labels):
    rows = ["| Beat | Chord | Melody |", "|---|---|---|"]
    spans = chord_spans(notes, labels)
    for start, end, label in spans:
        melody = " ".join(note_name(n["pitch"]) for n in notes if start <= n["offset"] < end)
        rows.append(f"| {start + 1:g} | **{chord_name(label)}** | {melody} |")
    return "\n".join(rows)


def harmonize(notes, decoder, tempo):
    if not notes:
        raise gr.Error("No notes found. Type a melody or upload a MIDI file first.")
    if len(notes) > MAX_NOTES:
        raise gr.Error(f"That melody has {len(notes)} notes; the demo handles up to {MAX_NOTES}.")

    shift = octave_shift(notes)
    model_notes = [dict(n, pitch=n["pitch"] + shift) for n in notes]
    pitch_ids, dur_ids, _ = TOKENIZER.encode({"notes": model_notes})
    with torch.no_grad():
        logits = MODEL(torch.tensor(pitch_ids)[None], torch.tensor(dur_ids)[None])[0]
    logits[:, PAD_ID] = float("-inf")
    if decoder == SMOOTHED:
        chord_ids = transition_viterbi_decode(logits, LOG_T, LAM)
    else:
        chord_ids = logits.argmax(-1).tolist()
    labels = fill_opening_rest(TOKENIZER.decode_chords(chord_ids))

    pm = build_midi(notes, labels, tempo)
    audio = pm.fluidsynth(fs=SAMPLE_RATE)
    audio = (audio / max(np.abs(audio).max(), 1e-9) * 0.9 * 32767).astype(np.int16)
    midi_file = tempfile.NamedTemporaryFile(prefix="harmonized_", suffix=".mid", delete=False)
    pm.write(midi_file.name)

    rare = sum(1 for n in notes if n["dur"] not in KNOWN_DURS)
    status = [f"{len(notes)} notes, {len(chord_spans(notes, labels))} chord changes."]
    if shift:
        status.append(
            f"Moved {abs(shift) // 12} octave(s) {'up' if shift > 0 else 'down'} for the model; "
            "playback uses your original octave."
        )
    if rare:
        status.append(f"{rare} note length(s) were unusual and treated as a generic length.")
    return (SAMPLE_RATE, audio), chord_chart(notes, labels), midi_file.name, " ".join(status)


def from_text(text, decoder, tempo):
    return harmonize(parse_text(text or ""), decoder, tempo)


def from_midi(path, decoder, tempo):
    if path is None:
        raise gr.Error("Upload a MIDI file first.")
    return harmonize(parse_midi(path), decoder, tempo)


with gr.Blocks(title="Harmonizer") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            with gr.Tab("Type notes"):
                text = gr.Textbox(label="Melody", lines=4, placeholder="D4 E4:0.5 F#4:0.5 G4:2 R:1 A4")
                gr.Markdown(NOTATION_HELP)
                text_button = gr.Button("Harmonize", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=text, example_labels=EXAMPLE_LABELS)
            with gr.Tab("Upload MIDI"):
                upload = gr.File(label="MIDI file", file_types=[".mid", ".midi"], type="filepath")
                gr.Markdown("The first instrument track is used; where notes overlap, the highest one is kept.")
                midi_button = gr.Button("Harmonize", variant="primary")
            decoder = gr.Radio([SMOOTHED, RAW], value=SMOOTHED, label="Decoding")
            tempo = gr.Slider(60, 180, value=110, step=5, label="Playback tempo (BPM)")
        with gr.Column():
            audio = gr.Audio(label="Harmonization", type="numpy")
            status = gr.Markdown()
            midi_out = gr.File(label="Download MIDI")
            chart = gr.Markdown()

    outputs = [audio, chart, midi_out, status]
    text_button.click(from_text, [text, decoder, tempo], outputs)
    midi_button.click(from_midi, [upload, decoder, tempo], outputs)


if __name__ == "__main__":
    demo.launch()
