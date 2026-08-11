import json
from collections import Counter

class Tokenizer:
    def __init__(self, pitch_vocab, dur_vocab, chord_vocab, genre_vocab):
        self.pitch_stoi = {p: i for i, p in enumerate(pitch_vocab)}
        self.pitch_itos = {i: p for i, p in enumerate(pitch_vocab)}

        self.duration_stoi = {d: i for i, d in enumerate(dur_vocab)}
        self.duration_itos = {i: d for i, d in enumerate(dur_vocab)}

        self.chord_stoi = {c: i for i, c in enumerate(chord_vocab)}
        self.chord_itos = {i: c for i, c in enumerate(chord_vocab)}

        self.genre_stoi = {g: i for i, g in enumerate(genre_vocab)}
        self.genre_itos = {i: g for i, g in enumerate(genre_vocab)}

    @classmethod
    def from_songs(cls, songs):
        pitch = ["<PAD>"] + sorted({n["pitch"] for s in songs for n in s["notes"]})

        duration_counts = Counter(n["dur"] for s in songs for n in s["notes"])
        duration = ["<PAD>"] + sorted([d for d, count in duration_counts.items() if count >= 10]) + ["<RARE_DUR>"]

        roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        qualities = ["maj", "min", "dim", "aug", "dom7", "min7", "maj7", "hdim7", "dim7", "oth"]
        chord = ["<PAD>"] + [f"{r}:{q}" for r in roots for q in qualities] + ["NC"]

        genre = ["<folk>"]
        return cls(pitch, duration, chord, genre)

    def encode(self, song):
        rare = self.duration_stoi["<RARE_DUR>"]
        pitch_ids = [self.pitch_stoi[n["pitch"]] for n in song["notes"]]
        dur_ids   = [self.duration_stoi.get(n["dur"], rare) for n in song["notes"]]
        chord_ids = [self.chord_stoi[n["chord"]] for n in song["notes"]]
        return pitch_ids, dur_ids, chord_ids

    def decode_chords(self, ids):
        return [self.chord_itos[i] for i in ids]


    def save(self, path):
        with open(path, "w") as f:
            json.dump({"pitch": self.pitch_itos, "duration": self.duration_itos, "chord": self.chord_itos, "genre": self.genre_itos}, f)
    
    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            d = json.load(f)
        pitch_vocab = cls._itos_to_vocab_list(d["pitch"])
        duration_vocab = cls._itos_to_vocab_list(d["duration"])
        chord_vocab = cls._itos_to_vocab_list(d["chord"])
        genre_vocab = cls._itos_to_vocab_list(d["genre"])
        return cls(pitch_vocab, duration_vocab, chord_vocab, genre_vocab)

    @staticmethod
    def _itos_to_vocab_list(itos):
        # convert a dict of {index: value} to a list of values sorted by index
        sorted_items = sorted(itos.items(), key=lambda pair: int(pair[0]))
        return [value for key, value in sorted_items]

def load_songs(path):
    songs = []
    with open(path, "r") as f:
        for line in f:
            songs.append(json.loads(line))
    return songs

if __name__ == "__main__": 
    songs = load_songs("data/processed/pairs.jsonl")  

    tokenizer = Tokenizer.from_songs(songs)
    print("Pitch vocab size:", len(tokenizer.pitch_stoi))
    print("Duration vocab:", tokenizer.duration_stoi)
    print("Chord vocab size:", len(tokenizer.chord_stoi))

    song = songs[0]
    pitch_ids, dur_ids, chord_ids = tokenizer.encode(song)

    original_chords = [n["chord"] for n in song["notes"]]
    decoded_chords = tokenizer.decode_chords(chord_ids)

    print("Original chords:", original_chords)
    print("Decoded chords:", decoded_chords)
    assert decoded_chords == original_chords, "Round trip failed!"
    print("Round trip OK")

    tokenizer.save("artifacts/tokenizer.json")
    reloaded = Tokenizer.load("artifacts/tokenizer.json")
    assert reloaded.chord_stoi == tokenizer.chord_stoi, "Save/load mismatch!"
    print("Save/load OK")
