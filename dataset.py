import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 

from tokenizer import Tokenizer

class HarmonizerDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.songs = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.songs.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        song = self.songs[idx]
        pitch_ids, dur_ids, chord_ids = self.tokenizer.encode(song)
        return pitch_ids, dur_ids, chord_ids

    def collate_fn(self, batch):
        pitch_seqs, dur_seqs, chord_seqs = zip(*batch)

        pitch_padded = pad_sequence([torch.tensor(p) for p in pitch_seqs],
                                    batch_first=True, padding_value=self.tokenizer.pitch_stoi["<PAD>"])
        dur_padded = pad_sequence([torch.tensor(d) for d in dur_seqs],
                                batch_first=True, padding_value=self.tokenizer.duration_stoi["<PAD>"])
        chord_padded = pad_sequence([torch.tensor(c) for c in chord_seqs],
                                    batch_first=True, padding_value=self.tokenizer.chord_stoi["<PAD>"])

        lengths = torch.tensor([len(p) for p in pitch_seqs])
        return pitch_padded, dur_padded, chord_padded, lengths



if __name__ == "__main__":
    tokenizer = Tokenizer.from_songs([json.loads(l) for l in open("data/processed/train.jsonl")])
    dataset = HarmonizerDataset("data/processed/train.jsonl", tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)

    pitch_batch, dur_batch, chord_batch = next(iter(loader))
    print("Pitch batch shape:", pitch_batch.shape)
    print("Duration batch shape:", dur_batch.shape)
    print("Chord batch shape:", chord_batch.shape)