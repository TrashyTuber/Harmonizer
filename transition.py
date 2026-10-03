import json
import torch

from tokenizer import Tokenizer

def build_transition_matrix(train_path, tokenizer):
    C = len(tokenizer.chord_stoi)
    counts = torch.ones(C, C)  

    with open(train_path) as f:
        for line in f:
            song = json.loads(line)
            _, _, chord_ids = tokenizer.encode(song)
            for c_prev, c_next in zip(chord_ids, chord_ids[1:]):
                counts[c_prev, c_next] += 1

    probs = counts / counts.sum(dim=1, keepdim=True)
    logT = probs.log()
    logT[:, tokenizer.chord_stoi["<PAD>"]] = float("-inf")
    return logT

if __name__ == "__main__":
    tokenizer = Tokenizer.load("artifacts/tokenizer.json")
    logT = build_transition_matrix("data/processed/train.jsonl", tokenizer)
    torch.save(logT, "artifacts/transition_matrix.pt")
    print(f"Saved {logT.shape} log-transition matrix")