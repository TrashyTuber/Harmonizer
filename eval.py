import argparse
import json
import torch

from checkpoint import load_model
from tokenizer import Tokenizer
from decode import transition_viterbi_decode, viterbi_decode

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="artifacts/lstm.pt")
parser.add_argument("--split", default="data/processed/val.jsonl")
parser.add_argument("--decoder", choices=["argmax", "uniform", "transition"], default="argmax")
parser.add_argument("--switch-penalty", type=float, default=0.25)
parser.add_argument("--lam", type=float, default=0.05)
args = parser.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = Tokenizer.load("artifacts/tokenizer.json")
model, config = load_model(args.checkpoint, tokenizer, device)

FUNCTION = {
    "major": {
        0: "T",   # I
        2: "PD",  # ii
        4: None,  # iii
        5: "PD",  # IV
        7: "D",   # V
        9: "T",   # vi
        11: "D",  # vii°
    },
    "minor": {
        0: "T",   # i
        2: "PD",  # ii°
        3: "T",   # III
        5: "PD",  # iv
        7: "D",   # v/V
        8: None,  # VI
        10: None, # VII
    },
}

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_TO_PC = {r: i for i, r in enumerate(ROOTS)}

def chord_root(label):
    if label == "NC":
        return "NC"
    return label.split(":")[0]

def count_changes(labels):
    return sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])

def estimate_key(chord_labels):
    # chord_labels: list of strings like "G:maj", "D:dom7", skip "NC"
    real_chords = [c for c in chord_labels if c != "NC"]
    if not real_chords:
        return (0, "major")  # degenerate fallback

    last_root_pc = ROOT_TO_PC[real_chords[-1].split(":")[0]]

    best_score = -1
    best_key = (0, "major")

    for tonic_pc in range(12):
        for mode in ["major", "minor"]:
            diatonic_intervals = set(FUNCTION[mode].keys())
            in_key_count = 0
            for label in real_chords:
                root = label.split(":")[0]
                root_pc = ROOT_TO_PC[root]
                interval = (root_pc - tonic_pc) % 12
                if interval in diatonic_intervals:
                    in_key_count += 1
            coverage = in_key_count / len(real_chords)

            cadence_bonus = 0.3 if last_root_pc == tonic_pc else 0.0
            score = coverage + cadence_bonus

            if score > best_score:
                best_score = score
                best_key = (tonic_pc, mode)

    return best_key

songs = [json.loads(l) for l in open(args.split)]

all_preds = []
all_targets = []

exact_count = 0
root_match_count = 0
functional_match_count = 0
total_notes = 0
pred_changes = 0
true_changes = 0

if args.decoder == "transition":
    logT = torch.load("artifacts/transition_matrix.pt")

with torch.no_grad():
    for song in songs:
        pitch_ids, dur_ids, chord_ids = tokenizer.encode(song)
        pitch_tensor = torch.tensor(pitch_ids).unsqueeze(0).to(device)
        dur_tensor = torch.tensor(dur_ids).unsqueeze(0).to(device)

        chord_pred = model(pitch_tensor, dur_tensor)
        logits = chord_pred.squeeze(0).cpu()
        logits[:, tokenizer.chord_stoi["<PAD>"]] = float("-inf")
        if args.decoder == "transition":
            pred_ids = transition_viterbi_decode(logits, logT, args.lam)
        elif args.decoder == "uniform":
            pred_ids = viterbi_decode(logits, args.switch_penalty)
        else:
            pred_ids = logits.argmax(-1).tolist()
        preds = torch.tensor(pred_ids).to(device)
        target = torch.tensor(chord_ids).to(device)

        all_preds.append(preds)
        all_targets.append(target)

        pred_labels = tokenizer.decode_chords(preds.tolist())
        true_labels = [n["chord"] for n in song["notes"]]
        key = estimate_key(true_labels)

        pred_changes += count_changes(pred_labels)
        true_changes += count_changes(true_labels)

        for pred_label, true_label in zip(pred_labels, true_labels):
            total_notes += 1

            exact_count += int(pred_label == true_label)

            pred_root = chord_root(pred_label)
            true_root = chord_root(true_label)
            root_match_count += int(pred_root == true_root)

            if pred_root == "NC" or true_root == "NC":
                func_match = (pred_root == true_root)
            else:
                pred_interval = (ROOT_TO_PC[pred_root] - key[0]) % 12
                true_interval = (ROOT_TO_PC[true_root] - key[0]) % 12
                pred_func = FUNCTION[key[1]].get(pred_interval)
                true_func = FUNCTION[key[1]].get(true_interval)

                if pred_func is None or true_func is None:
                    func_match = (pred_root == true_root)
                else:
                    func_match = (pred_func == true_func)

            functional_match_count += int(func_match)

        
all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

avg_pred_changes = pred_changes / len(songs)
avg_true_changes = true_changes / len(songs)

micro_accuracy = (all_preds == all_targets).float().mean().item()
macro_accuracies = []
for chord_idx in range(len(tokenizer.chord_stoi)):
    mask = all_targets == chord_idx
    if mask.sum() > 0:
        class_accuracy = (all_preds[mask] == all_targets[mask]).float().mean().item()
        macro_accuracies.append(class_accuracy)
macro_accuracy = sum(macro_accuracies) / len(macro_accuracies) if macro_accuracies else 0.0

note_exact_accuracy = exact_count / total_notes # Should be equal to micro_accuracy
root_accuracy = root_match_count / total_notes
functional_accuracy = functional_match_count / total_notes

decoder_desc = {
    "argmax": "argmax",
    "uniform": f"uniform penalty {args.switch_penalty}",
    "transition": f"transition lam {args.lam}",
}[args.decoder]
print(f"[{config['arch']} | {args.checkpoint} | {args.split} | {decoder_desc}]")
print(f"micro_accuracy={micro_accuracy:.4f}, macro_accuracy={macro_accuracy:.4f}, root_accuracy={root_accuracy:.4f}, functional_accuracy={functional_accuracy:.4f}")
print(f"avg_pred_changes={avg_pred_changes:.4f}, avg_true_changes={avg_true_changes:.4f}")

