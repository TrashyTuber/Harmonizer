import json

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from tokenizer import Tokenizer
from dataset import HarmonizerDataset
from model import build_model

config = {
    "arch": "transformer",
    "lr": 1e-3,
    "batch_size": 32,
    "max_len": 512,
    "num_epochs": 30,
    "model":  {
        "lstm": {
            "pitch_embed_dim": 128, "dur_embed_dim": 64,
            "hidden_dim": 256, "num_layers": 2, "dropout": 0.3,
        },
        "transformer": {
            "pitch_embed_dim": 128, "dur_embed_dim": 64,
            "d_model": 256, "nhead": 4, "num_layers": 3,
            "dim_feedforward": 1024, "dropout": 0.1,
        },
    },
    "seed": 57,
}

torch.manual_seed(config["seed"])

model_cfg = config["model"][config["arch"]]
run_config = {k: v for k, v in config.items() if k != "model"} | {"model": model_cfg}
wandb.init(project="Harmonizer", config=run_config)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = Tokenizer.from_songs([json.loads(l) for l in open("data/processed/train.jsonl")])
tokenizer.save("artifacts/tokenizer.json")

train_dataset = HarmonizerDataset("data/processed/train.jsonl", tokenizer, max_len=config["max_len"])
val_dataset = HarmonizerDataset("data/processed/val.jsonl", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=val_dataset.collate_fn)

model = build_model(config["arch"], model_cfg, tokenizer).to(device)
print(f"{config['arch']}: {sum(p.numel() for p in model.parameters()):,} parameters")

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.chord_stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

ckpt_path = f"artifacts/{config['arch']}.pt"
best_val_loss = float("inf")

for epoch in range(config["num_epochs"]):

    model.train()
    total_train_loss = 0
    for batch in train_loader:
        pitch_batch, dur_batch, chord_batch, lengths = [b.to(device) for b in batch]
        optimizer.zero_grad()
        chord_pred = model(pitch_batch, dur_batch)
        loss = criterion(chord_pred.transpose(1, 2), chord_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            pitch_batch, dur_batch, chord_batch, lengths = [b.to(device) for b in batch]
            chord_pred = model(pitch_batch, dur_batch)
            val_loss = criterion(chord_pred.transpose(1, 2), chord_batch)
            total_val_loss += val_loss.item()

            preds = chord_pred.argmax(-1)
            mask = chord_batch != 0
            correct += (preds[mask] == chord_batch[mask]).sum().item()
            total += mask.sum().item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    val_acc = correct / total if total > 0 else 0

    wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": val_acc})

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({"config": dict(wandb.config), "state_dict": model.state_dict()}, ckpt_path)

    print(f"Epoch {epoch+1}/{config['num_epochs']}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}")