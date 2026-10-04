import math

import torch
import torch.nn as nn

class LSTMHarmonizer(nn.Module):
    def __init__(self, pitch_vocab_size, dur_vocab_size, chord_vocab_size, pitch_embed_dim, dur_embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(LSTMHarmonizer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.pitch_embedding = nn.Embedding(pitch_vocab_size, pitch_embed_dim, padding_idx=0)
        self.dur_embedding = nn.Embedding(dur_vocab_size, dur_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=pitch_embed_dim + dur_embed_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True, batch_first=True)

        self.chord_out = nn.Linear(hidden_dim * 2, chord_vocab_size)

    def forward(self, pitch_seq, dur_seq):
        pitch_embedded = self.pitch_embedding(pitch_seq)
        dur_embedded = self.dur_embedding(dur_seq)

        combined = torch.cat((pitch_embedded, dur_embedded), dim=-1)

        lstm_out, _ = self.lstm(combined)

        chord_pred = self.chord_out(lstm_out)

        return chord_pred

class TransformerHarmonizer(nn.Module):
    def __init__(self, pitch_vocab_size, dur_vocab_size, chord_vocab_size,
                 pitch_embed_dim, dur_embed_dim, d_model=256,
                 nhead=4, num_layers=2, dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, pitch_embed_dim, padding_idx=0)
        self.dur_embedding = nn.Embedding(dur_vocab_size, dur_embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(pitch_embed_dim + dur_embed_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.chord_out = nn.Linear(d_model, chord_vocab_size)

    def forward(self, pitch_seq, dur_seq):
        pad_mask = pitch_seq == 0

        x = torch.cat((self.pitch_embedding(pitch_seq),
                       self.dur_embedding(dur_seq)), dim=-1)
        x = self.input_proj(x)
        x = x + sinusoidal_encoding(pitch_seq.size(1), x.size(-1), x.device)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        return self.chord_out(self.norm(x))

def build_model(arch, model_cfg, tokenizer):
    vocab = dict(
        pitch_vocab_size=len(tokenizer.pitch_stoi),
        dur_vocab_size=len(tokenizer.duration_stoi),
        chord_vocab_size=len(tokenizer.chord_stoi),
    )
    if arch == "lstm":
        return LSTMHarmonizer(**vocab, **model_cfg)
    if arch == "transformer":
        return TransformerHarmonizer(**vocab, **model_cfg)
    raise ValueError(f"unknown arch: {arch}")

def sinusoidal_encoding(length, d_model, device):
    pos = torch.arange(length, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device)
    freq = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe
