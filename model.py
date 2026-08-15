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
