"""
GRU Autoencoder model for tactile anomaly detection.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        return out, h_n


class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h_n):
        B = h_n.shape[1]
        last_h = h_n[-1]
        inp = last_h.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.gru(inp, h_n)
        return self.fc(out)


class GRUAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, seq_len):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecoder(hidden_dim, input_dim, num_layers, dropout, seq_len)

    def forward(self, x):
        _, h_n = self.encoder(x)
        return self.decoder(h_n)

    def reconstruction_error(self, x: torch.Tensor, weight_alpha: float = 0.0) -> torch.Tensor:
        """
        x: (B, T, D)
        weight_alpha: 0=uniform, >0=recent frames weighted more
        Returns: (B,) reconstruction error per sample
        """
        recon = self.forward(x)
        seq_len = x.shape[1]
        if weight_alpha > 0:
            weights = torch.exp(
                torch.linspace(-weight_alpha, 0.0, seq_len, device=x.device)
            )
            weights = weights / weights.sum()
            frame_err = ((x - recon) ** 2).mean(dim=2)
            return (frame_err * weights.unsqueeze(0)).sum(dim=1)
        else:
            return ((x - recon) ** 2).mean(dim=(1, 2))
