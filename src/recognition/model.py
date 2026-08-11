import math
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, d_model=256, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2)
        return x.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q_in, kv_in, mask=None):
        B, Tq, _ = q_in.shape
        Tk = kv_in.shape[1]
        Q = self.q_proj(q_in).view(B, Tq, self.h, self.dk).transpose(1, 2)
        K = self.k_proj(kv_in).view(B, Tk, self.h, self.dk).transpose(1, 2)
        V = self.v_proj(kv_in).view(B, Tk, self.h, self.dk).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.out_proj(out)


def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, ff_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, self_mask):
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, mask=self_mask)
        x = x + self.cross_attn(self.norm2(x), memory, mask=None)
        x = x + self.ffn(self.norm3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=3, ff_dim=1024, max_len=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ff_dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt_in, memory):
        B, T = tgt_in.shape
        x = self.embed(tgt_in) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        mask = causal_mask(T, tgt_in.device)
        for layer in self.layers:
            x = layer(x, memory, mask)
        x = self.norm_out(x)
        return self.fc_out(x)


class OCRModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=3, ff_dim=1024, max_len=50):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.decoder = Decoder(vocab_size, d_model, n_heads, n_layers, ff_dim, max_len)

    def forward(self, imgs, tgt_in):
        memory = self.encoder(imgs)
        return self.decoder(tgt_in, memory)