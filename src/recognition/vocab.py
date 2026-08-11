import json

PAD, BOS, EOS = 0, 1, 2

def build_vocab(charset="0123456789abcdefghijklmnopqrstuvwxyz"):
    chars = list(charset)
    stoi = {c: i + 3 for i, c in enumerate(chars)}
    stoi['<PAD>'], stoi['<BOS>'], stoi['<EOS>'] = PAD, BOS, EOS
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos

def save_vocab(stoi, itos, path):
    with open(path, 'w') as f:
        json.dump({'stoi': stoi, 'itos': {str(k): v for k, v in itos.items()}}, f)

def load_vocab(path):
    with open(path) as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    itos = {int(k): v for k, v in vocab['itos'].items()}
    return stoi, itos

def encode(word, stoi):
    return [stoi[c] for c in word.lower() if c in stoi]

def decode(ids, itos):
    return ''.join(itos[i] for i in ids if i not in (PAD, BOS, EOS))