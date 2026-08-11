import torch
from .vocab import PAD, BOS, EOS


@torch.no_grad()
def generate(model, img, itos, device, max_len=25, ngram_block=None):
    model.eval()
    img = img.unsqueeze(0).to(device)
    memory = model.encoder(img)

    seq = [BOS]
    for _ in range(max_len):
        tgt = torch.tensor([seq], device=device)
        logits = model.decoder(tgt, memory)[0, -1]

        if ngram_block and len(seq) >= ngram_block:
            banned = set()
            prefix = tuple(seq[-(ngram_block - 1):])
            for i in range(len(seq) - ngram_block + 1):
                if tuple(seq[i:i + ngram_block - 1]) == prefix:
                    banned.add(seq[i + ngram_block - 1])
            for tok in banned:
                if tok not in (PAD, BOS, EOS):
                    logits[tok] = float('-inf')

        next_id = logits.argmax().item()
        if next_id == EOS:
            break
        seq.append(next_id)

    return ''.join(itos[i] for i in seq[1:])


@torch.no_grad()
def generate_batch(model, imgs, itos, device, max_len=25, ngram_block=None):
    return [generate(model, img, itos, device, max_len=max_len, ngram_block=ngram_block) for img in imgs]