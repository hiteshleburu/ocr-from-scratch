import argparse
import csv
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from PIL import Image

from .vocab import build_vocab, save_vocab, load_vocab
from .dataset import OCRDataset, collate_fn, build_transform, load_labels
from .model import OCRModel
from .metrics import word_accuracy, cer
from .generate import generate


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_loss(model, loader, device, vocab_size):
    model.eval()
    total = 0
    with torch.no_grad():
        for imgs, tgt_in, tgt_out in loader:
            imgs, tgt_in, tgt_out = imgs.to(device), tgt_in.to(device), tgt_out.to(device)
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                logits = model(imgs, tgt_in)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_out.reshape(-1), ignore_index=0)
            total += loss.item()
    model.train()
    return total / len(loader)


def log_run(log_path, row):
    file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    if os.path.exists(args.vocab_path):
        stoi, itos = load_vocab(args.vocab_path)
    else:
        stoi, itos = build_vocab()
        save_vocab(stoi, itos, args.vocab_path)
    vocab_size = len(stoi)

    paths, labels = load_labels(args.labels_path, args.image_dir)
    train_transform = build_transform(args.img_h, args.img_w, augment=True)
    val_transform = build_transform(args.img_h, args.img_w, augment=False)

    full_ds = OCRDataset(paths, labels, train_transform, stoi)
    n = len(full_ds)
    val_size = int(args.val_frac * n)
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n - val_size, val_size], generator=generator)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    val_indices_path = os.path.join(args.checkpoint_dir, f"{run_id}_val_indices.pt")
    torch.save(val_ds.indices, val_indices_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                               shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                             shuffle=False, num_workers=args.num_workers)

    model = OCRModel(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                      n_layers=args.n_layers, ff_dim=args.ff_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    else:
        warmup_steps = args.warmup_steps
        scheduler = LambdaLR(optimizer, lambda step: min((step + 1) / warmup_steps, 1.0))

    scaler = torch.amp.GradScaler()

    with torch.no_grad():
        imgs, tgt_in, tgt_out = next(iter(train_loader))
        imgs, tgt_in, tgt_out = imgs.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(imgs, tgt_in)
        loss0 = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_out.reshape(-1), ignore_index=0)
    print(f"pre-training loss check: {loss0.item():.4f}")

    best_val_loss = float('inf')
    best_ckpt_path = None
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, tgt_in, tgt_out in train_loader:
            imgs, tgt_in, tgt_out = imgs.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                logits = model(imgs, tgt_in)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_out.reshape(-1),
                                        ignore_index=0, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device, vocab_size)
        print(f"epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(args.checkpoint_dir, f"{run_id}_best.pt")
            torch.save(model.state_dict(), best_ckpt_path)

    elapsed = time.time() - start_time

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()
    val_paths = [paths[i] for i in val_ds.indices]
    val_labels = [labels[i] for i in val_ds.indices]

    preds = []
    for p in val_paths[:args.eval_sample_size]:
        img = val_transform(Image.open(p).convert('RGB'))
        preds.append(generate(model, img, itos, device, ngram_block=args.ngram_block))
    sample_labels = val_labels[:args.eval_sample_size]

    acc = word_accuracy(preds, sample_labels)
    error_rate = cer(preds, sample_labels)
    print(f"final word accuracy: {acc:.4f}  CER: {error_rate:.4f}")

    log_run(args.log_path, {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'labels_path': args.labels_path,
        'n_train_images': len(train_ds),
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'epochs': args.epochs,
        'lr': args.lr,
        'lr_schedule': args.lr_schedule,
        'batch_size': args.batch_size,
        'best_val_loss': round(best_val_loss, 4),
        'word_accuracy': round(acc, 4),
        'cer': round(error_rate, 4),
        'n_params': n_params,
        'train_time_sec': round(elapsed, 1),
        'checkpoint_path': best_ckpt_path,
        'val_indices_path': val_indices_path,
    })
    print(f"run logged to {args.log_path}")
    print(f"best checkpoint: {best_ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, default='src/recognition/vocab.json')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--img_h', type=int, default=32)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--ff_dim', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_schedule', type=str, default='flat', choices=['flat', 'cosine'])
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_sample_size', type=int, default=1000)
    parser.add_argument('--ngram_block', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/recognition')
    parser.add_argument('--log_path', type=str, default='runs.csv')

    args = parser.parse_args()
    main(args)