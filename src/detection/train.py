import argparse
import csv
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import DetectionModel
from .dataset import DetectionDataset, DetectionTransform
from .postprocess import mask_to_boxes, evaluate_boxes


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


def evaluate_loss(model, loader, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            total += loss.item()
    model.train()
    return total / len(loader)


def evaluate_detection_metrics(model, dataset, device, sample_size=None):
    model.eval()
    indices = range(len(dataset)) if sample_size is None else range(min(sample_size, len(dataset)))
    all_p, all_r, all_f1 = [], [], []

    with torch.no_grad():
        for idx in indices:
            img, mask = dataset[idx]
            img = img.unsqueeze(0).to(device)
            logits = model(img)
            probs = torch.sigmoid(logits)
            pred_mask_np = (probs[0, 0] > 0.5).float().cpu().numpy()
            true_mask_np = mask[0].cpu().numpy()

            pred_boxes = mask_to_boxes(pred_mask_np)
            true_boxes = mask_to_boxes(true_mask_np)

            p, r, f1 = evaluate_boxes(pred_boxes, true_boxes)
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)

    model.train()
    n = len(all_p)
    return sum(all_p) / n, sum(all_r) / n, sum(all_f1) / n


def log_run(log_path, row):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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

    transform = DetectionTransform(img_h=args.img_h, img_w=args.img_w)
    full_ds = DetectionDataset(args.image_dir, args.mask_dir, args.n_images, transform)

    n = len(full_ds)
    val_size = int(args.val_frac * n)
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n - val_size, val_size], generator=generator)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    val_indices_path = os.path.join(args.checkpoint_dir, f"{run_id}_val_indices.pt")
    torch.save(val_ds.indices, val_indices_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DetectionModel(in_channels=1, base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    best_ckpt_path = None
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"epoch {epoch}: train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(args.checkpoint_dir, f"{run_id}_best.pt")
            torch.save(model.state_dict(), best_ckpt_path)

    elapsed = time.time() - start

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    precision, recall, f1 = evaluate_detection_metrics(model, val_ds, device, sample_size=args.eval_sample_size)
    print(f"precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")

    log_run(args.log_path, {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'image_dir': args.image_dir,
        'n_images': args.n_images,
        'base_channels': args.base_channels,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'best_val_loss': round(best_val_loss, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'n_params': n_params,
        'train_time_sec': round(elapsed, 1),
        'checkpoint_path': best_ckpt_path,
        'val_indices_path': val_indices_path,
    })
    print(f"run logged to {args.log_path}")
    print(f"best checkpoint: {best_ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--n_images', type=int, required=True)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=256)

    parser.add_argument('--base_channels', type=int, default=32)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--eval_sample_size', type=int, default=1000)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/detection')
    parser.add_argument('--log_path', type=str, default='runs/runs_detection.csv')

    args = parser.parse_args()
    main(args)