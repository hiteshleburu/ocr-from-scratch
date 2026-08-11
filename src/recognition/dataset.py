import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from .vocab import PAD, BOS, EOS


def build_transform(img_h=32, img_w=128, augment=False):
    ops = [T.Resize((img_h, img_w)), T.Grayscale()]

    if augment:
        ops.append(T.RandomApply(
            [T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.9, 1.1))], p=0.5
        ))
        ops.append(T.RandomApply([T.GaussianBlur(3)], p=0.3))
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3))

    ops += [T.ToTensor(), T.Normalize([0.5], [0.5])]
    return T.Compose(ops)


class OCRDataset(Dataset):
    def __init__(self, paths, labels, transform, stoi):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.stoi = stoi

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.paths[idx]).convert('RGB'))
        ids = [self.stoi[c] for c in self.labels[idx].lower() if c in self.stoi]
        return img, ids


def collate_fn(batch):
    imgs, label_lists = zip(*batch)
    imgs = torch.stack(imgs)

    max_len = max(len(l) for l in label_lists) + 1
    tgt_in = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    tgt_out = torch.full((len(batch), max_len), PAD, dtype=torch.long)

    for i, ids in enumerate(label_lists):
        seq = [BOS] + ids
        tgt_in[i, :len(seq)] = torch.tensor(seq)
        out = ids + [EOS]
        tgt_out[i, :len(out)] = torch.tensor(out)

    return imgs, tgt_in, tgt_out


def load_labels(labels_path, image_dir):
    paths, labels = [], []
    with open(labels_path) as f:
        for line in f:
            fname, label = line.strip().split("\t")
            paths.append(f"{image_dir}/{fname}")
            labels.append(label)
    return paths, labels