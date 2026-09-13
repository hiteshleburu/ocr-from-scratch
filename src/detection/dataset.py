import os
from PIL import Image, ImageOps
import torchvision.transforms as T
from torch.utils.data import Dataset


def resize_with_padding(img, target_size, fill=0):
    """
    Resizes img to fit within target_size while preserving aspect ratio,
    then pads with `fill` to reach the exact target size.
    """
    target_w, target_h = target_size
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = max(target_w - new_w, 0)
    pad_h = max(target_h - new_h, 0)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

    return ImageOps.expand(img_resized, padding, fill=fill)


class DetectionTransform:
    """
    A class (not a closure) so it stays picklable for multiprocessing DataLoader workers.
    """
    def __init__(self, img_h=256, img_w=256, fill=0):
        self.img_h = img_h
        self.img_w = img_w
        self.fill = fill

    def __call__(self, img):
        img = resize_with_padding(img, (self.img_w, self.img_h), fill=self.fill)
        return T.ToTensor()(img)


class DetectionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, n_samples, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.n_samples = n_samples
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, f"{idx:05d}.png")).convert('L')
        mask = Image.open(os.path.join(self.mask_dir, f"{idx:05d}.png")).convert('L')

        img = self.transform(img)
        mask = self.transform(mask)
        mask = (mask > 0.5).float()

        return img, mask