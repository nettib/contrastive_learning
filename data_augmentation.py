"""
Contrastive dataset wrapper: for each image, produces TWO augmented views
(x1, x2), which form a positive pair, per Section 5 of the assignment.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


def get_weak_augmentation():
    """Small, gentle transformations -> very similar views -> easy contrastive task."""
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(28, scale=(0.85, 1.0)),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2),
        T.ToTensor(),
    ])


def get_strong_augmentation():
    """Heavy distortions -> more different views -> harder contrastive task."""
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(28, scale=(0.5, 1.0)),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.6, contrast=0.6),
        T.ToTensor(),
        T.RandomErasing(p=0.5, scale=(0.05, 0.2)),
    ])


class ContrastiveMNIST(Dataset):
    """
    Wraps raw MNIST numpy arrays. Each __getitem__ returns two augmented
    views of the SAME image (a positive pair), plus the true label
    (label is NOT used in training -- only kept for later evaluation).
    """
    def __init__(self, images, labels, augmentation="weak"):
        self.images = images  # (N, 28, 28) uint8
        self.labels = labels
        if augmentation == "weak":
            self.transform = get_weak_augmentation()
        elif augmentation == "strong":
            self.transform = get_strong_augmentation()
        else:
            raise ValueError("augmentation must be 'weak' or 'strong'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (28,28) uint8
        view1 = self.transform(img)
        view2 = self.transform(img)
        label = int(self.labels[idx])
        return view1, view2, label


class PlainMNIST(Dataset):
    """No augmentation -- used for evaluation (k-NN, retrieval, t-SNE)."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1,28,28)
        label = int(self.labels[idx])
        return img, label


if __name__ == "__main__":
    from mnist_loader import load_mnist
    tr_x, tr_y, te_x, te_y = load_mnist('/home/claude/data/mnist_raw')
    ds = ContrastiveMNIST(tr_x[:100], tr_y[:100], augmentation="weak")
    v1, v2, y = ds[0]
    print("view1 shape:", v1.shape, "view2 shape:", v2.shape, "label:", y)
    print("view1 range:", v1.min().item(), v1.max().item())
