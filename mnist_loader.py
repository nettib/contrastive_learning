import numpy as np
from torchvision import datasets


def load_mnist(root="./data"):
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True
    )

    train_images = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()

    test_images = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()

    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    tr_x, tr_y, te_x, te_y = load_mnist()

    print("Train images:", tr_x.shape, tr_x.dtype)
    print("Train labels:", tr_y.shape, tr_y.dtype, "unique:", np.unique(tr_y))
    print("Test images:", te_x.shape)
    print("Pixel range:", tr_x.min(), tr_x.max())