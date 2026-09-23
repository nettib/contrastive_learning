"""
Training loop for the contrastive learning pipeline.
"""
import torch
from torch.utils.data import DataLoader
import time

from models import ContrastiveModel
from losses import info_nce_loss
from data_augmentation import ContrastiveMNIST


def train_contrastive_model(train_images, train_labels,
                             embedding_dim=128, proj_dim=32,
                             use_projection_head=True,
                             augmentation="weak",
                             temperature=0.5,
                             batch_size=256,
                             epochs=10,
                             lr=1e-3,
                             device="cpu",
                             log_every=10,
                             seed=0,
                             encoder_type="small_cnn"):
    torch.manual_seed(seed)

    dataset = ContrastiveMNIST(train_images, train_labels, augmentation=augmentation)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, drop_last=True)

    model = ContrastiveModel(embedding_dim=embedding_dim, proj_dim=proj_dim,
                              use_projection_head=use_projection_head,
                              encoder_type=encoder_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_idx, (v1, v2, _labels) in enumerate(loader):
            v1, v2 = v1.to(device), v2.to(device)

            _h1, z1 = model(v1)
            _h2, z2 = model(v2)

            loss = info_nce_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"  epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | elapsed={elapsed:.1f}s")

    return model, loss_history


if __name__ == "__main__":
    from mnist_loader import load_mnist
    tr_x, tr_y, te_x, te_y = load_mnist('/home/claude/data/mnist_raw')

    # Tiny smoke test: 500 images, 2 epochs, just to confirm nothing crashes
    # and loss actually decreases.
    print("Running smoke test (500 images, 2 epochs)...")
    model, history = train_contrastive_model(
        tr_x[:500], tr_y[:500],
        epochs=2, batch_size=64, log_every=1
    )
    print("Loss history:", history)
    assert history[-1] < history[0], "Loss did not decrease -- something is wrong!"
    print("Smoke test passed: loss decreased.")
