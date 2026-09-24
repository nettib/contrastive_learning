import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mnist_loader import load_mnist
from data_augmentation import PlainMNIST
from models import ContrastiveModel
from evaluate import get_embeddings

DEVICE = "cpu"
TEST_N = 1500


def retrieve_top_k(embeddings, query_idx, k=5):
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sims = emb_norm @ emb_norm[query_idx]
    sims[query_idx] = -np.inf
    top_k = np.argpartition(-sims, k)[:k]
    top_k = top_k[np.argsort(-sims[top_k])]
    return top_k, sims[top_k]


def plot_retrieval_grid(images, labels, embeddings, query_indices, title, save_path, k=5):
    n_queries = len(query_indices)
    fig, axes = plt.subplots(n_queries, k + 1, figsize=(2 * (k + 1), 2 * n_queries))

    for row, qi in enumerate(query_indices):
        neighbors, sims = retrieve_top_k(embeddings, qi, k=k)

        ax = axes[row, 0]
        ax.imshow(images[qi], cmap='gray')
        ax.set_title(f"QUERY\nlabel={labels[qi]}", fontsize=9, color='blue')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)

        for col, (ni, sim) in enumerate(zip(neighbors, sims)):
            ax = axes[row, col + 1]
            ax.imshow(images[ni], cmap='gray')
            correct = (labels[ni] == labels[qi])
            color = 'green' if correct else 'red'
            ax.set_title(f"label={labels[ni]}\nsim={sim:.2f}", fontsize=8, color=color)
            ax.axis('off')

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    tr_x, tr_y, te_x, te_y = load_mnist('/home/claude/data/mnist_raw')
    te_x, te_y = te_x[:TEST_N], te_y[:TEST_N]

    test_plain = PlainMNIST(te_x, te_y)
    test_loader = DataLoader(test_plain, batch_size=256, shuffle=False)

    # Pick fixed query indices (mix of "easy" and commonly-confused digits: 4,9 and 3,5,8)
    rng = np.random.RandomState(1)
    query_indices = []
    for digit in [4, 9, 3, 5, 8, 1]:
        idx = np.where(te_y == digit)[0]
        query_indices.append(int(rng.choice(idx)))

    # ---- Random (untrained) model ----
    torch.manual_seed(0)
    random_model = ContrastiveModel(embedding_dim=128, proj_dim=32,
                                     use_projection_head=True).to(DEVICE)
    emb_random, labels_check = get_embeddings(random_model, test_loader, DEVICE, use_projection=False)
    plot_retrieval_grid(te_x, te_y, emb_random, query_indices,
                         "Retrieval BEFORE training (random encoder)",
                         "outputs/retrieval_before.png")

    # ---- Trained baseline model ----
    trained_model = ContrastiveModel(embedding_dim=128, proj_dim=32, use_projection_head=True).to(DEVICE)
    trained_model.load_state_dict(torch.load("outputs/model_baseline.pt", map_location=DEVICE))
    emb_trained, _ = get_embeddings(trained_model, test_loader, DEVICE, use_projection=False)
    plot_retrieval_grid(te_x, te_y, emb_trained, query_indices,
                         "Retrieval AFTER contrastive training (baseline config)",
                         "outputs/retrieval_after.png")


if __name__ == "__main__":
    main()
