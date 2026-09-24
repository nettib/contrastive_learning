"""
Representation Quality & Geometry evaluation
"""
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.no_grad()
def get_embeddings(model, dataloader, device, use_projection=False):
    """Runs the encoder (h) -- or optionally the full projector (z) -- over a dataset."""
    model.eval()
    all_emb, all_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        h, z = model(imgs)
        emb = z if use_projection else h
        all_emb.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_emb), np.concatenate(all_labels)


def knn_accuracy(train_emb, train_labels, test_emb, test_labels, k=5):
    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    clf.fit(train_emb, train_labels)
    acc = clf.score(test_emb, test_labels)
    return acc


def similarity_gap(embeddings, labels, n_samples=2000, seed=0):
    """S_same - S_different, per Section 10."""
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    emb = embeddings[idx]
    lab = labels[idx]

    # normalize for cosine similarity via dot product
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim_matrix = emb_norm @ emb_norm.T

    same_mask = (lab[:, None] == lab[None, :])
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)

    s_same = sim_matrix[same_mask].mean()
    s_diff = sim_matrix[diff_mask].mean()
    return s_same, s_diff, s_same - s_diff


def cluster_silhouette(embeddings, labels, n_samples=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    return silhouette_score(embeddings[idx], labels[idx], metric='cosine')


def recall_at_k(embeddings, labels, k=5, n_queries=500, seed=0):
    """For n_queries random samples, check if a same-label sample appears
    in the top-k nearest neighbors (excluding itself)."""
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    query_idx = rng.choice(n, size=min(n_queries, n), replace=False)

    sim = emb_norm[query_idx] @ emb_norm.T   # (n_queries, n)
    hits = 0
    for row_i, qi in enumerate(query_idx):
        sims = sim[row_i].copy()
        sims[qi] = -np.inf  # exclude self
        top_k_idx = np.argpartition(-sims, k)[:k]
        if labels[qi] in labels[top_k_idx]:
            hits += 1
    return hits / len(query_idx)


def plot_tsne(embeddings, labels, title, save_path, n_samples=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    emb = embeddings[idx]
    lab = labels[idx]

    tsne = TSNE(n_components=2, random_state=seed, init='pca', perplexity=30)
    proj = tsne.fit_transform(emb)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=lab, cmap='tab10', s=8, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='digit')
    plt.title(title)
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def run_full_evaluation(model, train_loader, test_loader, device,
                         use_projection=False, tag="model"):
    """Runs all evaluation metrics and returns a results dict."""
    train_emb, train_labels = get_embeddings(model, train_loader, device, use_projection)
    test_emb, test_labels = get_embeddings(model, test_loader, device, use_projection)

    acc = knn_accuracy(train_emb, train_labels, test_emb, test_labels, k=5)
    s_same, s_diff, gap = similarity_gap(test_emb, test_labels)
    sil = cluster_silhouette(test_emb, test_labels)
    rec5 = recall_at_k(test_emb, test_labels, k=5)

    results = {
        "knn_accuracy": float(acc),
        "s_same": float(s_same),
        "s_diff": float(s_diff),
        "similarity_gap": float(gap),
        "silhouette": float(sil),
        "recall_at_5": float(rec5),
    }
    print(f"[{tag}] kNN acc={acc:.4f} | S_same={s_same:.4f} S_diff={s_diff:.4f} "
          f"gap={gap:.4f} | silhouette={sil:.4f} | recall@5={rec5:.4f}")
    return results, test_emb, test_labels
