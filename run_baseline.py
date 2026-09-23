"""
Baseline run: compare a RANDOM (untrained) encoder vs a CONTRASTIVELY
TRAINED encoder, to demonstrate the effect of training (Section 3:
"compare the representation before and after contrastive training").
"""
import torch
from torch.utils.data import DataLoader
import json

from mnist_loader import load_mnist
from data_augmentation import PlainMNIST
from models import ContrastiveModel
from train import train_contrastive_model
from evaluate import run_full_evaluation, plot_tsne

DEVICE = "cpu"
TRAIN_N = 6000
TEST_N = 1500
EPOCHS = 20
BATCH_SIZE = 256
TEMPERATURE = 0.5
SEED = 0

def main():
    tr_x, tr_y, te_x, te_y = load_mnist('/home/claude/data/mnist_raw')
    tr_x, tr_y = tr_x[:TRAIN_N], tr_y[:TRAIN_N]
    te_x, te_y = te_x[:TEST_N], te_y[:TEST_N]
    print(f"Using {len(tr_x)} train / {len(te_x)} test images (documented subset).")

    train_plain = PlainMNIST(tr_x, tr_y)
    test_plain = PlainMNIST(te_x, te_y)
    train_loader = DataLoader(train_plain, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_plain, batch_size=256, shuffle=False)

    all_results = {}

    # ---- BEFORE: random, untrained encoder ----
    print("\n=== BEFORE (random untrained encoder) ===")
    torch.manual_seed(SEED)
    random_model = ContrastiveModel(embedding_dim=128, proj_dim=32,
                                     use_projection_head=True).to(DEVICE)
    results_before, emb_before, labels_before = run_full_evaluation(
        random_model, train_loader, test_loader, DEVICE, tag="BEFORE (random)")
    all_results["before"] = results_before
    plot_tsne(emb_before, labels_before, "t-SNE: BEFORE training (random encoder)",
               "outputs/tsne_before.png")

    # ---- AFTER: contrastively trained encoder ----
    print("\n=== TRAINING (baseline config) ===")
    model, loss_history = train_contrastive_model(
        tr_x, tr_y,
        embedding_dim=128, proj_dim=32,
        use_projection_head=True,
        augmentation="weak",
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE,
        log_every=5,
        seed=SEED,
    )
    all_results["baseline_loss_history"] = loss_history

    print("\n=== AFTER (contrastively trained encoder) ===")
    results_after, emb_after, labels_after = run_full_evaluation(
        model, train_loader, test_loader, DEVICE, tag="AFTER (trained)")
    all_results["after"] = results_after
    plot_tsne(emb_after, labels_after, "t-SNE: AFTER contrastive training",
               "outputs/tsne_after.png")

    with open("outputs/baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"kNN accuracy: {results_before['knn_accuracy']:.4f} -> {results_after['knn_accuracy']:.4f}")
    print(f"Similarity gap: {results_before['similarity_gap']:.4f} -> {results_after['similarity_gap']:.4f}")
    print(f"Silhouette: {results_before['silhouette']:.4f} -> {results_after['silhouette']:.4f}")
    print(f"Recall@5: {results_before['recall_at_5']:.4f} -> {results_after['recall_at_5']:.4f}")

    torch.save(model.state_dict(), "outputs/baseline_model.pt")
    return model, all_results

if __name__ == "__main__":
    main()
