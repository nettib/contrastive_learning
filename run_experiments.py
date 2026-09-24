"""
Runs the 3 required controlled experiments, each changing
ONE factor relative to the baseline config:
    baseline: temperature=0.5, augmentation=weak, projection_head=True

Experiment 1 - Temperature:      0.05  vs  0.5 (baseline)  vs  1.0
Experiment 2 - Augmentation:     weak (baseline)  vs  strong
Experiment 3 - Projection head:  with (baseline)  vs  without
"""
import torch
from torch.utils.data import DataLoader
import json

from mnist_loader import load_mnist
from data_augmentation import PlainMNIST
from train import train_contrastive_model
from evaluate import run_full_evaluation, plot_tsne

DEVICE = "cpu"
TRAIN_N = 6000
TEST_N = 1500
EPOCHS = 20
BATCH_SIZE = 256
SEED = 0

BASELINE_CFG = dict(temperature=0.5, augmentation="weak", use_projection_head=True)

CONFIGS = {
    "baseline":        dict(temperature=0.5,  augmentation="weak",   use_projection_head=True),
    "temp_low_0.05":   dict(temperature=0.05, augmentation="weak",   use_projection_head=True),
    "temp_high_1.0":   dict(temperature=1.0,  augmentation="weak",   use_projection_head=True),
    "aug_strong":      dict(temperature=0.5,  augmentation="strong", use_projection_head=True),
    "no_proj_head":    dict(temperature=0.5,  augmentation="weak",   use_projection_head=False),
}


def main():
    tr_x, tr_y, te_x, te_y = load_mnist('/home/claude/data/mnist_raw')
    tr_x, tr_y = tr_x[:TRAIN_N], tr_y[:TRAIN_N]
    te_x, te_y = te_x[:TEST_N], te_y[:TEST_N]

    train_plain = PlainMNIST(tr_x, tr_y)
    test_plain = PlainMNIST(te_x, te_y)
    train_loader = DataLoader(train_plain, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_plain, batch_size=256, shuffle=False)

    all_results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}\nRUNNING CONFIG: {name} -> {cfg}\n{'='*60}")
        model, loss_history = train_contrastive_model(
            tr_x, tr_y,
            embedding_dim=128, proj_dim=32,
            use_projection_head=cfg["use_projection_head"],
            augmentation=cfg["augmentation"],
            temperature=cfg["temperature"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            device=DEVICE,
            log_every=5,
            seed=SEED,
        )

        # IMPORTANT: for evaluation, always read out `h` (encoder output),
        # NOT the projection z -- per Section 6, h is the representation
        # used downstream, regardless of whether a projection head was
        # used during training.
        results, emb, labels = run_full_evaluation(
            model, train_loader, test_loader, DEVICE,
            use_projection=False, tag=name)

        plot_tsne(emb, labels, f"t-SNE: {name}", f"outputs/tsne_{name}.png")

        all_results[name] = {
            "config": cfg,
            "loss_history": loss_history,
            "metrics": results,
        }

        torch.save(model.state_dict(), f"outputs/model_{name}.pt")

    with open("outputs/experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n" + "="*60)
    print("FINAL SUMMARY TABLE")
    print("="*60)
    header = f"{'config':<16} {'final_loss':>10} {'knn_acc':>8} {'sim_gap':>8} {'silhouette':>10} {'recall@5':>9}"
    print(header)
    for name, r in all_results.items():
        m = r["metrics"]
        print(f"{name:<16} {r['loss_history'][-1]:>10.4f} {m['knn_accuracy']:>8.4f} "
              f"{m['similarity_gap']:>8.4f} {m['silhouette']:>10.4f} {m['recall_at_5']:>9.4f}")

    return all_results


if __name__ == "__main__":
    main()
