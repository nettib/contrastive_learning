"""
InfoNCE loss, implemented by hand (per Section 8), using in-batch
negatives (per Section 9).

Given a batch of N images, we have 2N views (view1 for all N, then
view2 for all N). For each view, its positive is the OTHER view of the
same original image; every other view in the batch (2N-2 of them) is
a negative.
"""
import torch
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: (N, D) normalized embeddings -- view1 and view2 of N images.
    Returns: scalar loss (averaged over all 2N anchors).
    """
    N = z1.shape[0]
    device = z1.device

    # Stack into one big batch of 2N embeddings:
    # indices [0..N-1] = view1, indices [N..2N-1] = view2
    z = torch.cat([z1, z2], dim=0)          # (2N, D)

    # Cosine similarity matrix between every pair of the 2N embeddings.
    # Since z is already unit-normalized, dot product == cosine similarity.
    sim_matrix = torch.matmul(z, z.T)       # (2N, 2N)
    sim_matrix = sim_matrix / temperature

    # Mask out the diagonal (an embedding's similarity with ITSELF) --
    # a sample is never its own negative or positive candidate.
    mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask, -1e9)

    # For anchor i in [0..N-1] (view1), its positive is i+N (view2).
    # For anchor i in [N..2N-1] (view2), its positive is i-N (view1).
    positive_indices = torch.arange(2 * N, device=device)
    positive_indices = (positive_indices + N) % (2 * N)

    # InfoNCE = cross-entropy where the "correct class" for each anchor
    # is its positive's index, and the "logits" are the similarity row.
    loss = F.cross_entropy(sim_matrix, positive_indices)
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 4, 8
    z1 = F.normalize(torch.randn(N, D), dim=1)
    z2 = F.normalize(torch.randn(N, D), dim=1)

    for tau in [0.05, 0.5, 5.0]:
        loss = info_nce_loss(z1, z2, temperature=tau)
        print(f"temperature={tau:<5} -> loss={loss.item():.4f}")

    # Sanity check: if z2 == z1 exactly (perfect positive match, easy task),
    # loss should be much lower than random embeddings.
    loss_perfect = info_nce_loss(z1, z1.clone(), temperature=0.5)
    loss_random = info_nce_loss(z1, z2, temperature=0.5)
    print(f"\nPerfect positive match loss: {loss_perfect.item():.4f}")
    print(f"Random (mismatched) loss:    {loss_random.item():.4f}")
    assert loss_perfect < loss_random, "Sanity check failed!"
    print("Sanity check passed: perfect matches give lower loss.")
