"""
Encoder (small CNN) and Projection Head, per Sections 3 and 6 of the
reference document.
"""
import torch
import torch.nn as nn


class SmallCNNEncoder(nn.Module):
    """
    Input:  (B, 1, 28, 28)
    Output: h, shape (B, embedding_dim)  -- the "main learned representation"
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 14x14 -> 7x7
        )
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        h = self.fc(x)
        return h


class ProjectionHead(nn.Module):
    """
    Small MLP: h (embedding_dim) -> z (proj_dim)
    Used ONLY during contrastive training; discarded afterward.
    """
    def __init__(self, embedding_dim=128, hidden_dim=64, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, h):
        z = self.net(h)
        return z


class ContrastiveModel(nn.Module):
    """
    Full model: encoder -> (optional) projection head.
    If use_projection_head=False, the contrastive loss is applied
    directly on (a linearly reduced) h, for the "no projection head"
    ablation in Experiment 3.

    encoder_type: "small_cnn" (default) or "resnet18"
    """
    def __init__(self, embedding_dim=128, proj_dim=32, use_projection_head=True,
                 encoder_type="small_cnn"):
        super().__init__()
        if encoder_type == "small_cnn":
            self.encoder = SmallCNNEncoder(embedding_dim=embedding_dim)
        elif encoder_type == "resnet18":
            from resnet_encoder import ResNet18Encoder
            self.encoder = ResNet18Encoder(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projector = ProjectionHead(embedding_dim, 64, proj_dim)
        else:
            self.projector = None

    def forward(self, x):
        h = self.encoder(x)
        if self.use_projection_head:
            z = self.projector(h)
        else:
            z = h  # loss applied directly on encoder output
        z = nn.functional.normalize(z, dim=1)  # unit-length embeddings
        return h, z


if __name__ == "__main__":
    model = ContrastiveModel(embedding_dim=128, proj_dim=32, use_projection_head=True)
    x = torch.randn(4, 1, 28, 28)
    h, z = model(x)
    print("h shape:", h.shape)   # (4, 128)
    print("z shape:", z.shape)   # (4, 32)
    print("z norms (should all be 1.0):", z.norm(dim=1))

    n_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", n_params)
