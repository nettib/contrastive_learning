# Contrastive Learning on MNIST — Custom CNN

A complete, from-scratch contrastive learning pipeline (SimCLR-style) built
and evaluated on MNIST, following the assignment **"Contrastive Learning and
Representation Geometry: From Similarity to Structure."**

The project uses a custom CNN encoder and investigates how contrastive
learning affects representation quality and geometry through controlled
experiments involving temperature, data augmentation, and the projection
head.

---

## Setup / Requirements

This project uses Python and a virtual environment.

### Requirements

* Python 3.14
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib

Exact package versions are specified in `requirements.txt`.

### 1. Create a virtual environment

From the project directory:

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv)
```

### 3. Install dependencies

Use the Python interpreter from the active virtual environment:

```powershell
python -m pip install -r requirements.txt
```

### 4. Verify the environment

You can verify that the correct Python interpreter is being used with:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

The executable path should point to the project's `.venv` directory.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST is downloaded and loaded using:

```python
torchvision.datasets.MNIST
```

The dataset is automatically downloaded the first time the loader is run and
cached locally under the project's `data/` directory.

The loader returns:

* Images: NumPy arrays with shape `(N, 28, 28)`
* Labels: NumPy arrays with shape `(N,)`
* Image type: `uint8`
* Pixel range: `0–255`
* Number of classes: `10` (`0` through `9`)

The full MNIST dataset contains:

* 60,000 training images
* 10,000 test images

The experiments use a documented subset of the dataset for faster CPU-based
experimentation.

---

## Dataset & Experimental Setup

### Dataset

**MNIST subset**

* 6,000 training images
* 1,500 test images
* 10 digit classes
* Image size: `28 × 28`
* Grayscale images

The subset is used to allow repeated controlled experiments to run efficiently
on CPU.

### Encoder

**Custom CNN**

The encoder consists of:

* 2 convolutional layers
* ReLU activations
* Max pooling
* Fully connected representation layer

The encoder produces a learned representation:

```text
image → CNN → h
```

where `h` is the encoder representation used for representation analysis.

See `models.py` for the implementation.

### Projection Head

A small MLP projection head is used for the contrastive objective:

```text
h → Linear → ReLU → Linear → z
```

Dimensions:

```text
128 → 64 → 32
```

The projection head produces the representation `z` used by the InfoNCE loss.

### Embeddings

The embeddings are L2-normalized before computing cosine similarity.

The normalized representation is therefore:

```text
z_normalized = z / ||z||₂
```

Cosine similarity is then used to measure how close two representations are.

### Contrastive Loss

The project uses a hand-implemented **InfoNCE loss**.

The implementation is provided in:

```text
losses.py
```

The loss uses in-batch negative samples.

With a batch size of 256:

```text
256 positive pairs
→ 512 augmented samples
→ 510 negatives per anchor
```

### Training

The baseline training configuration uses:

* Optimizer: Adam
* Learning rate: `1e-3`
* Epochs: `20`
* Batch size: `256`
* Temperature: `0.5`
* CPU execution

---

## Contrastive Learning Pipeline

The overall training pipeline is:

```text
                MNIST image
                     │
                     ▼
              Data augmentation
                 /        \
                /          \
               ▼            ▼
          View 1          View 2
             │               │
             ▼               ▼
          Custom CNN      Custom CNN
             │               │
             └───────┬───────┘
                     ▼
                  Encoder
               representations
                   h₁, h₂
                     │
                     ▼
             Projection Head
                   /   \
                  /     \
                 ▼       ▼
               z₁       z₂
                 \       /
                  \     /
                   ▼   ▼
              L2 normalization
                     │
                     ▼
            Cosine similarity
                     │
                     ▼
                InfoNCE loss
                     │
                     ▼
                Backpropagation
```

The CNN encoder learns representations by making two augmented views of the
same image similar while separating representations of different images.

---

## File Structure

```text
contrastive_learning_mnist_smallCNN/
│
├── data/
│   └── MNIST/                  # Downloaded automatically; not committed
│
├── outputs/                    # Generated experiment results
│   ├── ...
│
├── mnist_loader.py             # Loads MNIST using torchvision
├── data_augmentation.py        # Weak/strong augmentation pipelines
├── models.py                   # CustomCNNEncoder, ProjectionHead,
│                               # ContrastiveModel
├── losses.py                   # Hand-implemented InfoNCE loss
├── train.py                    # Training loop
├── evaluate.py                 # Representation evaluation
├── run_baseline.py             # Before/after training comparison
├── run_experiments.py          # Controlled experiments
├── run_retrieval.py            # Nearest-neighbor retrieval
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignored local/generated files
```

### Important directories

`data/`

Contains the downloaded MNIST dataset. It is generated automatically and should
not be committed to the repository.

`outputs/`

Contains generated models, plots, experiment results, logs, and analysis
artifacts.

`.venv/`

Contains the local Python virtual environment and should not be committed.

---

## How to Reproduce

Make sure the virtual environment is activated before running the commands.

### 1. Sanity-check the dataset and augmentations

```powershell
python mnist_loader.py
python data_augmentation.py
```

The dataset loader should produce output similar to:

```text
Train images: (60000, 28, 28) uint8
Train labels: (60000,) int64 unique: [0 1 2 3 4 5 6 7 8 9]
Test images: (10000, 28, 28)
Pixel range: 0 255
```

The first execution may download the MNIST dataset.

### 2. Confirm the model and loss work correctly

```powershell
python models.py
python losses.py
```

These scripts perform basic sanity checks for the model architecture and
InfoNCE implementation.

### 3. Run the baseline comparison

```powershell
python run_baseline.py
```

This compares representations before and after contrastive training.

The comparison evaluates whether contrastive training produces a more
structured representation space.

### 4. Run the controlled experiments

```powershell
python run_experiments.py
```

This runs the controlled experiments investigating:

1. Temperature
2. Data augmentation strength
3. Projection-head usage

The experiments generate numerical results, loss curves, and visualization
artifacts under `outputs/`.

### 5. Generate retrieval visualizations

```powershell
python run_retrieval.py
```

This generates nearest-neighbor retrieval visualizations showing query images
and their top-5 retrieved neighbors.

---

# Results Summary

## Before vs After Contrastive Training

Baseline configuration:

* Temperature: `τ = 0.5`
* Weak augmentation
* Projection head enabled

| Metric                             | Before (random encoder) | After (trained) |
| ---------------------------------- | ----------------------: | --------------: |
| k-NN accuracy                      |                   91.2% |       **96.0%** |
| Similarity gap (`S_same − S_diff`) |                   0.044 |       **0.350** |
| Silhouette score                   |                   0.110 |       **0.330** |
| Recall@5                           |                   97.8% |       **98.6%** |

The representation becomes more structured after contrastive training.

See:

```text
outputs/tsne_before.png
outputs/tsne_after.png
```

for the corresponding t-SNE visualizations.

---

## Controlled Experiments

| Configuration                                            | k-NN Accuracy | Similarity Gap | Silhouette | Recall@5 |
| -------------------------------------------------------- | ------------: | -------------: | ---------: | -------: |
| **Baseline** (τ=0.5, weak augmentation, projection head) |         0.960 |          0.350 |      0.330 |    0.986 |
| Temperature Low (τ=0.05)                                 |         0.937 |          0.226 |      0.191 |    0.988 |
| Temperature High (τ=1.0)                                 |         0.959 |          0.388 |      0.386 |    0.986 |
| Strong Augmentation                                      |         0.945 |          0.329 |      0.293 |    0.974 |
| No Projection Head                                       |         0.953 |      **0.487** |      0.354 |    0.984 |

Full numerical results:

```text
outputs/experiment_results.json
```

t-SNE visualizations:

```text
outputs/tsne_<config_name>.png
```

Experiment comparison:

```text
outputs/experiment_comparison_bars.png
```

Loss curves:

```text
outputs/loss_curves_all.png
```

---

# Key Findings

### 1. Temperature = 0.05

Temperature `τ = 0.05` produced a substantial decrease in training loss:

```text
4.46 → 0.17
```

However, representation quality decreased across the evaluation metrics.

This demonstrates that a lower contrastive training loss does not necessarily
mean that the learned representation has improved.

See:

```text
outputs/failure_analysis.md
```

for the detailed analysis.

### 2. Temperature = 1.0

The higher temperature configuration produced:

* A higher similarity gap
* A higher silhouette score
* Similar k-NN accuracy
* Similar Recall@5

compared with the baseline.

This provides a useful example of how temperature changes the geometry of the
learned representation space.

### 3. Strong Augmentation

Strong augmentation reduced representation quality across the evaluation
metrics.

The result is consistent with the possibility that overly aggressive
transformations can remove information that is useful for identifying the
digit.

This effect is also visualized in:

```text
outputs/augmentation_comparison.png
```

where strong transformations can substantially alter the appearance of a
digit.

### 4. Removing the Projection Head

Removing the projection head produced the highest raw similarity gap:

```text
0.487
```

This is an interesting result because the projection head is an important
component of the SimCLR framework.

The experiment therefore provides an opportunity to discuss how behavior
observed in large-scale SimCLR experiments may differ when working with a
small CNN and a simple dataset such as MNIST.

---

# Representation Geometry

A central goal of the project is not only to minimize the contrastive loss, but
to examine **what happens to the geometry of the learned representation
space**.

The project evaluates this using:

### Similarity Gap

The similarity gap is:

```text
S_same − S_diff
```

where:

* `S_same` is the average similarity between representations belonging to
  the same class.
* `S_diff` is the average similarity between representations belonging to
  different classes.

A larger gap indicates stronger separation between same-class and
different-class representations.

### Silhouette Score

The silhouette score evaluates how well samples are separated according to
their digit classes.

Higher values indicate stronger within-class cohesion and between-class
separation.

### k-NN Accuracy

A k-nearest-neighbor classifier is applied directly to the learned
representations.

This evaluates whether the representation space naturally places examples
from the same class near each other.

### Recall@5

Recall@5 measures whether at least one correct-class example appears among
the five nearest retrieved representations.

### t-SNE

t-SNE is used to visualize the representation space in two dimensions.

The visualizations allow the geometric changes produced by contrastive
training and controlled experimental changes to be inspected directly.

---

# Failure Case Analysis

The project documents three major failure cases using the following framework:

```text
What happened
      ↓
Why it happened
      ↓
Possible improvement
```

### Failure Case 1 — Low Temperature

Temperature `τ = 0.05` substantially reduced the training loss while producing
worse representation-quality metrics.

This demonstrates the difference between optimizing the training objective and
obtaining a useful representation.

### Failure Case 2 — Digit Confusion

The learned representation still contains confusion between visually similar
digits, particularly:

```text
4 ↔ 9
3 ↔ 8 ↔ 9
```

The confusion is documented using:

```text
outputs/confusion_matrix_baseline.png
```

### Failure Case 3 — Strong Augmentation

Strong transformations can alter the semantic appearance of MNIST digits.

As a result, the model may be encouraged to treat substantially different
visual structures as different views of the same underlying example.

The resulting representation quality decreases.

---

# Retrieval

The project also evaluates the learned representation through nearest-neighbor
retrieval.

The generated visualizations are:

```text
outputs/retrieval_before.png
outputs/retrieval_after.png
```

Each visualization shows:

* Query image
* Top-5 nearest neighbors
* Correct retrievals
* Incorrect retrievals
* Digit labels

The retrieval results provide a direct visual interpretation of the learned
representation geometry.

For example, some queries can retrieve visually similar but semantically
different digits, such as an `8` retrieving a `3`.

---

# Outputs

The `outputs/` directory contains the generated experimental artifacts.

Important outputs include:

```text
outputs/
├── experiment_results.json
├── experiment_comparison_bars.png
├── loss_curves_all.png
├── tsne_before.png
├── tsne_after.png
├── tsne_<config_name>.png
├── augmentation_comparison.png
├── confusion_matrix_baseline.png
├── retrieval_before.png
├── retrieval_after.png
└── failure_analysis.md
```

The exact set of files may vary depending on which experiments have been run.

---

# Known Limitations

* The experiments use a **6,000-image training subset** and a **1,500-image
  test subset** rather than the full MNIST dataset.
* The reduced dataset is used to allow repeated controlled experiments to run
  efficiently on CPU.
* Training uses a limited number of epochs.
* The experiments are configured for CPU execution.
* Silhouette score and similarity gap are computed using sampled test
  embeddings rather than necessarily evaluating every possible pair in the
  test set.
* t-SNE is a visualization technique and its two-dimensional geometry should
  not be interpreted as an exact representation of the original embedding
  space.

---

# Reproducibility

To reproduce the project from a clean checkout:

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Load and verify MNIST
python mnist_loader.py

# Check augmentations
python data_augmentation.py

# Check model
python models.py

# Check InfoNCE loss
python losses.py

# Run baseline
python run_baseline.py

# Run controlled experiments
python run_experiments.py

# Generate retrieval visualizations
python run_retrieval.py
```

The MNIST dataset will be downloaded automatically into `data/` when needed.

---

# License

This project was created for educational and experimental purposes as part of
a study of contrastive learning, representation learning, and representation
geometry.
