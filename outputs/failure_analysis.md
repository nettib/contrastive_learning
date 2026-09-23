# Failure Case Analysis

## Failure Case 1: Training loss decreases sharply, but representation quality gets WORSE
**Configuration:** Temperature = 0.05 (Experiment 1)

**What happened:**
The InfoNCE training loss for τ=0.05 dropped from 2.87 to 0.17 over 20 epochs —
by far the lowest final loss of any configuration tested (compare: baseline τ=0.5
ended at 4.46). Despite this dramatically lower loss, every representation-quality
metric got WORSE relative to the baseline:
- k-NN accuracy: 96.0% (baseline) -> 93.7% (temp=0.05)
- Similarity gap: 0.350 -> 0.226
- Silhouette score: 0.330 -> 0.191

**Why it happened:**
A very small temperature makes the softmax in InfoNCE extremely "sharp" (see
Section 8 of the reference document). This means the model can achieve a very
low loss simply by making each anchor's embedding trivially distinguishable
from the specific 2N-2 negatives in ITS OWN batch, without needing to build a
globally consistent, well-organized embedding space. In effect, the sharp
temperature lets the model "memorize" easy shortcuts for each individual batch
(e.g., tiny, class-irrelevant pixel differences between the two augmented views
and their in-batch negatives) rather than learning the broad, semantically
meaningful structure (grouping by true digit identity) that generalizes to
k-NN classification on the full test set. This is a direct illustration of the
InfoNCE loss value not being a reliable, standalone proxy for representation
quality -- exactly the discrepancy the assignment's Section 5 asks us to watch
for.

**Possible improvement:**
Use a moderate temperature (0.3-1.0 range, as suggested by our τ=1.0 result,
which had the BEST similarity gap and silhouette score of all configurations).
Alternatively, always validate temperature choices using a held-out
representation-quality metric (k-NN accuracy, similarity gap) rather than
training loss alone, and consider a temperature warm-up/decay schedule rather
than a single fixed sharp value.


## Failure Case 2: Persistent visual confusion between digits 4 and 9 (and 3/8/9 more broadly)
**Configuration:** Baseline (τ=0.5, weak augmentation, with projection head)

**What happened:**
Even after contrastive training, the confusion matrix on k-NN classification
shows digit 4 is misclassified as 9 in 10 of 167 test cases -- the single
largest off-diagonal error of any digit pair. Qualitatively, our retrieval
visualization also shows an "8" query retrieving a "3" as a nearest neighbor
(similarity 0.92) after training, and a "3" query retrieving a "9" (similarity
0.93).

**Why it happened:**
Contrastive learning organizes the embedding space based on visual/structural
similarity created by the augmentation pipeline, not by explicit semantic
class labels (Section 4 of the reference document). Certain digit pairs are
genuinely close in raw visual appearance depending on handwriting style: a
loosely-written "4" can have a nearly closed loop resembling a "9"; an "8" and
a "3" both consist of stacked curved loops; a "3" and a "9" can share a similar
upper curve. Because the model never receives the true label, it has no signal
to specifically pull these visually-similar-but-different classes apart beyond
whatever separation naturally emerges from augmentation-invariance training.
Weak augmentation (small crops/rotations) preserves this ambiguity rather than
forcing the model to find more robust distinguishing features.

**Possible improvement:**
- Increase the diversity/hardness of positive pairs moderately (without going
  as far as our "strong" augmentation setting, which we found reduces overall
  quality) to encourage the encoder to focus on more robust structural
  features rather than superficial stroke similarity.
- Use a larger batch size or more training epochs, giving more and harder
  in-batch negatives so genuinely different-but-similar-looking classes (4 vs
  9, 3 vs 8) get pushed apart more explicitly.
- Supplement with a small amount of label supervision (e.g., a lightweight
  supervised fine-tuning step on top of the frozen contrastive encoder) if
  perfect separation of these specific hard pairs is required for a downstream
  task.


## Failure Case 3: Strong augmentation reduces representation quality across the board
**Configuration:** Strong augmentation (Experiment 2)

**What happened:**
Compared to the weak-augmentation baseline, the strong-augmentation
configuration showed lower scores on every metric: k-NN accuracy (94.5% vs
96.0%), similarity gap (0.329 vs 0.350), silhouette (0.293 vs 0.330), and
notably the largest drop in Recall@5 (0.974 vs 0.986) of any experiment.

**Why it happened:**
As shown directly in our augmentation visualization, strong augmentation
(heavy crop, large rotation, random erasing) can remove enough of a digit's
defining strokes that the two "positive" views of the same image no longer
reliably represent the same visual content -- our own example showed a "0"
distorted into a shape resembling a "C" under strong augmentation. When
positive pairs are corrupted this way, the model is effectively being told
"these two different-looking things should be considered similar" for reasons
unrelated to true digit identity, injecting a genuinely misleading training
signal. This matches Section 5 of the reference document's warning: "Too much
transformation -> Semantic information is lost -> Poor positive pair."

**Possible improvement:**
Tune augmentation strength specifically for the dataset's visual simplicity;
MNIST digits have very little redundant visual information compared to natural
photos (which is why SimCLR's original strong augmentations, designed for
ImageNet, are often excessive for simple datasets like MNIST). A milder
"strong" setting (e.g., moderate rotation without aggressive cutout) or
class-aware augmentation limits (e.g., capping crop/rotation to a range
verified not to change digit identity) would likely recover quality while
still testing the effect of augmentation strength.
