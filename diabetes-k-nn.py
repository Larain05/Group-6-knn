# =============================================================
# KNN DIABETES CLASSIFICATION — FULL MANUAL IMPLEMENTATION
# University of Southern Mindanao
# Computational Science for Computer Science
#
# Everything here is built from scratch using only
# numpy and pandas. No sklearn anywhere in this file.
# The class API (.fit / .predict / .score) mirrors sklearn
# exactly so you understand what sklearn does under the hood.
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')


# =============================================================
# SECTION 1 — DISTANCE FUNCTIONS
# =============================================================
# All four distance metrics take two 1-D arrays (a, b)
# and return a single float representing how "far apart"
# those two points are in feature space.

def euclidean_distance(a, b):
    """
    Most common distance metric. Measures the straight-line
    ("as the crow flies") distance between two points.

    Formula:  d = sqrt( sum( (ai - bi)^2 ) )

    Best for: continuous features on the same scale.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    """
    Also called "city block" or "taxicab" distance. Measures
    the total absolute difference along each feature axis —
    like navigating a city grid (only right angles, no diagonals).

    Formula:  d = sum( |ai - bi| )

    Best for: high-dimensional data, when outliers are a concern.
    """
    return np.sum(np.abs(a - b))


def minkowski_distance(a, b, p=3):
    """
    Generalized distance that includes Euclidean (p=2) and
    Manhattan (p=1) as special cases. Higher p values give
    more weight to the largest difference across features.

    Formula:  d = ( sum( |ai - bi|^p ) )^(1/p)

    p=1  → Manhattan
    p=2  → Euclidean
    p=3+ → Minkowski (used here; emphasizes the largest gaps)
    """
    return np.sum(np.abs(a - b) ** p) ** (1.0 / p)


def chebyshev_distance(a, b):
    """
    Takes only the single largest difference across all features.
    Ignores everything else — useful when you care about the
    worst-case mismatch on any one feature.

    Formula:  d = max( |ai - bi| )

    Best for: board-game-style movement (king moves in chess),
              or when extreme differences on one feature matter most.
    """
    return np.max(np.abs(a - b))


# =============================================================
# SECTION 2 — MANUAL STANDARD SCALER
# =============================================================
# Mirrors sklearn's StandardScaler.
# .fit()           → learns mean and std from training data
# .transform()     → applies (x - mean) / std
# .fit_transform() → fit + transform in one call

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to
    unit variance. Required before KNN so that no single feature
    dominates the distance calculation due to its scale.

    After scaling:  mean = 0,  std = 1  for every feature.
    """

    def __init__(self):
        self.mean_ = None   # Learned from training data
        self.std_  = None   # Learned from training data

    def fit(self, X):
        """Compute mean and std from X (training data only)."""
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        # Protect against zero-variance features (constant columns)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        """Subtract mean and divide by std."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit on X, then transform X — convenience method."""
        return self.fit(X).transform(X)


# =============================================================
# SECTION 3 — MANUAL TRAIN / TEST SPLIT
# =============================================================
# Mirrors sklearn's train_test_split().
# Shuffles all indices randomly, then cuts at the split point.

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Randomly shuffle data then split into training and test sets.

    Parameters
    ----------
    X           : 2-D array of features
    y           : 1-D array of labels
    test_size   : fraction to hold out for testing (default 0.2 = 20%)
    random_state: seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n         = len(X)
    indices   = np.random.permutation(n)        # Shuffle all row indices
    split_at  = int(n * (1 - test_size))        # Cut point (80% train)

    train_idx = indices[:split_at]
    test_idx  = indices[split_at:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =============================================================
# SECTION 4 — MANUAL CONFUSION MATRIX & ACCURACY
# =============================================================

def confusion_matrix(y_true, y_pred):
    """
    Returns a 2x2 confusion matrix for binary classification.

        Predicted: No  |  Predicted: Yes
    Actual: No   [ TN  |  FP ]
    Actual: Yes  [ FN  |  TP ]

    TN = True Negative  (correctly predicted Non-diabetic)
    FP = False Positive (predicted Diabetic, actually wasn't)
    FN = False Negative (predicted Non-diabetic, actually was)
    TP = True Positive  (correctly predicted Diabetic)
    """
    classes = sorted(set(y_true))
    n       = len(classes)
    cm      = np.zeros((n, n), dtype=int)

    label_to_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true]][label_to_idx[pred]] += 1

    return cm


def accuracy_score(y_true, y_pred):
    """Fraction of predictions that match the true label."""
    return np.mean(np.array(y_true) == np.array(y_pred))


# =============================================================
# SECTION 5 — KNN CLASS (sklearn-style API)
# =============================================================
# This class deliberately mirrors sklearn's KNeighborsClassifier:
#
#   knn = KNNClassifier(n_neighbors=5)
#   knn.fit(X_train, y_train)
#   y_pred = knn.predict(X_test)
#   acc    = knn.score(X_test, y_test)
#
# Understanding this class tells you exactly what sklearn
# is doing when you call KNeighborsClassifier.

class KNNClassifier:
    """
    K-Nearest Neighbors classifier built from scratch.

    Algorithm (prediction for one test point):
      1. Compute distance from the test point to EVERY training point.
      2. Sort distances → find the K smallest (nearest neighbors).
      3. Look up the label of each neighbor.
      4. Take a majority vote → that's the prediction.

    Parameters
    ----------
    n_neighbors  : number of neighbors K  (default 5)
    metric       : distance function to use
                   'euclidean' | 'manhattan' | 'minkowski' | 'chebyshev'
    p            : power for Minkowski distance (ignored otherwise)
    """

    METRICS = {
        'euclidean' : euclidean_distance,
        'manhattan' : manhattan_distance,
        'chebyshev' : chebyshev_distance,
    }

    def __init__(self, n_neighbors=5, metric='euclidean', p=3):
        self.n_neighbors = n_neighbors
        self.metric      = metric
        self.p           = p
        self._X_train    = None
        self._y_train    = None

    def _get_dist_fn(self):
        """Return the correct distance function based on self.metric."""
        if self.metric == 'minkowski':
            # Bind p into the function via a default argument
            return lambda a, b: minkowski_distance(a, b, self.p)
        if self.metric not in self.METRICS:
            raise ValueError(
                f"Unknown metric '{self.metric}'. "
                f"Choose from: {list(self.METRICS.keys()) + ['minkowski']}"
            )
        return self.METRICS[self.metric]

    def fit(self, X_train, y_train):
        """
        Store the training data.
        KNN is a lazy learner — there is no real 'training'.
        All the computation happens at prediction time.
        """
        self._X_train = np.array(X_train)
        self._y_train = np.array(y_train)
        return self                         # Allows method chaining

    def _predict_one(self, x, dist_fn):
        """
        Predict the class label for a single test point x.

        Steps:
          1. Compute distance from x to every training point.
          2. Find indices of the K smallest distances.
          3. Retrieve labels of those K neighbors.
          4. Return the majority-vote label.
        """
        # Step 1 — compute all distances at once
        distances = np.array([dist_fn(x, xt) for xt in self._X_train])

        # Step 2 — argsort gives indices that would sort the array;
        #           [:k] keeps only the K nearest
        nn_indices = np.argsort(distances)[:self.n_neighbors]

        # Step 3 — labels of the K nearest neighbors
        nn_labels  = self._y_train[nn_indices]

        # Step 4 — majority vote via bincount
        #           bincount([0,0,1]) → [2, 1]  → argmax → 0
        counts = np.bincount(nn_labels.astype(int))
        return int(np.argmax(counts))

    def predict(self, X_test):
        """
        Predict class labels for every row in X_test.
        Returns a 1-D array of predictions.
        """
        if self._X_train is None:
            raise RuntimeError("Call .fit() before .predict()")

        dist_fn = self._get_dist_fn()
        return np.array([self._predict_one(x, dist_fn) for x in X_test])

    def score(self, X_test, y_test):
        """
        Compute accuracy on the given test set.
        Equivalent to accuracy_score(y_test, self.predict(X_test)).
        """
        return accuracy_score(y_test, self.predict(X_test))


# =============================================================
# SECTION 6 — BEST-K FINDER
# =============================================================

def find_best_k(X_train, y_train, X_test, y_test,
                k_range=range(1, 16), metric='euclidean'):
    """
    Evaluate KNN for every K in k_range and return the K
    that gives the highest accuracy on the test set.

    Parameters
    ----------
    k_range : iterable of K values to try  (default 1–15)
    metric  : distance metric to use for all evaluations

    Returns
    -------
    best_k        : int
    best_acc      : float
    all_k         : list of K values tried
    all_acc       : list of corresponding accuracies
    """
    all_k   = list(k_range)
    all_acc = []

    for k in all_k:
        knn = KNNClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        all_acc.append(knn.score(X_test, y_test))

    best_idx = int(np.argmax(all_acc))
    return all_k[best_idx], all_acc[best_idx], all_k, all_acc


# =============================================================
# SECTION 7 — LOAD & PREPROCESS DATASET (FINAL FIX)
# =============================================================

print("=" * 60)
print("  KNN DIABETES CLASSIFICATION — MANUAL IMPLEMENTATION")
print("=" * 60)

df = pd.read_csv("diabetes-k-nn.csv")

print("\n── FIRST 5 ROWS ─────────────────────────────────────────")
print(df.head())

print("\n── DATASET INFO ─────────────────────────────────────────")
print(df.info())

print("\n── SUMMARY BEFORE PREPROCESSING ────────────────────────")
print(df.describe().round(2))


# =============================================================
# ✅ STEP 1 — CHECK ZERO VALUES
# =============================================================

columns_with_zero = ['Glucose', 'BloodPressure',
                     'SkinThickness', 'Insulin', 'BMI']

print("\n── ZERO VALUE CHECK ────────────────────────────────────")
for col in columns_with_zero:
    zero_count = (df[col] == 0).sum()
    print(f"{col:<20}: {zero_count} zero values")


# =============================================================
# ✅ STEP 2 — CONVERT ZERO → NaN (treat as missing)
# =============================================================

df[columns_with_zero] = df[columns_with_zero].replace(0, np.nan)


# =============================================================
# ✅ STEP 3 — MEDIAN IMPUTATION
# =============================================================

df[columns_with_zero] = df[columns_with_zero].apply(
    lambda col: col.fillna(col.median())
)


print("\n── SUMMARY AFTER PREPROCESSING ─────────────────────────")
print(df.describe().round(2))


# =============================================================
# CONTINUE AS NORMAL
# =============================================================

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure',
                 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction', 'Age']

X = df[feature_names].values
y = df['Outcome'].values


# =============================================================
# SECTION 8 — SPLIT FIRST, THEN SCALE (FIXED)
# =============================================================

# Split BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Now scale using ONLY training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n── FEATURE SCALING ──────────────────────────────────────")
print(f"  Learned mean (first 3 features): {scaler.mean_[:3].round(3)}")
print(f"  Learned std  (first 3 features): {scaler.std_[:3].round(3)}")

print(f"\n── TRAIN / TEST SPLIT ───────────────────────────────────")
print(f"  Total samples  : {len(X)}")
print(f"  Training samples: {len(X_train)}  ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Testing  samples: {len(X_test)}   ({len(X_test)/len(X)*100:.0f}%)")

# =============================================================
# SECTION 9 — TRAIN KNN (K = 3, 5, 7) WITH EUCLIDEAN DISTANCE
# =============================================================
# This section satisfies Part 3 of the activity:
# train with at least three different K values.

k_values   = [3, 5, 7]
accuracies = []
cms        = []
all_preds  = []

print("\n" + "=" * 60)
print("  PART 3 — KNN RESULTS  (Euclidean distance)")
print("=" * 60)

for k in k_values:
    knn    = KNNClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc    = knn.score(X_test, y_test)
    cm     = confusion_matrix(y_test, y_pred)

    accuracies.append(acc)
    cms.append(cm)
    all_preds.append(y_pred)

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n  K = {k}")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={tn}  FP={fp}")
    print(f"      FN={fn}  TP={tp}")

best_k_idx = int(np.argmax(accuracies))
best_k     = k_values[best_k_idx]
best_acc   = accuracies[best_k_idx]
print(f"\n  ★  Best K = {best_k}  (Accuracy: {best_acc*100:.2f}%)")


# =============================================================
# SECTION 10 — MANUAL STEP-BY-STEP DISTANCE COMPUTATION
# =============================================================
# Part 3 requirement: for ONE test instance, manually compute
# the Euclidean distance to at least 10 training samples,
# showing every feature's contribution explicitly.

test_instance = X_test[0]
true_label    = int(y_test[0])

print("\n" + "=" * 60)
print("  PART 3 — MANUAL DISTANCE COMPUTATION  (step-by-step)")
print("=" * 60)
print(f"\n  Selected test instance #{0}  "
      f"(True label: {'Diabetic (1)' if true_label==1 else 'Non-diabetic (0)'})")
print(f"\n  Scaled feature values:")
for fname, fval in zip(feature_names, test_instance):
    print(f"    {fname:<28}: {fval:>8.4f}")

manual_distances = []

for i in range(10):
    train_sample = X_train[i]
    label        = int(y_train[i])

    print(f"\n  ── Distance to Training Sample {i} "
          f"({'Diabetic' if label==1 else 'Non-diabetic'}) ──")

    squared_diffs = []
    for fname, tv, trv in zip(feature_names, test_instance, train_sample):
        diff    = tv - trv
        diff_sq = diff ** 2
        squared_diffs.append(diff_sq)
        print(f"    {fname:<28}: ({tv:>7.4f} - {trv:>7.4f})^2"
              f" = ({diff:>7.4f})^2 = {diff_sq:.4f}")

    total_sq = sum(squared_diffs)
    dist     = np.sqrt(total_sq)
    print(f"    {'─'*52}")
    print(f"    Sum of squared differences       : {total_sq:.4f}")
    print(f"    Euclidean distance = sqrt({total_sq:.4f}) : {dist:.4f}")

    manual_distances.append((i, dist, label))

# Sort to find nearest neighbors
manual_sorted = sorted(manual_distances, key=lambda x: x[1])

print("\n  ── Sorted distances ─────────────────────────────────")
print(f"  {'Rank':<6} {'Sample':<8} {'Distance':<12} {'Label':<20} {'Role'}")
print(f"  {'─'*60}")
for rank, (idx, dist, lbl) in enumerate(manual_sorted, 1):
    role  = f"← Neighbor #{rank}" if rank <= best_k else ""
    label = "Diabetic (1)" if lbl == 1 else "Non-diabetic (0)"
    print(f"  {rank:<6} {idx:<8} {dist:<12.4f} {label:<20} {role}")

# Majority vote
top_labels   = [d[2] for d in manual_sorted[:best_k]]
n_diabetic   = sum(top_labels)
n_non        = best_k - n_diabetic
prediction   = 1 if n_diabetic > n_non else 0

print(f"\n  Majority vote (K={best_k}):")
print(f"    Non-diabetic (0) : {n_non} vote(s)")
print(f"    Diabetic     (1) : {n_diabetic} vote(s)")
print(f"    → Predicted class: "
      f"{'Diabetic (1)' if prediction==1 else 'Non-diabetic (0)'}")
print(f"    → True class     : "
      f"{'Diabetic (1)' if true_label==1 else 'Non-diabetic (0)'}")
print(f"    → Correct?       : {'YES ✓' if prediction==true_label else 'NO ✗'}")


# =============================================================
# SECTION 11 — ALL FOUR DISTANCE METRICS
# =============================================================
# Show how accuracy changes depending on which distance
# function is used — with K = best_k throughout.

print("\n" + "=" * 60)
print(f"  ALL FOUR DISTANCE METRICS  (K = {best_k})")
print("=" * 60)

metric_configs = [
    ('euclidean',  'Euclidean  (straight-line)',         {}),
    ('manhattan',  'Manhattan  (city block)',             {}),
    ('minkowski',  'Minkowski  (p=3, generalized)',      {'p': 3}),
    ('chebyshev',  'Chebyshev  (max single-feature gap)',{}),
]

metric_names   = []
metric_accs    = []
metric_cms     = []

for metric, label, kwargs in metric_configs:
    knn    = KNNClassifier(n_neighbors=best_k, metric=metric, **kwargs)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc    = knn.score(X_test, y_test)
    cm     = confusion_matrix(y_test, y_pred)

    metric_names.append(label)
    metric_accs.append(acc)
    metric_cms.append(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {label}")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")


# =============================================================
# SECTION 12 — FIND BEST K  (K = 1 to 15)
# =============================================================

print("\n" + "=" * 60)
print("  BEST-K SEARCH  (K = 1 to 15, Euclidean)")
print("=" * 60)

best_k_auto, best_acc_auto, all_k, all_acc = find_best_k(
    X_train, y_train, X_test, y_test,
    k_range=range(1, 16), metric='euclidean'
)

for k, acc in zip(all_k, all_acc):
    marker = " ← best" if k == best_k_auto else ""
    print(f"  K={k:>2}  accuracy={acc:.4f}  ({acc*100:.1f}%){marker}")

print(f"\n  ★  Best K found automatically: {best_k_auto}"
      f"  (Accuracy: {best_acc_auto*100:.2f}%)")


# =============================================================
# SECTION 13 — AESTHETIC RESULTS DASHBOARD
# =============================================================

BG      = "#0D0F1A"
CARD    = "#13162A"
ACCENT  = "#6C63FF"
ACCENT2 = "#FF6584"
GREEN   = "#2DD4BF"
MUTED   = "#8B8FA8"
WHITE   = "#F0F2FF"
GRID_C  = "#1E2038"
ORANGE  = "#FFB347"
PINK    = "#FF6584"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : CARD,
    "axes.edgecolor"   : GRID_C,
    "axes.labelcolor"  : MUTED,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "text.color"       : WHITE,
    "grid.color"       : GRID_C,
    "grid.linewidth"   : 0.8,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(20, 17), facecolor=BG)

fig.text(0.5, 0.979,
         "KNN Diabetes Classification — Full Manual Implementation",
         ha="center", va="top",
         fontsize=20, fontweight="bold", color=WHITE)
fig.text(0.5, 0.962,
         "No sklearn used  |  768 patients  |  Train: 614  |  Test: 154  |  8 features",
         ha="center", va="top", fontsize=12, color=MUTED)

gs = GridSpec(3, 3, figure=fig,
              top=0.95, bottom=0.05,
              left=0.06, right=0.97,
              hspace=0.50, wspace=0.36)


# ── PLOT 1: Accuracy vs K (1–15) ─────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title("Accuracy vs K  (K = 1 to 15, Euclidean)",
              fontsize=14, fontweight="bold", color=WHITE,
              pad=12, loc="left")

bar_colors = [GREEN if k == best_k_auto else ACCENT for k in all_k]
bars = ax1.bar(all_k, [a * 100 for a in all_acc],
               color=bar_colors, alpha=0.80, width=0.7,
               zorder=3, linewidth=0)

for bar, v, k in zip(bars, all_acc, all_k):
    if k == best_k_auto:
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.3,
                 f"{v*100:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=GREEN)
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 100 / 2,
                 "BEST", ha="center", va="center",
                 fontsize=8, color=BG, fontweight="bold")

ax1.plot(all_k, [a * 100 for a in all_acc],
         color=WHITE, lw=1.2, ls="--", alpha=0.35, zorder=4)
ax1.scatter(all_k, [a * 100 for a in all_acc],
            color=WHITE, s=40, zorder=5)
ax1.set_xticks(all_k)
ax1.set_xticklabels([str(k) for k in all_k], fontsize=9)
ax1.set_xlabel("K (number of neighbors)", fontsize=11)
ax1.set_ylim(60, 82)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.yaxis.grid(True, zorder=0)
ax1.set_axisbelow(True)
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_C)
ax1.legend(
    handles=[mpatches.Patch(color=GREEN,  label=f"Best K = {best_k_auto}"),
             mpatches.Patch(color=ACCENT, label="Other K values")],
    loc="lower right", fontsize=10,
    facecolor=CARD, edgecolor=GRID_C, labelcolor=WHITE
)


# ── PLOT 2: Key Metric Cards ──────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_facecolor(BG)
ax_c.axis("off")
ax_c.set_title(f"Best Model  (K = {best_k_auto})",
               fontsize=14, fontweight="bold",
               color=WHITE, pad=12, loc="left")

bi              = int(np.argmax(accuracies))
bcm             = cms[bi]
tn, fp, fn, tp_val = bcm.ravel()
precision_best  = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else 0
recall_best     = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else 0

cards_data = [
    ("Accuracy",  f"{accuracies[bi]*100:.1f}%",     GREEN),
    ("Precision", f"{precision_best*100:.1f}%",      ACCENT),
    ("Recall",    f"{recall_best*100:.1f}%",         ACCENT2),
    ("Test Size", "154",                              MUTED),
]

for i, (lbl, val, c) in enumerate(cards_data):
    row, col = divmod(i, 2)
    x0 = col * 0.52
    y0 = 0.97 - row * 0.52
    rect = FancyBboxPatch(
        (x0, y0 - 0.42), 0.46, 0.38,
        boxstyle="round,pad=0.02",
        facecolor=CARD, edgecolor=c, linewidth=1.5,
        transform=ax_c.transAxes, clip_on=False
    )
    ax_c.add_patch(rect)
    ax_c.text(x0 + 0.23, y0 - 0.10, val,
              ha="center", va="center",
              fontsize=17, fontweight="bold", color=c,
              transform=ax_c.transAxes)
    ax_c.text(x0 + 0.23, y0 - 0.32, lbl,
              ha="center", va="center",
              fontsize=10, color=MUTED,
              transform=ax_c.transAxes)


# ── PLOT 3: Confusion Matrices (K=3,5,7) ─
for idx, (k, cm) in enumerate(zip(k_values, cms)):
    ax = fig.add_subplot(gs[1, idx])
    ttl = f"Confusion Matrix  K = {k}" + ("  ★" if k == best_k else "")
    ax.set_title(ttl, fontsize=12, fontweight="bold",
                 color=GREEN if k == best_k else WHITE,
                 pad=10, loc="left")

    cell_colors = [[GREEN, ACCENT2], [ACCENT2, GREEN]]
    cell_alpha  = [[0.25, 0.18], [0.18, 0.25]]
    labels_cm   = [["TN", "FP"], ["FN", "TP"]]
    vals        = cm.tolist()

    for r in range(2):
        for c in range(2):
            ch = cell_colors[r][c]
            al = cell_alpha[r][c]
            rect = FancyBboxPatch(
                (c + 0.05, 1 - r - 0.95), 0.9, 0.9,
                boxstyle="round,pad=0.03",
                facecolor=ch, alpha=al,
                edgecolor=ch, linewidth=1.5,
                transform=ax.transData
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, 1.5 - r, str(vals[r][c]),
                    ha="center", va="center",
                    fontsize=22, fontweight="bold", color=ch)
            ax.text(c + 0.5, 1.5 - r - 0.28, labels_cm[r][c],
                    ha="center", va="center",
                    fontsize=9, color=MUTED)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Pred: No", "Pred: Yes"], fontsize=10)
    ax.set_yticklabels(["Act: Yes", "Act: No"],   fontsize=10)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)


# ── PLOT 4: Manual Distance Bars ─────────
ax4 = fig.add_subplot(gs[2, :2])
ax4.set_title(
    f"Manual Euclidean Distances — Test Sample #0 vs 10 Training Samples",
    fontsize=12, fontweight="bold", color=WHITE, pad=12, loc="left"
)

idxs  = [d[0] for d in manual_sorted]
dvals = [d[1] for d in manual_sorted]
dlbls = [d[2] for d in manual_sorted]
y_pos = list(range(10))[::-1]

for i, (yp, dv, idx_, lbl) in enumerate(zip(y_pos, dvals, idxs, dlbls)):
    c  = ACCENT2 if lbl == 1 else GREEN
    al = 1.0 if i < best_k else 0.28
    ax4.barh(yp, dv, color=c, alpha=al, height=0.6, zorder=3)
    ntag    = f"  ← neighbor #{i+1}" if i < best_k else ""
    outcome = "Diabetic" if lbl == 1 else "Non-diabetic"
    ax4.text(dv + 0.04, yp,
             f"{dv:.4f}   Sample {idx_} — {outcome}{ntag}",
             va="center", fontsize=9,
             color=c, alpha=1.0 if al == 1.0 else 0.55)

ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"Rank {10-i}" for i in range(10)], fontsize=9)
ax4.set_xlabel("Euclidean Distance", fontsize=11)
ax4.set_xlim(0, max(dvals) + 1.4)
ax4.xaxis.grid(True, zorder=0)
ax4.set_axisbelow(True)
for sp in ax4.spines.values(): sp.set_edgecolor(GRID_C)
ax4.legend(
    handles=[mpatches.Patch(color=ACCENT2, label="Diabetic (1)"),
             mpatches.Patch(color=GREEN,   label="Non-diabetic (0)")],
    fontsize=10, facecolor=CARD, edgecolor=GRID_C, labelcolor=WHITE
)


# ── PLOT 5: All Four Distance Metrics ────
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_title(f"4 Distance Metrics  (K = {best_k})",
              fontsize=12, fontweight="bold",
              color=WHITE, pad=12, loc="left")

short_names  = ["Euclidean", "Manhattan", "Minkowski\n(p=3)", "Chebyshev"]
metric_colors = [GREEN, ACCENT, ORANGE, PINK]
m_pct = [a * 100 for a in metric_accs]

b = ax5.bar(range(4), m_pct,
            color=metric_colors, alpha=0.85,
            width=0.55, zorder=3, linewidth=0)

for bar, v, c in zip(b, m_pct, metric_colors):
    ax5.text(bar.get_x() + bar.get_width() / 2, v + 0.35,
             f"{v:.1f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=c)

best_metric_idx = int(np.argmax(m_pct))
ax5.text(b[best_metric_idx].get_x() + b[best_metric_idx].get_width() / 2,
         m_pct[best_metric_idx] / 2,
         "BEST", ha="center", va="center",
         fontsize=8, color=BG, fontweight="bold")

ax5.set_xticks(range(4))
ax5.set_xticklabels(short_names, fontsize=9)
ax5.set_ylim(60, 82)
ax5.set_ylabel("Accuracy (%)", fontsize=10)
ax5.yaxis.grid(True, zorder=0)
ax5.set_axisbelow(True)
for sp in ax5.spines.values(): sp.set_edgecolor(GRID_C)


# ── SAVE ─────────────────────────────────
plt.savefig("knn_results.png", dpi=170,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("\nDashboard saved as knn_results.png")

# =============================================================
# END OF CODE
# =============================================================
