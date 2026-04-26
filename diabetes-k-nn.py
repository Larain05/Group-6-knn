# =========================================
# KNN DIABETES CLASSIFICATION
# University of Southern Mindanao
# Computational Science for Computer Science
# =========================================

# ── 1. IMPORT LIBRARIES ──────────────────
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')


# ── 2. LOAD DATASET ──────────────────────
df = pd.read_csv("diabetes-k-nn.csv")

print("=" * 55)
print("FIRST 5 ROWS:")
print("=" * 55)
print(df.head())

print("\nDATASET INFO:")
print(df.info())

print("\nSUMMARY STATISTICS BEFORE PREPROCESSING:")
print(df.describe().round(2))


# ── 3. DATA PREPROCESSING ────────────────

# Replace biologically impossible zero values with NaN.
# Columns like Glucose, BloodPressure, Insulin cannot be
# zero in a living patient — they represent missing data.
columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_with_zero:
    df[col] = df[col].replace(0, np.nan)

# Fill missing values with the column median.
# Median is preferred over mean because it is not affected
# by extreme outlier values (e.g., very high Insulin readings).
for col in columns_with_zero:
    df[col] = df[col].fillna(df[col].median())

print("\nSUMMARY AFTER HANDLING MISSING VALUES:")
print(df.describe().round(2))


# ── 4. SPLIT FEATURES AND TARGET ─────────
X = df.drop("Outcome", axis=1)   # 8 input features
y = df["Outcome"]                 # Target: 0 = Non-diabetic, 1 = Diabetic

feature_names = list(X.columns)


# ── 5. FEATURE SCALING ───────────────────
# KNN relies entirely on distance. Without scaling, features
# with large ranges (e.g. Insulin: 0–800) dominate features
# with small ranges (e.g. Age: 20–80), distorting results.
# StandardScaler transforms each feature to mean=0, std=1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 6. TRAIN-TEST SPLIT ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")


# ── 7. TRAIN KNN MODELS (K = 3, 5, 7) ───
k_values   = [3, 5, 7]
accuracies = []
cms        = []

print("\n" + "=" * 55)
print("KNN RESULTS")
print("=" * 55)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    accuracies.append(acc)
    cms.append(cm)

    print(f"\nK = {k}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")


# ── 8. BEST K VALUE ──────────────────────
best_k   = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\nBest K value : {best_k}  (Accuracy: {best_acc*100:.2f}%)")


# ── 9. MANUAL DISTANCE COMPUTATION ───────
# Part 3 requirement: for ONE selected test instance,
# manually compute Euclidean distance to at least 10
# training samples and show step-by-step computation.
#
# Euclidean distance formula:
#   d = sqrt( sum( (xi - yi)^2 ) )  for all features i

test_instance = X_test[0]

print("\n" + "=" * 55)
print("MANUAL EUCLIDEAN DISTANCE — STEP-BY-STEP")
print("=" * 55)
print(f"\nSelected test instance (scaled feature values):")
for fname, fval in zip(feature_names, test_instance):
    print(f"  {fname:<28}: {fval:>8.4f}")

distances = []

for i in range(10):
    train_sample = X_train[i]

    print(f"\n--- Distance to Training Sample {i} ---")
    squared_diffs = []
    for j, (fname, tv, trv) in enumerate(
            zip(feature_names, test_instance, train_sample)):
        diff        = tv - trv
        diff_sq     = diff ** 2
        squared_diffs.append(diff_sq)
        print(f"  {fname:<28}: ({tv:.4f} - {trv:.4f})^2 = {diff_sq:.4f}")

    sum_sq   = sum(squared_diffs)
    distance = np.sqrt(sum_sq)
    label    = int(y_train.iloc[i])

    print(f"  Sum of squared differences : {sum_sq:.4f}")
    print(f"  Euclidean distance (sqrt)  : {distance:.4f}")
    print(f"  Actual label               : {'Diabetic (1)' if label==1 else 'Non-diabetic (0)'}")

    distances.append((i, distance, label))

# Sort by distance to find nearest neighbors
distances_sorted = sorted(distances, key=lambda x: x[1])

print("\n" + "=" * 55)
print("SORTED DISTANCES — NEAREST NEIGHBORS")
print("=" * 55)
for rank, (idx, dist, lbl) in enumerate(distances_sorted, 1):
    tag = f"  ← neighbor #{rank}" if rank <= 5 else ""
    print(f"  Rank {rank:>2} | Sample {idx:>2} | "
          f"dist={dist:.4f} | "
          f"{'Diabetic (1)' if lbl==1 else 'Non-diabetic (0)'}{tag}")

# Majority vote using best K
top_k_labels = [d[2] for d in distances_sorted[:best_k]]
vote         = max(set(top_k_labels), key=top_k_labels.count)
n_diabetic   = sum(top_k_labels)
n_non        = best_k - n_diabetic

print(f"\nTop {best_k} neighbors vote:")
print(f"  Non-diabetic (0): {n_non} vote(s)")
print(f"  Diabetic     (1): {n_diabetic} vote(s)")
print(f"  → Predicted class: {'Diabetic (1)' if vote==1 else 'Non-diabetic (0)'}")


# ── 10. BONUS: LOGISTIC REGRESSION ───────
print("\n" + "=" * 55)
print("BONUS: KNN vs LOGISTIC REGRESSION")
print("=" * 55)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_cm   = confusion_matrix(y_test, lr_pred)

print(f"\nLogistic Regression")
print(f"  Accuracy : {lr_acc:.4f}  ({lr_acc*100:.2f}%)")
print(f"  Confusion Matrix:")
print(f"    TN={lr_cm[0,0]}  FP={lr_cm[0,1]}")
print(f"    FN={lr_cm[1,0]}  TP={lr_cm[1,1]}")

print(f"\nKNN (best K={best_k})")
print(f"  Accuracy : {best_acc:.4f}  ({best_acc*100:.2f}%)")
winner = "KNN" if best_acc >= lr_acc else "Logistic Regression"
print(f"\nWinner: {winner}")


# ── 11. AESTHETIC RESULTS DASHBOARD ──────
BG      = "#0D0F1A"
CARD    = "#13162A"
ACCENT  = "#6C63FF"
ACCENT2 = "#FF6584"
GREEN   = "#2DD4BF"
MUTED   = "#8B8FA8"
WHITE   = "#F0F2FF"
GRID_C  = "#1E2038"
ORANGE  = "#FFB347"

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

fig = plt.figure(figsize=(18, 15), facecolor=BG)

fig.text(0.5, 0.977,
         "KNN Diabetes Classification — Results Dashboard",
         ha="center", va="top",
         fontsize=20, fontweight="bold", color=WHITE)
fig.text(0.5, 0.958,
         "Dataset: 768 patients  |  Train: 614  |  Test: 154  |  Features: 8",
         ha="center", va="top", fontsize=12, color=MUTED)

gs = GridSpec(3, 3, figure=fig,
              top=0.94, bottom=0.05,
              left=0.06, right=0.97,
              hspace=0.52, wspace=0.36)


# ── PLOT 1: Accuracy vs K ─────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_title("Accuracy vs K Value", fontsize=14,
              fontweight="bold", color=WHITE, pad=12, loc="left")

pct        = [a * 100 for a in accuracies]
bar_colors = [GREEN if k == best_k else ACCENT for k in k_values]

bars = ax1.bar(k_values, pct, width=0.55,
               color=bar_colors, alpha=0.85, zorder=3, linewidth=0)

for bar, v, k in zip(bars, pct, k_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
             f"{v:.1f}%", ha="center", va="bottom",
             fontsize=13, fontweight="bold",
             color=GREEN if k == best_k else ACCENT)
    if k == best_k:
        ax1.text(bar.get_x() + bar.get_width() / 2, v / 2,
                 "BEST", ha="center", va="center",
                 fontsize=10, color=BG, fontweight="bold")

ax1.plot(k_values, pct, color=WHITE, lw=1.5, ls="--", alpha=0.4, zorder=4)
ax1.scatter(k_values, pct, color=WHITE, s=70, zorder=5)
ax1.set_xticks(k_values)
ax1.set_xticklabels([f"K = {k}" for k in k_values], fontsize=12)
ax1.set_ylim(65, 80)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.yaxis.grid(True, zorder=0)
ax1.set_axisbelow(True)
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_C)
ax1.legend(
    handles=[mpatches.Patch(color=GREEN,  label=f"Best K = {best_k}"),
             mpatches.Patch(color=ACCENT, label="Other K values")],
    loc="lower right", fontsize=10,
    facecolor=CARD, edgecolor=GRID_C, labelcolor=WHITE
)


# ── PLOT 2: Key Metric Cards ──────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_facecolor(BG)
ax_c.axis("off")
ax_c.set_title(f"Key Metrics  (K = {best_k})", fontsize=14,
               fontweight="bold", color=WHITE, pad=12, loc="left")

bi  = np.argmax(accuracies)
bcm = cms[bi]
tn, fp, fn, tp_val = bcm.ravel()

cards_data = [
    ("Accuracy",  f"{accuracies[bi]*100:.1f}%",       GREEN),
    ("Precision", f"{tp_val/(tp_val+fp)*100:.1f}%",   ACCENT),
    ("Recall",    f"{tp_val/(tp_val+fn)*100:.1f}%",   ACCENT2),
    ("Test Size", "154",                               MUTED),
]

for i, (label, val, c) in enumerate(cards_data):
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
    ax_c.text(x0 + 0.23, y0 - 0.32, label,
              ha="center", va="center",
              fontsize=10, color=MUTED,
              transform=ax_c.transAxes)


# ── PLOT 3: Confusion Matrices (all K) ───
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
                    ha="center", va="center", fontsize=9, color=MUTED)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Pred: No", "Pred: Yes"], fontsize=10)
    ax.set_yticklabels(["Act: Yes", "Act: No"],   fontsize=10)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_C)


# ── PLOT 4: Manual Distance Bars ─────────
ax4 = fig.add_subplot(gs[2, :2])
ax4.set_title(
    "Manual Euclidean Distances — Test Sample vs 10 Training Samples",
    fontsize=12, fontweight="bold", color=WHITE, pad=12, loc="left"
)

idxs  = [d[0] for d in distances_sorted]
dvals = [d[1] for d in distances_sorted]
dlbls = [d[2] for d in distances_sorted]
y_pos = list(range(10))[::-1]

for i, (yp, dv, idx_, lbl) in enumerate(zip(y_pos, dvals, idxs, dlbls)):
    c  = ACCENT2 if lbl == 1 else GREEN
    al = 1.0 if i < best_k else 0.28
    ax4.barh(yp, dv, color=c, alpha=al, height=0.6, zorder=3)
    ntag    = f"  ← #{i+1} neighbor" if i < best_k else ""
    outcome = "Diabetic" if lbl == 1 else "Non-diabetic"
    ax4.text(dv + 0.04, yp,
             f"{dv:.4f}   Sample {idx_} — {outcome}{ntag}",
             va="center", fontsize=9,
             color=c, alpha=1.0 if al == 1.0 else 0.5)

ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"Rank {10-i}" for i in range(10)], fontsize=9)
ax4.set_xlabel("Euclidean Distance", fontsize=11)
ax4.set_xlim(0, max(dvals) + 1.2)
ax4.xaxis.grid(True, zorder=0)
ax4.set_axisbelow(True)
for sp in ax4.spines.values(): sp.set_edgecolor(GRID_C)
ax4.legend(
    handles=[mpatches.Patch(color=ACCENT2, label="Diabetic (1)"),
             mpatches.Patch(color=GREEN,   label="Non-diabetic (0)")],
    fontsize=10, facecolor=CARD, edgecolor=GRID_C, labelcolor=WHITE
)


# ── PLOT 5: KNN vs Logistic Regression ───
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_title("KNN vs Logistic Regression\n(Bonus Comparison)",
              fontsize=11, fontweight="bold", color=WHITE,
              pad=10, loc="left")

models      = [f"KNN\n(K={best_k})", "Logistic\nRegression"]
model_accs  = [best_acc * 100, lr_acc * 100]
model_colors= [GREEN, ORANGE]

b = ax5.bar(models, model_accs, color=model_colors,
            alpha=0.85, width=0.45, zorder=3, linewidth=0)

for bar, v, c in zip(b, model_accs, model_colors):
    ax5.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
             f"{v:.1f}%", ha="center", va="bottom",
             fontsize=13, fontweight="bold", color=c)

win_idx = 0 if best_acc >= lr_acc else 1
ax5.text(b[win_idx].get_x() + b[win_idx].get_width() / 2,
         model_accs[win_idx] / 2,
         "WINNER", ha="center", va="center",
         fontsize=9, color=BG, fontweight="bold")

ax5.set_ylim(60, 85)
ax5.set_ylabel("Accuracy (%)", fontsize=10)
ax5.yaxis.grid(True, zorder=0)
ax5.set_axisbelow(True)
for sp in ax5.spines.values(): sp.set_edgecolor(GRID_C)


# ── SAVE ─────────────────────────────────
plt.savefig("knn_results.png", dpi=170, bbox_inches="tight", facecolor=BG)
plt.show()
print("\nDashboard saved as knn_results.png")

# =========================================
# END OF CODE
# =========================================
