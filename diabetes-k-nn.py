"""
============================================================
  KNN on Diabetes Dataset — Manual Implementation
============================================================
"""

import csv
import math
import random
import os
import sys
import heapq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['axes.unicode_minus'] = False
SECTION_WIDTH = 62

def sep(char='─', n=SECTION_WIDTH): return char * n
def section(title):
    print(f"\n{sep('═')}\n" + (' ' * ((SECTION_WIDTH - len(title) - 2) // 2)) + f" {title} \n{sep('═')}")
def subsection(title):
    print(f"\n{sep('─')}\n  {title}\n{sep('─')}")

def mean(values): return sum(values) / len(values) if values else 0.0
def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0: return 0.0
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0

# ── LOAD DATA ────────────────────────────────────────────────────────────────
try: script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: script_dir = os.getcwd()
csv_path = os.path.join(script_dir, 'diabetes-k-nn.csv')

FEAT_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
IMPUTE_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

raw_data = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader: raw_data.append({k: float(v) for k, v in row.items()})

section('DATA PREPROCESSING')
subsection('Zero-value scan (Proxy for Missing Data)')
total = len(raw_data)
print(f'  {"Feature":<28} {"Zero Count":>12} {"Percentage":>12}\n  {sep("-",52)}')
for col in FEAT_COLS:
    z = sum(1 for r in raw_data if r[col] == 0.0)
    flag = '  ← missing!' if col in IMPUTE_COLS and z > 0 else ''
    print(f'  {col:<28} {z:>12}   {100*z/total:>8.1f}%{flag}')

# ── SPLIT THEN IMPUTE ────────────────────────────────────────────────────────
random.seed(42)
shuffled = raw_data[:]
random.shuffle(shuffled)
split = int(0.8 * len(shuffled))
train_raw, test_raw = shuffled[:split], shuffled[split:]

subsection('Dataset Split (Before Imputation)')
print(f'  Total: {len(raw_data)} | Train: {len(train_raw)} (80%) | Test: {len(test_raw)} (20%)')

medians = {col: median([r[col] for r in train_raw if r[col] != 0.0]) for col in IMPUTE_COLS}
def impute_row(row):
    new_row = dict(row)
    for col in IMPUTE_COLS:
        if new_row[col] == 0.0: new_row[col] = medians[col]
    return new_row

train_imputed = [impute_row(r) for r in train_raw]
test_imputed  = [impute_row(r) for r in test_raw]

# ── MIN-MAX SCALING ──────────────────────────────────────────────────────────
feat_min = {col: min(r[col] for r in train_imputed) for col in FEAT_COLS}
feat_max = {col: max(r[col] for r in train_imputed) for col in FEAT_COLS}

def normalize_to_tuple(row):
    features = []
    for col in FEAT_COLS:
        rng = feat_max[col] - feat_min[col]
        features.append((row[col] - feat_min[col]) / rng if rng != 0 else 0.0)
    features.append(row['Outcome'])
    return tuple(features)

train_set = [normalize_to_tuple(r) for r in train_imputed]
test_set  = [normalize_to_tuple(r) for r in test_imputed]

subsection('Min-Max Normalisation Preview (First Row)')
print(f'  {"Feature":<28} {"Before":>18} {"After":>20}\n  {sep("-",68)}')
for i, col in enumerate(FEAT_COLS):
    print(f'  {col:<28} {train_imputed[0][col]:>18.4f} {train_set[0][i]:>20.6f}')

# ── KNN LOGIC ────────────────────────────────────────────────────────────────
section('KNN IMPLEMENTATION & EVALUATION')

def euclidean_distance(a, b): return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(FEAT_COLS))))
def knn_predict(train, test_row, k):
    distances = ((euclidean_distance(test_row, tr), tr[-1]) for tr in train)
    neighbours = heapq.nsmallest(k, distances, key=lambda x: x[0])
    votes = sum(int(label) for _, label in neighbours)
    return 1 if votes > k // 2 else 0, neighbours

subsection('Manual Distance Computation (Test Instance #0 vs Top 10)')
test_instance = test_set[0]
all_dists = [(euclidean_distance(test_instance, tr), tr) for tr in train_set]
all_dists.sort(key=lambda x: x[0])
top10 = all_dists[:10]

print(f'  {"#":<4} {"Distance":>10}  {"Step-by-step (Δ² per feature)"}\n  {sep("-",SECTION_WIDTH)}')
for rank, (dist, tr) in enumerate(top10, 1):
    squares = [(f, (test_instance[i] - tr[i]) ** 2) for i, f in enumerate(FEAT_COLS)]
    sq_str  = ' + '.join(f'{sq:.3f}' for _, sq in squares)
    total_sq = sum(sq for _, sq in squares)
    print(f'  #{rank:<3} d = {dist:>8.6f} = √({sq_str}) = √({total_sq:.4f})  [Label: {int(tr[-1])}]')

subsection('Model Evaluation Matrices (K = 3, 5, 7)')
K_VALUES = [3, 5, 7]
cm_results = {}
print(f'  {"K":<6} {"Accuracy":>10} {"TP":>6} {"TN":>6} {"FP":>6} {"FN":>6} {"Precision":>11} {"Recall":>9}\n  {sep("-",68)}')

for k in K_VALUES:
    preds, actuals = [], []
    for t in test_set:
        pred, _ = knn_predict(train_set, t, k)
        preds.append(pred)
        actuals.append(int(t[-1]))
    
    tp = sum(a == 1 and p == 1 for a, p in zip(actuals, preds))
    tn = sum(a == 0 and p == 0 for a, p in zip(actuals, preds))
    fp = sum(a == 0 and p == 1 for a, p in zip(actuals, preds))
    fn = sum(a == 1 and p == 0 for a, p in zip(actuals, preds))
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    
    cm_results[k] = dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec)
    print(f'  K={k:<4} {acc*100:>9.2f}%  {tp:>6} {tn:>6} {fp:>6} {fn:>6}  {prec*100:>9.2f}%  {rec*100:>7.2f}%')

best_k = max(cm_results, key=lambda k: cm_results[k]['acc'])
print(f'\n  ★ Best K = {best_k} with {cm_results[best_k]["acc"]*100:.2f}% Accuracy\n')

# ══════════════════════════════════════════════════════════════════════════════
#  BONUS — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

section('BONUS — GENERATING VISUALISATIONS')
print('\n  Building plots … (close the window to exit)')

data = train_imputed + test_imputed

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif'],
    'font.size'         : 10,
    'figure.facecolor'  : '#0F172A',
    'axes.facecolor'    : '#1E293B',
    'axes.edgecolor'    : '#334155',
    'axes.linewidth'    : 0.0,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.spines.left'  : False,
    'axes.spines.bottom': True,
    'axes.labelcolor'   : '#94A3B8',
    'xtick.color'       : '#64748B',
    'ytick.color'       : '#64748B',
    'grid.color'        : '#334155',
    'grid.linewidth'    : 0.8,
    'grid.alpha'        : 0.5,
    'legend.frameon'    : False,
    'legend.labelcolor' : '#E2E8F0',
    'text.color'        : '#F8FAFC'
})

P = {
    'cyan'   : '#22D3EE',
    'violet' : '#A78BFA',
    'rose'   : '#FB7185',
    'emerald': '#34D399',
    'amber'  : '#FBBF24',
    'bg'     : '#0F172A',
    'text'   : '#F8FAFC',
    'muted'  : '#94A3B8'
}

def style_card(ax, title):
    """Applies a clean, modern card aesthetic to an axes."""
    ax.set_title(title, fontsize=12, fontweight='bold', color=P['text'], pad=15, loc='left')
    ax.grid(True, axis='y', color=P['muted'], alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines['bottom'].set_color(P['muted'])
    ax.spines['bottom'].set_alpha(0.3)

# ── Figure Layout ─────────────────────────────────────────────────────────────
# layout='constrained' with no forced 'y' value prevents top clipping
fig = plt.figure(figsize=(16, 10), layout='constrained')
fig.patch.set_facecolor(P['bg'])
fig.suptitle('KNN · Diabetes Classification Dashboard', 
             fontsize=18, fontweight='bold', color=P['text'])

# 6-column grid to allow 3-item rows and 2-item rows to be perfectly symmetric
gs = GridSpec(3, 6, figure=fig)

# ══ ROW 1: Metrics (3 items, 2 cols each) ═════════════════════════════════════

# Plot 1: Accuracy
ax1 = fig.add_subplot(gs[0, 0:2])
style_card(ax1, 'Accuracy vs K')
accs = [cm_results[k]['acc'] * 100 for k in K_VALUES]
ax1.plot(K_VALUES, accs, color=P['cyan'], lw=3, marker='o', ms=8, zorder=4)
ax1.fill_between(K_VALUES, [min(accs) - 2]*len(accs), accs, color=P['cyan'], alpha=0.1, zorder=3)
ax1.axvline(best_k, color=P['emerald'], ls=':', lw=2, label=f'Best K={best_k}', zorder=5)
ax1.set_xticks(K_VALUES)
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(min(accs) - 2, max(accs) + 2)
ax1.legend(loc='lower right')

# Plot 2: Precision & Recall
ax2 = fig.add_subplot(gs[0, 2:4])
style_card(ax2, 'Precision & Recall')
precs = [cm_results[k]['prec'] * 100 for k in K_VALUES]
recs  = [cm_results[k]['rec']  * 100 for k in K_VALUES]
ax2.plot(K_VALUES, precs, color=P['violet'], lw=3, marker='o', label='Precision')
ax2.plot(K_VALUES, recs, color=P['rose'], lw=3, marker='o', label='Recall')
ax2.set_xticks(K_VALUES)
ax2.set_ylabel('%')
ax2.legend()

# Plot 3: Distances
ax3 = fig.add_subplot(gs[0, 4:6])
style_card(ax3, 'Top-10 Neighbour Distances')
ranks = list(range(1, 11))
dists_vals = [dist for dist, _ in top10]
colors = [P['rose'] if int(tr[-1]) == 1 else P['cyan'] for _, tr in top10]
ax3.bar(ranks, dists_vals, color=colors, alpha=0.85, width=0.6, zorder=3)
ax3.set_xticks(ranks)
ax3.set_xlabel('Neighbour rank')
ax3.set_ylabel('Distance')
ax3.legend(handles=[mpatches.Patch(color=P['cyan'], label='Non-diabetic'),
                    mpatches.Patch(color=P['rose'], label='Diabetic')])


# ══ ROW 2: Confusion Matrices (3 items, 2 cols each) ══════════════════════════

CM_STYLE = [
    (P['cyan'],   0.15), # TN
    (P['rose'],   0.15), # FP
    (P['amber'],  0.15), # FN
    (P['emerald'],0.15)  # TP
]

for idx, k in enumerate(K_VALUES):
    ax = fig.add_subplot(gs[1, idx*2 : (idx*2)+2]) # Spans cols: 0:2, 2:4, 4:6
    ax.set_facecolor(P['bg'])
    ax.set_aspect('equal')
    ax.axis('off')

    d = cm_results[k]
    vals = [d['tn'], d['fp'], d['fn'], d['tp']]
    coords = [(0, 1), (1, 1), (0, 0), (1, 0)]
    labels = ['TN', 'FP', 'FN', 'TP']

    for (cx, cy), v, lk, (fg_col, alpha) in zip(coords, vals, labels, CM_STYLE):
        ax.add_patch(plt.Rectangle((cx, cy), 0.92, 0.92, facecolor='#1E293B', edgecolor=fg_col, lw=2, zorder=1))
        ax.add_patch(plt.Rectangle((cx, cy), 0.92, 0.92, facecolor=fg_col, alpha=alpha, zorder=2))
        ax.text(cx + 0.46, cy + 0.55, f'{v}', ha='center', va='center', fontsize=22, fontweight='bold', color=P['text'])
        ax.text(cx + 0.46, cy + 0.25, f'{lk}', ha='center', va='center', fontsize=11, fontweight='bold', color=fg_col)

    ax.set_xlim(-0.05, 2); ax.set_ylim(-0.05, 2)
    ax.set_title(f'K = {k}  |  Acc: {d["acc"]*100:.1f}%', fontsize=12, pad=12, fontweight='bold', color=P['text'])


# ══ ROW 3: Distributions (2 items, 3 cols each for perfect symmetry) ══════════

g0 = [r['Glucose'] for r in data if r['Outcome'] == 0]
g1 = [r['Glucose'] for r in data if r['Outcome'] == 1]
b0 = [r['BMI'] for r in data if r['Outcome'] == 0]
b1 = [r['BMI'] for r in data if r['Outcome'] == 1]

# Plot 7: Glucose (Spans columns 0 to 3)
ax7 = fig.add_subplot(gs[2, 0:3])
style_card(ax7, 'Glucose Distribution')
ax7.hist(g0, bins=15, color=P['cyan'], alpha=0.5, label='Non-diabetic', zorder=3)
ax7.hist(g1, bins=15, color=P['rose'], alpha=0.5, label='Diabetic', zorder=4)
ax7.legend()

# Plot 8: BMI (Spans columns 3 to 6)
ax8 = fig.add_subplot(gs[2, 3:6])
style_card(ax8, 'BMI Distribution')
ax8.hist(b0, bins=15, color=P['cyan'], alpha=0.5, label='Non-diabetic', zorder=3)
ax8.hist(b1, bins=15, color=P['amber'], alpha=0.5, label='Diabetic', zorder=4)
ax8.legend()

# ── Save & Show ───────────────────────────────────────────────────────────────
plt.savefig(os.path.join(script_dir, 'knn_results_dark_modern.png'), dpi=150, bbox_inches='tight', facecolor=P['bg'])
print('\n  Saved: knn_results.png')
plt.show()

print('\n' + sep('═'))
print('  Done. All tasks completed successfully.')
print(sep('═') + '\n')