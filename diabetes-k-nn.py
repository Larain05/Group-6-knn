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
#  BONUS — LOGISTIC REGRESSION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

section('BONUS — LOGISTIC REGRESSION COMPARISON')

import math as _math

def sigmoid(z): return 1.0 / (1.0 + _math.exp(-max(-500, min(500, z))))

def lr_train(train, lr=0.1, epochs=1000):
    n_feat = len(FEAT_COLS)
    weights = [0.0] * n_feat
    bias = 0.0
    for _ in range(epochs):
        dw = [0.0] * n_feat
        db = 0.0
        for row in train:
            feats = list(row[:n_feat])
            label = row[-1]
            z     = sum(w * x for w, x in zip(weights, feats)) + bias
            pred  = sigmoid(z)
            err   = pred - label
            for i in range(n_feat):
                dw[i] += err * feats[i]
            db += err
        m = len(train)
        weights = [w - lr * d / m for w, d in zip(weights, dw)]
        bias   -= lr * db / m
    return weights, bias

def lr_predict(row, weights, bias, threshold=0.5):
    n_feat = len(FEAT_COLS)
    z = sum(w * x for w, x in zip(weights, list(row[:n_feat]))) + bias
    return 1 if sigmoid(z) >= threshold else 0

subsection('Training Logistic Regression (gradient descent, 1000 epochs, lr=0.1)')
lr_weights, lr_bias = lr_train(train_set, lr=0.1, epochs=1000)

print(f'\n  {"Feature":<28} {"Weight":>12}')
print(f'  {sep("-", 42)}')
for feat, w in zip(FEAT_COLS, lr_weights):
    print(f'  {feat:<28} {w:>12.6f}')
print(f'  {"Bias":<28} {lr_bias:>12.6f}')

# ── Evaluate LR ───────────────────────────────────────────────────────────────
lr_preds   = [lr_predict(t, lr_weights, lr_bias) for t in test_set]
lr_actuals = [int(t[-1]) for t in test_set]

lr_tp = sum(a == 1 and p == 1 for a, p in zip(lr_actuals, lr_preds))
lr_tn = sum(a == 0 and p == 0 for a, p in zip(lr_actuals, lr_preds))
lr_fp = sum(a == 0 and p == 1 for a, p in zip(lr_actuals, lr_preds))
lr_fn = sum(a == 1 and p == 0 for a, p in zip(lr_actuals, lr_preds))
lr_total = lr_tp + lr_tn + lr_fp + lr_fn
lr_acc  = (lr_tp + lr_tn) / lr_total if lr_total else 0.0
lr_prec = lr_tp / (lr_tp + lr_fp) if (lr_tp + lr_fp) else 0.0
lr_rec  = lr_tp / (lr_tp + lr_fn) if (lr_tp + lr_fn) else 0.0
lr_f1   = 2 * lr_prec * lr_rec / (lr_prec + lr_rec) if (lr_prec + lr_rec) else 0.0

subsection('Head-to-Head Comparison: Best KNN vs Logistic Regression')
bk = cm_results[best_k]
knn_f1 = (2 * bk['prec'] * bk['rec'] / (bk['prec'] + bk['rec'])
          if (bk['prec'] + bk['rec']) else 0.0)

print(f'\n  {"Metric":<18} {"KNN (K=" + str(best_k) + ")":>14} {"Logistic Reg.":>16}')
print(f'  {sep("-", 50)}')
print(f'  {"Accuracy":<18} {bk["acc"]*100:>13.2f}%  {lr_acc*100:>14.2f}%')
print(f'  {"Precision":<18} {bk["prec"]*100:>13.2f}%  {lr_prec*100:>14.2f}%')
print(f'  {"Recall":<18} {bk["rec"]*100:>13.2f}%  {lr_rec*100:>14.2f}%')
print(f'  {"F1 Score":<18} {knn_f1*100:>13.2f}%  {lr_f1*100:>14.2f}%')
print(f'  {"TP":<18} {bk["tp"]:>14}   {lr_tp:>14}')
print(f'  {"TN":<18} {bk["tn"]:>14}   {lr_tn:>14}')
print(f'  {"FP":<18} {bk["fp"]:>14}   {lr_fp:>14}')
print(f'  {"FN":<18} {bk["fn"]:>14}   {lr_fn:>14}')

lr_results = dict(tp=lr_tp, tn=lr_tn, fp=lr_fp, fn=lr_fn,
                  acc=lr_acc, prec=lr_prec, rec=lr_rec, f1=lr_f1)
knn_f1_best = knn_f1

# ══════════════════════════════════════════════════════════════════════════════
#  BONUS — VISUALISATIONS (ADJUSTED HIERARCHY)
# ══════════════════════════════════════════════════════════════════════════════

section('BONUS — GENERATING VISUALISATIONS')
print('\n  Building plots … (close the window to exit)')

data = train_imputed + test_imputed

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
    'cyan': '#22D3EE', 'violet': '#A78BFA', 'rose': '#FB7185', 
    'emerald': '#34D399', 'amber': '#FBBF24', 'bg': '#0F172A', 
    'text': '#F8FAFC', 'muted': '#94A3B8'
}

def style_card(ax, title):
    ax.set_title(title, fontsize=12, fontweight='bold', color=P['text'], pad=15, loc='left')
    ax.grid(True, axis='y', color=P['muted'], alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines['bottom'].set_color(P['muted'])
    ax.spines['bottom'].set_alpha(0.3)

fig = plt.figure(figsize=(16, 15), layout='constrained')
fig.patch.set_facecolor(P['bg'])
fig.suptitle('KNN · Diabetes Classification Dashboard', fontsize=18, fontweight='bold', color=P['text'])

# GridSpec with adjusted height ratios: The last row (bonus) is explicitly shorter (0.75)
gs = GridSpec(4, 6, figure=fig, height_ratios=[1.2, 1.2, 1.2, 0.75])

# ══ ROW 1: Metrics (3 items, 2 cols each) ═════════════════════════════════════
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

ax2 = fig.add_subplot(gs[0, 2:4])
style_card(ax2, 'Precision & Recall')
precs = [cm_results[k]['prec'] * 100 for k in K_VALUES]
recs  = [cm_results[k]['rec']  * 100 for k in K_VALUES]
ax2.plot(K_VALUES, precs, color=P['violet'], lw=3, marker='o', label='Precision')
ax2.plot(K_VALUES, recs, color=P['rose'], lw=3, marker='o', label='Recall')
ax2.set_xticks(K_VALUES)
ax2.set_ylabel('%')
ax2.legend()

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
CM_STYLE = [(P['cyan'], 0.15), (P['rose'], 0.15), (P['amber'], 0.15), (P['emerald'], 0.15)]
for idx, k in enumerate(K_VALUES):
    ax = fig.add_subplot(gs[1, idx*2 : (idx*2)+2])
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


# ══ ROW 3: Distributions (Promoted to emphasize dataset) ══════════════════════
g0 = [r['Glucose'] for r in data if r['Outcome'] == 0]
g1 = [r['Glucose'] for r in data if r['Outcome'] == 1]
b0 = [r['BMI'] for r in data if r['Outcome'] == 0]
b1 = [r['BMI'] for r in data if r['Outcome'] == 1]

ax7 = fig.add_subplot(gs[2, 0:3])
style_card(ax7, 'Glucose Distribution')
ax7.hist(g0, bins=15, color=P['cyan'], alpha=0.5, label='Non-diabetic', zorder=3)
ax7.hist(g1, bins=15, color=P['rose'], alpha=0.5, label='Diabetic', zorder=4)
ax7.legend()

ax8 = fig.add_subplot(gs[2, 3:6])
style_card(ax8, 'BMI Distribution')
ax8.hist(b0, bins=15, color=P['cyan'], alpha=0.5, label='Non-diabetic', zorder=3)
ax8.hist(b1, bins=15, color=P['amber'], alpha=0.5, label='Diabetic', zorder=4)
ax8.legend()


# ══ ROW 4: Logistic Regression Bonus (Demoted to bottom) ══════════════════════
ax_comp1 = fig.add_subplot(gs[3, 0:3])
style_card(ax_comp1, 'Accuracy & F1 (KNN vs Log. Reg.)')
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
knn_vals = [bk['acc']*100, bk['prec']*100, bk['rec']*100, knn_f1_best*100]
lr_vals  = [lr_acc*100, lr_prec*100, lr_rec*100, lr_f1*100]
x = range(len(metrics_labels))
w = 0.35
bars1 = ax_comp1.bar([i - w/2 for i in x], knn_vals, width=w, color=P['cyan'],   alpha=0.6, label=f'KNN', zorder=3)
bars2 = ax_comp1.bar([i + w/2 for i in x], lr_vals,  width=w, color=P['violet'], alpha=0.6, label='Log. Reg.', zorder=3)
for bar in list(bars1) + list(bars2):
    ax_comp1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{bar.get_height():.1f}%', ha='center', va='bottom',
                  fontsize=8, color=P['muted'])
ax_comp1.set_xticks(list(x))
ax_comp1.set_xticklabels(metrics_labels)
ax_comp1.set_ylabel('%')
ax_comp1.set_ylim(0, 100)
ax_comp1.legend(loc='upper right')

ax_comp2 = fig.add_subplot(gs[3, 3:6])
style_card(ax_comp2, 'Matrix Counts (KNN vs Log. Reg.)')
cm_labels = ['TP', 'TN', 'FP', 'FN']
knn_cm = [bk['tp'], bk['tn'], bk['fp'], bk['fn']]
lr_cm  = [lr_tp,    lr_tn,    lr_fp,    lr_fn]
x2 = range(len(cm_labels))
bars3 = ax_comp2.bar([i - w/2 for i in x2], knn_cm, width=w, color=P['cyan'],   alpha=0.6, label=f'KNN', zorder=3)
bars4 = ax_comp2.bar([i + w/2 for i in x2], lr_cm,  width=w, color=P['violet'], alpha=0.6, label='Log. Reg.', zorder=3)
for bar in list(bars3) + list(bars4):
    ax_comp2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  str(int(bar.get_height())), ha='center', va='bottom',
                  fontsize=9, color=P['muted'])
ax_comp2.set_xticks(list(x2))
ax_comp2.set_xticklabels(cm_labels)
ax_comp2.set_ylabel('Count')
ax_comp2.legend(loc='upper right')

# ── Save & Show ───────────────────────────────────────────────────────────────
plt.savefig(os.path.join(script_dir, 'knn_results.png'), dpi=150, bbox_inches='tight', facecolor=P['bg'])
print('\n  Saved: knn_results.png')
plt.show()

print('\n' + sep('═'))
print('  Done. All tasks completed successfully.')
print(sep('═') + '\n')