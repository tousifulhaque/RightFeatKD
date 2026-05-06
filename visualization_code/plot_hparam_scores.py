import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12

BASE = "/home/bgu9/icpr_2026/FeatureKD/exps/smartfall_fall_kd/student/hparam/student"
EXPS = {
    32:  "watch_acc_32",
    64:  "watch_acc_64",
    128: "watch_acc_128",
    256: "watch_acc_256",
}
METRICS       = ["accuracy", "f1_score", "precision", "recall"]
METRIC_LABELS = ["Accuracy", "F1 Score", "Precision", "Recall"]

# compute average per dimension per metric
dims = sorted(EXPS.keys())
avgs = {m: [] for m in METRICS}
for dim in dims:
    path = os.path.join(BASE, EXPS[dim], "scores.csv")
    df = pd.read_csv(path, index_col=0)
    df = df[df["test_subject"] != "Average"].copy()
    for m in METRICS:
        avgs[m].append(df[m].astype(float).mean())

embedding_dims = dims
accuracy  = avgs["accuracy"]
f1_score  = avgs["f1_score"]
recall = avgs["precision"]
precision    = avgs["recall"]

out_dir = "/home/bgu9/icpr_2026/FeatureKD/plots"
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

colors = {
    'accuracy':  '#0173B2',
    'f1':        '#DE8F05',
    'precision': '#029E73',
    'recall':    '#CC78BC',
}

line_width  = 2.0
marker_size = 7

ax.plot(embedding_dims, accuracy,  'o-', linewidth=line_width,
        markersize=marker_size, color=colors['accuracy'],
        label='Accuracy',  markerfacecolor='white', markeredgewidth=1.5)

ax.plot(embedding_dims, f1_score,  's-', linewidth=line_width,
        markersize=marker_size, color=colors['f1'],
        label='F1 Score',  markerfacecolor='white', markeredgewidth=1.5)

ax.plot(embedding_dims, precision, '^-', linewidth=line_width,
        markersize=marker_size, color=colors['precision'],
        label='Precision', markerfacecolor='white', markeredgewidth=1.5)

ax.plot(embedding_dims, recall,    'd-', linewidth=line_width,
        markersize=marker_size, color=colors['recall'],
        label='Recall',    markerfacecolor='white', markeredgewidth=1.5)

ax.set_xlabel('Embedding Dimension', fontsize=16 )
ax.set_ylabel('Score (%)',fontsize = 16)
ax.set_xticks(embedding_dims)
ax.set_xticklabels(embedding_dims)
ax.set_ylim(60, 90)
ax.set_yticks(np.arange(60, 91, 5))
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.95, ncol=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "hparam_avg_line.pdf"), format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hparam_avg_line.png"), format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: hparam_avg_line.pdf/.png")
print(f"Output dir: {out_dir}")
