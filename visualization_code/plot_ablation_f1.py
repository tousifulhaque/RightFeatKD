import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication-quality defaults
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

BASE = "/home/bgu9/icpr_2026/FeatureKD/exps/smartfall_kd/hparam"


def read_avg_f1(path):
    df = pd.read_csv(path, index_col=0)
    avg_row = df[df["test_subject"] == "Average"]
    return float(avg_row["f1_score"].values[0])


def load_hparam(subdir, prefix, param_re):
    """Return sorted (param_value, f1) pairs from subdirs matching prefix+param_re."""
    folder = os.path.join(BASE, subdir)
    results = []
    for name in os.listdir(folder):
        m = re.fullmatch(prefix + param_re, name)
        if m:
            val = float(m.group(1))
            f1 = read_avg_f1(os.path.join(folder, name, "scores.csv"))
            results.append((val, f1))
    results.sort()
    return [r[0] for r in results], [r[1] for r in results]


focal_x,  focal_f1  = load_hparam("focal",       r"watch_acc_gamma",  r"(\d*\.?\d+)")
temp_x,   temp_f1   = load_hparam("temperature",  r"watch_acc_t",      r"(\d*\.?\d+)")
weight_x, weight_f1 = load_hparam("weights",      r"watch_acc_weight", r"(\d*\.?\d+)")

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

color = '#2E5C8A'
marker_size = 8
line_width = 2.5
marker_edge_width = 1.5

configs = [
    {
        'ax': axes[0],
        'x': focal_x,
        'y': focal_f1,
        'xlabel': r'Focal Loss $\gamma$',
        'title': 'Focal Loss',
        'xticks': focal_x,
    },
    {
        'ax': axes[1],
        'x': temp_x,
        'y': temp_f1,
        'xlabel': r'Distillation Temperature $T$',
        'title': 'Distillation Temperature',
        'xticks': temp_x,
    },
    {
        'ax': axes[2],
        'x': weight_x,
        'y': weight_f1,
        'xlabel': r'KD Loss Weight $\beta$',
        'title': 'KD Loss Weight',
        'xticks': weight_x,
    },
]

all_f1 = focal_f1 + temp_f1 + weight_f1
y_min = min(all_f1) - 2
y_max = max(all_f1) + 2

for config in configs:
    ax = config['ax']
    x  = config['x']
    y  = config['y']

    ax.plot(x, y, 'o-', color=color, linewidth=line_width,
            markersize=marker_size, markerfacecolor=color,
            markeredgecolor='white', markeredgewidth=marker_edge_width,
            label='F1 Score', zorder=3)

    # for xi, yi in zip(x, y):
    #     ax.text(xi, yi + 0.25, f'{yi:.1f}', ha='center', va='bottom',
    #             fontsize=10, color=color, fontweight='bold')

    ax.set_xlabel(config['xlabel'], fontsize=16)
    ax.set_ylabel('F1 Score (%)', fontsize=16)
    #ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=10)
    ax.set_xticks(config['xticks'])
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(width=0.8, color='#333333')

plt.tight_layout()

out_dir = "/home/bgu9/icpr_2026/FeatureKD/plots/"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'ablation_study.pdf'),
            dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig(os.path.join(out_dir, 'ablation_study.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.05)
print("Saved ablation_study_neurips.pdf/.png")
