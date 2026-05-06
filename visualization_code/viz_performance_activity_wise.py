import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
    'figure.figsize': (10, 7),
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Data from the plot
activities = [
    'Putonoff', 'Pickup', 'Step Up', 'Walking', 'Waving',
    'Sit and Stand', 'Washing', 'Drinking', 'Sweeping',
    'BackFall', 'Frontfall', 'LeftFall', 'Rightfall', 'RotateFall'
]

baseline_counts = [13, 23, 16, 23, 12, 12, 22, 16, 25, 15, 20, 9, 16, 13]
rightfeat_counts = [12, 25, 20, 24, 24, 19, 21, 24, 25, 22, 21, 20, 21, 23]

# Create horizontal bar chart
y = np.arange(len(activities))
height = 0.35

fig, ax = plt.subplots()

# Modern academic color scheme - Teal & Coral
baseline_color = '#2A9D8F'  # Teal
rightfeat_color = '#E76F51'  # Coral

bars1 = ax.barh(y + height/2, baseline_counts, height, label='Baseline',
                color=baseline_color, edgecolor='black', linewidth=0.8)
bars2 = ax.barh(y - height/2, rightfeat_counts, height, label='RightFeatKD',
                color=rightfeat_color, edgecolor='black', linewidth=0.8)

# Add value labels at the end of bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{int(value)}',
                ha='left', va='center', fontsize=10, fontweight='bold')

add_value_labels(bars1, baseline_counts)
add_value_labels(bars2, rightfeat_counts)

# Styling
ax.set_xlabel('Count out of 25', fontweight='bold')
ax.set_ylabel('Activities', fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(activities)
ax.set_xlim(0, 30)
ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False)
ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Invert y-axis to match original plot order
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('plots/performance_comparison_activity_wise.pdf', format='pdf', bbox_inches='tight')
plt.savefig('plots/performance_comparison_activity_wise.png', format='png', bbox_inches='tight')
print("Activity-wise plot saved successfully!")
print(f"\nTotal correct predictions:")
print(f"Baseline: {sum(baseline_counts)}/350 ({sum(baseline_counts)/350*100:.2f}%)")
print(f"RightFeatKD: {sum(rightfeat_counts)}/350 ({sum(rightfeat_counts)/350*100:.2f}%)")
print(f"Improvement: +{sum(rightfeat_counts) - sum(baseline_counts)} predictions")