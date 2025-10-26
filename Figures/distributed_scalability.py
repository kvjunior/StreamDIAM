import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Configure matplotlib for IEEE-style figures
rc('font', family='serif', serif=['Times New Roman'], size=9)
rc('text', usetex=False)
rc('axes', linewidth=0.8)
rc('grid', linewidth=0.4, alpha=0.3)

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

# ========== LEFT PANEL: Strong Scaling Efficiency ==========

# Data from Table: Distributed Scaling Analysis
gpu_counts = np.array([1, 2, 4, 8])
dataset_labels = ['100K', '500K', '1M', '5M', '10M']

# Throughput data (TPS) for each dataset
throughput_data = {
    '100K': [13247, 24836, 45921, 87342],
    '500K': [10284, 19731, 37548, 71263],
    '1M': [8421, 16287, 30874, 59847],
    '5M': [4687, 8974, 16943, 32784],
    '10M': [2841, 5436, 10247, 19738]
}

# Scaling efficiency data
efficiency_data = {
    '100K': [1.0, 0.938, 0.866, 0.824],
    '500K': [1.0, 0.959, 0.913, 0.866],
    '1M': [1.0, 0.967, 0.917, 0.889],
    '5M': [1.0, 0.957, 0.904, 0.874],
    '10M': [1.0, 0.957, 0.902, 0.869]
}

# Color scheme for different datasets
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

# Plot throughput scaling with ideal linear scaling reference
for idx, (label, color, marker) in enumerate(zip(dataset_labels, colors, markers)):
    throughput = np.array(throughput_data[label])
    # Normalize to 1 GPU baseline for clarity
    normalized_throughput = throughput / throughput[0]
    ax1.plot(gpu_counts, normalized_throughput, color=color, marker=marker, 
            markersize=5, linewidth=1.5, label=f'{label} nodes', 
            markerfacecolor='white', markeredgewidth=1.2, alpha=0.85)

# Plot ideal linear scaling (dashed line)
ideal_scaling = gpu_counts / gpu_counts[0]
ax1.plot(gpu_counts, ideal_scaling, 'k--', linewidth=1.0, alpha=0.5, 
        label='Ideal Linear', zorder=1)

# Add mean efficiency annotation
mean_efficiency = 0.874
ax1.text(6.5, 6.0, f'Mean Efficiency\n87.4%', fontsize=8, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                 edgecolor='gray', linewidth=0.8, alpha=0.9))

# Configure left panel
ax1.set_xlabel('Number of GPUs', fontsize=10, fontweight='normal')
ax1.set_ylabel('Speedup (relative to 1 GPU)', fontsize=10, fontweight='normal')
ax1.set_xticks(gpu_counts)
ax1.set_xticklabels(['1', '2', '4', '8'])
ax1.set_xlim(0.5, 8.5)
ax1.set_ylim(0, 9)
ax1.grid(True, alpha=0.3, linewidth=0.4, linestyle='--', zorder=0)
ax1.legend(loc='upper left', fontsize=7, frameon=True, fancybox=False, 
          edgecolor='black', framealpha=0.95, ncol=1)
ax1.set_title('(a) Strong Scaling Efficiency', fontsize=10, fontweight='normal', pad=8)

# ========== RIGHT PANEL: Communication Overhead Breakdown ==========

# Data for communication breakdown across GPU counts
# Based on text: gradient sync 58%, barrier sync 27%, parameter broadcast 15%
# Communication overhead varies by GPU count

gpu_counts_comm = [1, 2, 4, 8]
labels_comm = ['1 GPU', '2 GPUs', '4 GPUs', '8 GPUs']

# Computation time (percentage of total iteration time)
computation_pct = np.array([95.0, 88.5, 84.2, 82.4])

# Communication breakdown (percentage of total iteration time)
# Gradient synchronization increases with GPU count
gradient_sync_pct = np.array([0.0, 6.7, 9.2, 10.2])
# Barrier wait time
barrier_wait_pct = np.array([0.0, 3.1, 4.2, 4.7])
# Parameter broadcast
param_broadcast_pct = np.array([5.0, 1.7, 2.4, 2.7])

# Verify percentages sum to 100
totals = computation_pct + gradient_sync_pct + barrier_wait_pct + param_broadcast_pct
assert np.allclose(totals, 100.0), "Percentages must sum to 100%"

# Create stacked bar chart
x_pos = np.arange(len(gpu_counts_comm))
bar_width = 0.6

# Plot stacked bars
bar1 = ax2.bar(x_pos, computation_pct, bar_width, 
              label='Computation', color='#2ca02c', edgecolor='black', linewidth=0.6)
bar2 = ax2.bar(x_pos, gradient_sync_pct, bar_width, 
              bottom=computation_pct,
              label='Gradient Sync', color='#ff7f0e', edgecolor='black', linewidth=0.6)
bar3 = ax2.bar(x_pos, barrier_wait_pct, bar_width,
              bottom=computation_pct + gradient_sync_pct,
              label='Barrier Wait', color='#d62728', edgecolor='black', linewidth=0.6)
bar4 = ax2.bar(x_pos, param_broadcast_pct, bar_width,
              bottom=computation_pct + gradient_sync_pct + barrier_wait_pct,
              label='Parameter Broadcast', color='#9467bd', edgecolor='black', linewidth=0.6)

# Add percentage labels on bars for key components
for i, (comp, grad, barrier) in enumerate(zip(computation_pct, gradient_sync_pct, barrier_wait_pct)):
    # Label computation percentage
    ax2.text(i, comp/2, f'{comp:.1f}%', ha='center', va='center', 
            fontsize=7, fontweight='normal', color='white')
    
    # Label gradient sync if significant
    if grad > 5:
        ax2.text(i, comp + grad/2, f'{grad:.1f}%', ha='center', va='center',
                fontsize=7, fontweight='normal', color='white')
    
    # Label barrier wait if significant
    if barrier > 3:
        ax2.text(i, comp + grad + barrier/2, f'{barrier:.1f}%', ha='center', va='center',
                fontsize=6.5, fontweight='normal', color='white')

# Configure right panel
ax2.set_xlabel('GPU Configuration', fontsize=10, fontweight='normal')
ax2.set_ylabel('Time Distribution (%)', fontsize=10, fontweight='normal')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels_comm, fontsize=8)
ax2.set_ylim(0, 100)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.grid(True, axis='y', alpha=0.3, linewidth=0.4, linestyle='--', zorder=0)
ax2.legend(loc='upper right', fontsize=7, frameon=True, fancybox=False,
          edgecolor='black', framealpha=0.95, ncol=1)
ax2.set_title('(b) Communication Overhead Breakdown', fontsize=10, fontweight='normal', pad=8)

# Add annotation showing overhead increase
ax2.annotate('', xy=(3, 17.6), xytext=(0, 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2, alpha=0.6))
ax2.text(1.5, 12, 'Comm. overhead:\n5.0% → 17.6%', fontsize=7, ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor='red', linewidth=0.7, alpha=0.85))

# Adjust layout to prevent overlap
plt.tight_layout(pad=0.8, w_pad=2.0)

# Save figure in high resolution for publication
plt.savefig('distributed_scalability.pdf', dpi=300, bbox_inches='tight',
           pad_inches=0.02, format='pdf')
plt.savefig('distributed_scalability.png', dpi=300, bbox_inches='tight',
           pad_inches=0.02, format='png')

print("Figure saved as 'distributed_scalability.pdf' and 'distributed_scalability.png'")
print("Figure dimensions: 7 x 2.8 inches")
print("Resolution: 300 DPI")
print("\nLeft panel: Strong scaling efficiency with 87.4% mean efficiency")
print("Right panel: Communication overhead breakdown across GPU counts")

# Display the figure
plt.show()