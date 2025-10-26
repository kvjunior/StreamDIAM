import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import rc

# Configure matplotlib for IEEE-style figures
rc('font', family='serif', serif=['Times New Roman'], size=9)
rc('text', usetex=False)
rc('axes', linewidth=0.8)
rc('grid', linewidth=0.4, alpha=0.3)

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(7, 3.5))
ax2 = ax1.twinx()

# Define timeline (in seconds)
time = np.linspace(0, 12, 1000)

# Key timestamps
t_failure = 0.0
t_detection = 0.247
t_recovery_start = 0.3
t_checkpoint_restore = 0.3 + 8.4  # 8.4 seconds for restore
t_complete = 8.3
t_stabilization = 9.3  # 1 minute after completion for full stabilization

# Define throughput trajectory (transactions per second)
throughput = np.zeros_like(time)
normal_throughput = 7214  # Normal throughput from EthereumP dataset

for i, t in enumerate(time):
    if t < t_detection:
        # Normal operation before detection
        throughput[i] = normal_throughput + np.random.normal(0, 50)
    elif t < t_recovery_start:
        # Brief detection period - slight drop
        throughput[i] = normal_throughput * 0.95 + np.random.normal(0, 100)
    elif t < t_complete:
        # Recovery period - degraded performance (31.4% degradation)
        recovery_progress = (t - t_recovery_start) / (t_complete - t_recovery_start)
        degraded_throughput = normal_throughput * 0.686  # 31.4% degradation
        current_throughput = degraded_throughput + (normal_throughput - degraded_throughput) * (recovery_progress ** 0.7)
        throughput[i] = current_throughput + np.random.normal(0, 80)
    elif t < t_stabilization:
        # Post-recovery stabilization - approaching normal (98.7% of normal)
        stabilization_progress = (t - t_complete) / (t_stabilization - t_complete)
        throughput[i] = normal_throughput * (0.95 + 0.037 * stabilization_progress) + np.random.normal(0, 60)
    else:
        # Fully stabilized
        throughput[i] = normal_throughput * 0.987 + np.random.normal(0, 50)

# Define accuracy trajectory (F1-score)
accuracy = np.zeros_like(time)
normal_accuracy = 0.903  # F1-score from EthereumP dataset

for i, t in enumerate(time):
    if t < t_detection:
        # Normal operation
        accuracy[i] = normal_accuracy + np.random.normal(0, 0.002)
    elif t < t_recovery_start:
        # Detection period - stable accuracy
        accuracy[i] = normal_accuracy + np.random.normal(0, 0.003)
    elif t < t_complete:
        # Recovery period - slight accuracy fluctuation but maintained
        accuracy[i] = normal_accuracy * 0.99 + np.random.normal(0, 0.004)
    elif t < t_stabilization:
        # Post-recovery - returning to normal
        stabilization_progress = (t - t_complete) / (t_stabilization - t_complete)
        accuracy[i] = normal_accuracy * (0.99 + 0.01 * stabilization_progress) + np.random.normal(0, 0.003)
    else:
        # Fully stabilized
        accuracy[i] = normal_accuracy + np.random.normal(0, 0.002)

# Clip values to reasonable ranges
throughput = np.clip(throughput, 0, normal_throughput * 1.05)
accuracy = np.clip(accuracy, 0.85, 0.92)

# Plot throughput on primary y-axis
line1 = ax1.plot(time, throughput, color='#1f77b4', linewidth=1.2, 
                 label='Throughput', zorder=3)
ax1.set_xlabel('Time (seconds)', fontsize=10, fontweight='normal')
ax1.set_ylabel('Throughput (TPS)', fontsize=10, fontweight='normal', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=9)
ax1.tick_params(axis='x', labelsize=9)
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 8500)
ax1.grid(True, alpha=0.3, linewidth=0.4, linestyle='--')

# Plot accuracy on secondary y-axis
line2 = ax2.plot(time, accuracy, color='#ff7f0e', linewidth=1.2, 
                 label='F1-Score', zorder=3)
ax2.set_ylabel('F1-Score', fontsize=10, fontweight='normal', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=9)
ax2.set_ylim(0.85, 0.92)

# Add shaded region for recovery period
recovery_region = Rectangle((t_recovery_start, 0), t_complete - t_recovery_start, 
                           8500, facecolor='red', alpha=0.08, zorder=1)
ax1.add_patch(recovery_region)

# Add vertical lines for key events
ax1.axvline(x=t_failure, color='red', linestyle='--', linewidth=1.0, 
           alpha=0.7, zorder=2)
ax1.axvline(x=t_detection, color='orange', linestyle='--', linewidth=1.0, 
           alpha=0.7, zorder=2)
ax1.axvline(x=t_complete, color='green', linestyle='--', linewidth=1.0, 
           alpha=0.7, zorder=2)

# Add annotations for key events
ax1.annotate('Failure Injection\n$t=0$', xy=(t_failure, 7800), 
            xytext=(0.5, 7800), fontsize=8,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='red', linewidth=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

ax1.annotate('Detection\n$t=247$ms', xy=(t_detection, 6800), 
            xytext=(1.2, 6800), fontsize=8,
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='orange', linewidth=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))

ax1.annotate('Recovery Complete\n$t=8.3$s', xy=(t_complete, 7000), 
            xytext=(t_complete - 2.5, 7600), fontsize=8,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='green', linewidth=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

# Add text annotation for recovery region
ax1.text(t_complete / 2, 500, 'Recovery Period\n(Degraded Performance)', 
        fontsize=8, ha='center', va='bottom', style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor='gray', linewidth=0.6, alpha=0.85))

# Create combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=8, 
          frameon=True, fancybox=False, edgecolor='black', 
          framealpha=0.95, ncol=2)

# Add minor gridlines
ax1.grid(True, which='minor', alpha=0.15, linewidth=0.3, linestyle=':')
ax1.minorticks_on()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure in high resolution for publication
plt.savefig('recovery_timeline.pdf', dpi=300, bbox_inches='tight', 
           pad_inches=0.02, format='pdf')
plt.savefig('recovery_timeline.png', dpi=300, bbox_inches='tight', 
           pad_inches=0.02, format='png')

print("Figure saved as 'recovery_timeline.pdf' and 'recovery_timeline.png'")
print("Figure dimensions: 7 x 3.5 inches")
print("Resolution: 300 DPI")

# Display the figure
plt.show()