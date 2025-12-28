# main_generate_paper_figures.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# Configuration
BASE_PATH = Path("evaluation")
EXPERIMENTS = ["exp_03", "exp_04", "exp_05", "exp_06", "exp_07"]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_PATH / f"average_results_{TIMESTAMP}"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load aggregated statistics
with open(BASE_PATH / "aggregated_statistics.json", "r") as f:
    agg_stats = json.load(f)

print(f"Creating figures in: {OUTPUT_DIR}")
print("="*70)

# =============================================================================
# FIGURE 1: LEARNING CURVE
# =============================================================================
print("\n[1/3] Generating Learning Curve...")

# Load all training data
all_rewards = []
for exp in EXPERIMENTS:
    # Find the CSV file (timestamp varies)
    csv_files = list((BASE_PATH / exp).glob("*.csv"))
    training_csv = [f for f in csv_files if "collision" not in f.name][0]
    
    df = pd.read_csv(training_csv)
    all_rewards.append(df['total_reward'].values)
    print(f"  ✓ Loaded {exp}: {len(df)} episodes")

# Convert to numpy array (experiments x episodes)
all_rewards = np.array(all_rewards)

# Calculate mean and std across experiments
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

# Smooth with rolling average for better visualization
window = 100
mean_smooth = pd.Series(mean_rewards).rolling(window=window, center=True).mean()
std_smooth = pd.Series(std_rewards).rolling(window=window, center=True).mean()

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
episodes = np.arange(len(mean_rewards))

# Plot mean line
ax.plot(episodes, mean_smooth, color='#2E86AB', linewidth=2, label='Mean Reward')

# Plot shaded std area
ax.fill_between(episodes, 
                mean_smooth - std_smooth, 
                mean_smooth + std_smooth,
                color='#2E86AB', alpha=0.2, label='±1 Std Dev')

# Styling
ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
ax.set_title('Figure 1: Mean Training Reward over 20,000 Episodes (n=5 runs)', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 20000)

# Add phase markers
ax.axvline(5000, color='gray', linestyle=':', alpha=0.5)
ax.axvline(10000, color='gray', linestyle=':', alpha=0.5)
ax.text(2500, ax.get_ylim()[1]*0.95, 'Early', ha='center', fontsize=9, alpha=0.7)
ax.text(7500, ax.get_ylim()[1]*0.95, 'Mid', ha='center', fontsize=9, alpha=0.7)
ax.text(15000, ax.get_ylim()[1]*0.95, 'Late', ha='center', fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure1_learning_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: figure1_learning_curve.png")

# =============================================================================
# FIGURE 2: COLLISION DISTRIBUTION
# =============================================================================
print("\n[2/3] Generating Collision Distribution...")

# Load collision distribution data from all experiments
all_early_collisions = []
all_late_collisions = []

for exp in EXPERIMENTS:
    collision_csv = BASE_PATH / exp / "collision_analysis" / "collision_distribution.csv"
    df = pd.read_csv(collision_csv)
    
    # Early: first 5 epochs (rows 0-4), Late: last 10 epochs (rows 90-99)
    early = df.iloc[0:5, 2:14].sum(axis=0).values  # Pos_0 to Pos_11
    late = df.iloc[90:100, 2:14].sum(axis=0).values
    
    all_early_collisions.append(early)
    all_late_collisions.append(late)
    print(f"  ✓ Loaded {exp} collision data")

# Average across experiments
early_mean = np.mean(all_early_collisions, axis=0)
late_mean = np.mean(all_late_collisions, axis=0)

# Normalize to percentages
early_pct = (early_mean / early_mean.sum()) * 100
late_pct = (late_mean / late_mean.sum()) * 100

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))
positions = np.arange(12)
width = 0.35

# Create bars
bars1 = ax.bar(positions - width/2, early_pct, width, 
               label='Early Training (Epochs 0-4)', 
               color='#A23B72', alpha=0.8)
bars2 = ax.bar(positions + width/2, late_pct, width, 
               label='Late Training (Epochs 90-99)', 
               color='#F18F01', alpha=0.8)

# Styling
ax.set_xlabel('Paddle Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of Total Hits (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2: Paddle Contact Distribution - Early vs Late Training (n=5 runs)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(positions)
ax.set_xticklabels([f'{i}' for i in range(12)])
ax.legend(fontsize=10, loc='upper center')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add edge position annotations
ax.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='_nolegend_')
ax.axvspan(10.5, 11.5, alpha=0.1, color='red', label='_nolegend_')
ax.text(0, max(late_pct)*1.05, 'Edge', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
ax.text(11, max(late_pct)*1.05, 'Edge', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure2_collision_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: figure2_collision_distribution.png")

# =============================================================================
# TABLE 1: PERFORMANCE SUMMARY
# =============================================================================
print("\n[3/3] Generating Performance Summary Table...")

# Extract data from aggregated statistics
train_stats = agg_stats['aggregated']['training']

# Create table data
table_data = {
    'Metric': [
        'Win Rate (%)',
        'Mean Score (per match)',
        'Final 1000 Score',
        'Edge Hits (%)',
        'Mean Episode Length'
    ],
    'Random Baseline': [
        '0.0',
        '~0',
        '—',
        '—',
        '—'
    ],
    'Trained Agent': [
        f"{train_stats['win_rate_percent_mean']:.1f} (σ={train_stats['win_rate_percent_std']:.2f})",
        f"{train_stats['mean_player_score_mean']:.1f} (σ={train_stats['mean_player_score_std']:.2f})",
        f"{train_stats['final_1000_player_score_mean']:.1f} (σ={train_stats['final_1000_player_score_std']:.2f})",
        f"{agg_stats['aggregated']['collision']['late_edge_percent_mean']:.1f} (σ={agg_stats['aggregated']['collision']['late_edge_percent_std']:.2f})",
        f"{train_stats['mean_episode_length_mean']:.0f} (σ={train_stats['mean_episode_length_std']:.0f})"
    ],
    'CPU Opponent': [
        f"{100 - train_stats['win_rate_percent_mean']:.1f}",
        f"{train_stats['mean_cpu_score_mean']:.1f} (σ={train_stats['mean_cpu_score_std']:.2f})",
        f"{train_stats['final_1000_cpu_score_mean']:.1f} (σ={train_stats['final_1000_cpu_score_std']:.2f})",
        '—',
        '—'
    ]
}

# Create figure for table
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=[[table_data['Metric'][i], 
                            table_data['Random Baseline'][i],
                            table_data['Trained Agent'][i],
                            table_data['CPU Opponent'][i]] 
                           for i in range(len(table_data['Metric']))],
                colLabels=['Metric', 'Random Baseline', 'Trained Agent (n=5)', 'CPU Opponent'],
                cellLoc='center',
                loc='center',
                colWidths=[0.3, 0.2, 0.3, 0.2])

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header styling
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data['Metric']) + 1):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

# Add title
plt.title('Table 1: Performance Comparison Across Agents', 
          fontsize=13, fontweight='bold', pad=20)

plt.savefig(OUTPUT_DIR / "table1_performance_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: table1_performance_summary.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("✓ All figures generated successfully!")
print(f"\nOutput location: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. figure1_learning_curve.png")
print("  2. figure2_collision_distribution.png")
print("  3. table1_performance_summary.png")
print("\nYou can now insert these into your paper.")
print("="*70)