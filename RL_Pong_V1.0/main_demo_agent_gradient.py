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
# FIGURE: LEARNING CURVE
# =============================================================================
print("\n[1/6] Generating Learning Curve...")

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
ax.set_title('Mean Training Reward over 20,000 Episodes (n=5 runs)', 
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
plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: learning_curve.png")

# =============================================================================
# FIGURE: EDGE-HIT PROGRESSION (LINE GRAPH - ALL EPOCHS)
# =============================================================================
print("\n[2/6] Generating Edge-Hit Progression Line Graph...")

# Load collision summary data from all experiments
all_edge_percentages = []

for exp in EXPERIMENTS:
    collision_csv = BASE_PATH / exp / "collision_analysis" / "collision_summary.csv"
    df = pd.read_csv(collision_csv)
    
    # Get edge percentage for each epoch
    edge_pct = df['Edge_Percent'].values
    all_edge_percentages.append(edge_pct)
    print(f"  ✓ Loaded {exp} edge percentages")

# Average across experiments
all_edge_percentages = np.array(all_edge_percentages)
mean_edge = np.mean(all_edge_percentages, axis=0)
std_edge = np.std(all_edge_percentages, axis=0)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
epochs = np.arange(len(mean_edge))

# Plot mean line
ax.plot(epochs, mean_edge, color='#F18F01', linewidth=2, label='Mean Edge-Hit %')

# Plot shaded std area
ax.fill_between(epochs, 
                mean_edge - std_edge, 
                mean_edge + std_edge,
                color='#F18F01', alpha=0.2, label='±1 Std Dev')

# Styling
ax.set_xlabel('Epoch (200 episodes each)', fontsize=12, fontweight='bold')
ax.set_ylabel('Edge-Hit Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Edge-Hit Percentage Progression During Training (n=5 runs)', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 99)
ax.set_ylim(50, 100)

# Add horizontal reference line at 98.9%
ax.axhline(y=98.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(50, 99, f'Final: 98.9%', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "edge_hit_progression.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: edge_hit_progression.png")

# =============================================================================
# FIGURE: COLLISION DISTRIBUTION (BAR CHART - EARLY VS LATE)
# =============================================================================
print("\n[3/6] Generating Collision Distribution Bar Chart...")

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
ax.set_title('Paddle Contact Distribution - Early vs Late Training (n=5 runs)', 
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
plt.savefig(OUTPUT_DIR / "collision_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: collision_distribution.png")

# =============================================================================
# FIGURE: EPISODE LENGTH OVER TIME
# =============================================================================
print("\n[4/6] Generating Episode Length Progression...")

# Load all training data for episode length
all_lengths = []
for exp in EXPERIMENTS:
    csv_files = list((BASE_PATH / exp).glob("*.csv"))
    training_csv = [f for f in csv_files if "collision" not in f.name][0]
    
    df = pd.read_csv(training_csv)
    all_lengths.append(df['episode_length'].values)

# Convert to numpy array
all_lengths = np.array(all_lengths)

# Calculate mean and std
mean_lengths = np.mean(all_lengths, axis=0)
std_lengths = np.std(all_lengths, axis=0)

# Smooth with rolling average
window = 500
mean_smooth = pd.Series(mean_lengths).rolling(window=window, center=True).mean()
std_smooth = pd.Series(std_lengths).rolling(window=window, center=True).mean()

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
episodes = np.arange(len(mean_lengths))

# Plot mean line
ax.plot(episodes, mean_smooth, color='#6A4C93', linewidth=2, label='Mean Episode Length')

# Plot shaded std area
ax.fill_between(episodes, 
                mean_smooth - std_smooth, 
                mean_smooth + std_smooth,
                color='#6A4C93', alpha=0.2, label='±1 Std Dev')

# Styling
ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Episode Length (steps)', fontsize=12, fontweight='bold')
ax.set_title('Episode Length Progression During Training (n=5 runs)', 
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 20000)

# Add reference lines
ax.axhline(y=4092, color='blue', linestyle='--', alpha=0.3, linewidth=1)
ax.axhline(y=4440, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.text(1000, 4092-100, 'Early: 4,092', fontsize=9, color='blue')
ax.text(1000, 4440+100, 'Late: 4,440', fontsize=9, color='red')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "episode_length_progression.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: episode_length_progression.png")

# =============================================================================
# TABLE: PERFORMANCE SUMMARY (SIMPLE VERSION - NO EMPTY CELLS)
# =============================================================================
print("\n[5/6] Generating Performance Summary Table (Simple)...")

# Extract data from aggregated statistics
train_stats = agg_stats['aggregated']['training']

# Create table data
table_data = {
    'Metric': [
        'Win Rate (%)',
        'Mean Score (per match)',
    ],
    'Random Baseline': [
        '0.0',
        '~0',
    ],
    'Trained Agent': [
        f"{train_stats['win_rate_percent_mean']:.1f} (σ={train_stats['win_rate_percent_std']:.2f})",
        f"{train_stats['mean_player_score_mean']:.1f} (σ={train_stats['mean_player_score_std']:.2f})",
    ],
    'CPU Opponent': [
        f"{100 - train_stats['win_rate_percent_mean']:.1f}",
        f"{train_stats['mean_cpu_score_mean']:.1f} (σ={train_stats['mean_cpu_score_std']:.2f})",
    ]
}

# Create figure for table
fig, ax = plt.subplots(figsize=(10, 3))
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
table.scale(1, 2.5)

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
plt.title('Performance Comparison Across Agents', 
          fontsize=13, fontweight='bold', pad=20)

plt.savefig(OUTPUT_DIR / "performance_table_simple.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: performance_table_simple.png")

# =============================================================================
# TABLE: PERFORMANCE SUMMARY (DETAILED VERSION - AGENT ONLY)
# =============================================================================
print("\n[6/6] Generating Performance Summary Table (Detailed)...")

# Create detailed table with agent metrics only
detailed_data = {
    'Metric': [
        'Win Rate (%)',
        'Mean Score',
        'CPU Score',
        'Edge Hits - Early Training (%)',
        'Edge Hits - Late Training (%)',
        'Mean Episode Length (steps)',
    ],
    'Value': [
        f"{train_stats['win_rate_percent_mean']:.1f} (σ={train_stats['win_rate_percent_std']:.2f})",
        f"{train_stats['mean_player_score_mean']:.1f} (σ={train_stats['mean_player_score_std']:.2f})",
        f"{train_stats['mean_cpu_score_mean']:.1f} (σ={train_stats['mean_cpu_score_std']:.2f})",
        f"{agg_stats['aggregated']['collision']['early_edge_percent_mean']:.1f} (σ={agg_stats['aggregated']['collision']['early_edge_percent_std']:.2f})",
        f"{agg_stats['aggregated']['collision']['late_edge_percent_mean']:.1f} (σ={agg_stats['aggregated']['collision']['late_edge_percent_std']:.2f})",
        f"{train_stats['mean_episode_length_mean']:.0f} (σ={train_stats['mean_episode_length_std']:.0f})",
    ]
}

# Create figure for table
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=[[detailed_data['Metric'][i], 
                            detailed_data['Value'][i]] 
                           for i in range(len(detailed_data['Metric']))],
                colLabels=['Metric', 'Trained Agent (n=5)'],
                cellLoc='left',
                loc='center',
                colWidths=[0.6, 0.4])

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header styling
for i in range(2):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(detailed_data['Metric']) + 1):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

# Add title
plt.title('Trained Agent Performance Summary', 
          fontsize=13, fontweight='bold', pad=20)

plt.savefig(OUTPUT_DIR / "performance_table_detailed.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: performance_table_detailed.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("✓ All figures generated successfully!")
print(f"\nOutput location: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. learning_curve.png - Training reward progression")
print("  2. edge_hit_progression.png - Edge-hit % over all epochs (LINE)")
print("  3. collision_distribution.png - Early vs Late by position (BAR)")
print("  4. episode_length_progression.png - Episode length over time")
print("  5. performance_table_simple.png - Win rate & score comparison")
print("  6. performance_table_detailed.png - Complete agent metrics")
print("\nRecommendation for paper:")
print("  - Essential: #1, #2 (or #3), #5")
print("  - Optional: #4, #6")
print("="*70)