"""
Complete Statistics Analyzer for Paper Results
Multi-Experiment Support with Aggregation across runs

HOW TO USE:
1. Set EXPERIMENT_NAMES list to analyze multiple experiments
2. Run: python complete_statistics_analyzer.py
3. Get individual + aggregated statistics

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ============================================================================
# 🎯 CHANGE THIS TO ANALYZE MULTIPLE EXPERIMENTS
# ============================================================================

EXPERIMENT_NAMES = ["exp_03", "exp_04", "exp_05", "exp_06", "exp_07"]

# ============================================================================
# AUTO-CONFIGURED PATHS (DON'T CHANGE)
# ============================================================================

BASE_DIR = "evaluation"

# ============================================================================
# LOAD DATA FUNCTIONS
# ============================================================================

def find_training_csv(data_dir):
    """Automatically find the training CSV (timestamp file)"""
    csv_files = list(data_dir.glob("*.csv"))
    training_files = [f for f in csv_files if 'collision' not in f.name.lower()]
    return training_files[0] if training_files else None


def load_training_data(exp_name):
    """Load main training CSV file for a specific experiment"""
    data_dir = Path(BASE_DIR) / exp_name
    
    if not data_dir.exists():
        print(f"❌ {exp_name}: Directory not found")
        return None
    
    training_file = find_training_csv(data_dir)
    
    if training_file is None:
        print(f"❌ {exp_name}: No training CSV found")
        return None
    
    print(f"✓ {exp_name}: Loading {training_file.name}")
    df = pd.read_csv(training_file)
    
    return df


def load_collision_summary(exp_name):
    """Load collision summary CSV from collision_analysis subfolder"""
    collision_dir = Path(BASE_DIR) / exp_name / "collision_analysis"
    file_path = collision_dir / "collision_summary.csv"
    
    if not file_path.exists():
        print(f"⚠ {exp_name}: No collision summary")
        return None
    
    print(f"✓ {exp_name}: Loading collision_summary.csv")
    df = pd.read_csv(file_path)
    
    return df


# ============================================================================
# CALCULATE STATISTICS
# ============================================================================

def calculate_training_statistics(df):
    """Calculate all training statistics needed for paper"""
    
    stats = {}
    
    # BASIC TRAINING METRICS
    stats['total_episodes'] = len(df)
    stats['final_running_avg'] = df['running_average'].iloc[-1]
    stats['best_episode_reward'] = df['total_reward'].max()
    stats['worst_episode_reward'] = df['total_reward'].min()
    stats['mean_episode_reward'] = df['total_reward'].mean()
    stats['std_episode_reward'] = df['total_reward'].std()
    
    # LEARNING PHASES
    early = df[df['episode'] <= 5000]
    mid = df[(df['episode'] > 5000) & (df['episode'] <= 10000)]
    late = df[df['episode'] > 10000]
    final = df[df['episode'] > (df['episode'].max() - 1000)]
    
    stats['early_avg_reward'] = early['total_reward'].mean() if len(early) > 0 else 0
    stats['mid_avg_reward'] = mid['total_reward'].mean() if len(mid) > 0 else 0
    stats['late_avg_reward'] = late['total_reward'].mean() if len(late) > 0 else 0
    stats['final_1000_avg_reward'] = final['total_reward'].mean() if len(final) > 0 else 0
    
    # WIN RATE
    stats['total_wins'] = df['win_flag'].sum()
    stats['win_rate_percent'] = (df['win_flag'].sum() / len(df)) * 100
    stats['early_win_rate'] = (early['win_flag'].sum() / len(early) * 100) if len(early) > 0 else 0
    stats['late_win_rate'] = (late['win_flag'].sum() / len(late) * 100) if len(late) > 0 else 0
    stats['final_1000_win_rate'] = (final['win_flag'].sum() / len(final) * 100) if len(final) > 0 else 0
    
    # EPISODE LENGTH
    stats['mean_episode_length'] = df['episode_length'].mean()
    stats['std_episode_length'] = df['episode_length'].std()
    stats['early_episode_length'] = early['episode_length'].mean() if len(early) > 0 else 0
    stats['late_episode_length'] = late['episode_length'].mean() if len(late) > 0 else 0
    
    if stats['early_episode_length'] > 0:
        stats['length_increase_percent'] = (
            (stats['late_episode_length'] - stats['early_episode_length']) 
            / stats['early_episode_length'] * 100
        )
    else:
        stats['length_increase_percent'] = 0
    
    # SCORES
    stats['mean_player_score'] = df['player_score'].mean()
    stats['std_player_score'] = df['player_score'].std()
    stats['mean_cpu_score'] = df['cpu_score'].mean()
    stats['std_cpu_score'] = df['cpu_score'].std()
    stats['final_1000_player_score'] = final['player_score'].mean() if len(final) > 0 else 0
    stats['final_1000_cpu_score'] = final['cpu_score'].mean() if len(final) > 0 else 0
    
    # REWARD COMPONENTS
    stats['mean_score_reward'] = df['score_reward'].mean()
    stats['std_score_reward'] = df['score_reward'].std()
    stats['mean_hit_reward'] = df['hit_reward'].mean()
    stats['std_hit_reward'] = df['hit_reward'].std()
    stats['mean_loss_penalty'] = df['loss_penalty'].mean()
    stats['std_loss_penalty'] = df['loss_penalty'].std()
    
    return stats


def calculate_collision_statistics(df_collision):
    """Calculate collision analysis statistics"""
    
    if df_collision is None:
        return None
    
    collision_stats = {}
    
    df_clean = df_collision[df_collision['Epoch'] != 'TOTAL'].copy()
    
    if df_clean['Epoch'].dtype == 'object':
        df_clean['Epoch'] = pd.to_numeric(df_clean['Epoch'], errors='coerce')
    
    df_clean = df_clean.dropna(subset=['Epoch'])
    
    collision_stats['total_collisions'] = df_clean['Total_Hits'].sum()
    collision_stats['total_epochs'] = len(df_clean)
    
    early = df_clean[df_clean['Epoch'] <= 5]
    mid = df_clean[(df_clean['Epoch'] >= 10) & (df_clean['Epoch'] <= 20)]
    late = df_clean[df_clean['Epoch'] >= 90]
    final = df_clean[df_clean['Epoch'] >= 95]
    
    collision_stats['early_edge_percent'] = early['Edge_Percent'].mean() if len(early) > 0 else 0
    collision_stats['mid_edge_percent'] = mid['Edge_Percent'].mean() if len(mid) > 0 else 0
    collision_stats['late_edge_percent'] = late['Edge_Percent'].mean() if len(late) > 0 else 0
    collision_stats['final_edge_percent'] = final['Edge_Percent'].mean() if len(final) > 0 else 0
    
    collision_stats['edge_increase'] = (
        collision_stats['late_edge_percent'] - collision_stats['early_edge_percent']
    )
    
    return collision_stats


# ============================================================================
# AGGREGATE STATISTICS ACROSS EXPERIMENTS
# ============================================================================

def aggregate_statistics(all_stats):
    """Calculate mean and std across all experiments"""
    
    if not all_stats:
        return None
    
    agg = {}
    
    # Get all metric keys from first experiment
    first_exp = list(all_stats.values())[0]
    
    for key in first_exp['training'].keys():
        values = [exp['training'][key] for exp in all_stats.values() if exp['training'][key] is not None]
        
        if values:
            agg[f'{key}_mean'] = np.mean(values)
            agg[f'{key}_std'] = np.std(values)
    
    # Aggregate collision stats if present
    collision_values = {}
    for exp_name, exp_data in all_stats.items():
        if exp_data['collision'] is not None:
            for key, val in exp_data['collision'].items():
                if key not in collision_values:
                    collision_values[key] = []
                collision_values[key].append(val)
    
    agg_collision = {}
    for key, values in collision_values.items():
        if values:
            agg_collision[f'{key}_mean'] = np.mean(values)
            agg_collision[f'{key}_std'] = np.std(values)
    
    return {'training': agg, 'collision': agg_collision}


# ============================================================================
# FORMAT OUTPUT
# ============================================================================

def print_individual_statistics(exp_name, stats, collision_stats):
    """Print statistics for individual experiment"""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_name}")
    print("="*80)
    
    print(f"\nTotal episodes: {stats['total_episodes']:,}")
    print(f"Final running avg: {stats['final_running_avg']:.2f}")
    print(f"Mean reward: {stats['mean_episode_reward']:.2f} (σ={stats['std_episode_reward']:.2f})")
    print(f"Win rate: {stats['win_rate_percent']:.1f}%")
    print(f"Player score: {stats['mean_player_score']:.1f} | CPU: {stats['mean_cpu_score']:.1f}")
    
    if collision_stats:
        print(f"Edge-hit (late): {collision_stats['late_edge_percent']:.1f}%")


def print_aggregated_statistics(agg_stats, num_experiments):
    """Print aggregated statistics across all experiments"""
    
    print("\n" + "="*80)
    print(f"AGGREGATED STATISTICS (n={num_experiments} experiments)")
    print("="*80)
    
    tr = agg_stats['training']
    
    print("\n### TRAINING PROGRESS ###")
    print(f"Final running avg: μ={tr['final_running_avg_mean']:.2f} (σ={tr['final_running_avg_std']:.2f})")
    print(f"Mean reward: μ={tr['mean_episode_reward_mean']:.2f} (σ={tr['mean_episode_reward_std']:.2f})")
    
    print("\n### WIN RATE & SCORES ###")
    print(f"Win rate: μ={tr['win_rate_percent_mean']:.1f}% (σ={tr['win_rate_percent_std']:.2f})")
    print(f"Player score: μ={tr['mean_player_score_mean']:.1f} (σ={tr['mean_player_score_std']:.2f})")
    print(f"CPU score: μ={tr['mean_cpu_score_mean']:.1f} (σ={tr['mean_cpu_score_std']:.2f})")
    
    print("\n### EPISODE LENGTH ###")
    print(f"Mean length: μ={tr['mean_episode_length_mean']:.0f} (σ={tr['mean_episode_length_std']:.0f})")
    print(f"Early: μ={tr['early_episode_length_mean']:.0f} → Late: μ={tr['late_episode_length_mean']:.0f}")
    print(f"Increase: μ={tr['length_increase_percent_mean']:.1f}% (σ={tr['length_increase_percent_std']:.1f})")
    
    print("\n### REWARD COMPONENTS ###")
    print(f"Score: μ={tr['mean_score_reward_mean']:.2f} (σ={tr['mean_score_reward_std']:.2f})")
    print(f"Hit: μ={tr['mean_hit_reward_mean']:.4f} (σ={tr['mean_hit_reward_std']:.4f})")
    print(f"Loss: μ={tr['mean_loss_penalty_mean']:.2f} (σ={tr['mean_loss_penalty_std']:.2f})")
    
    if agg_stats['collision']:
        col = agg_stats['collision']
        print("\n### COLLISION ANALYSIS ###")
        print(f"Total collisions: μ={col['total_collisions_mean']:,.0f} (σ={col['total_collisions_std']:,.0f})")
        print(f"Early edge%: μ={col['early_edge_percent_mean']:.1f}% (σ={col['early_edge_percent_std']:.1f})")
        print(f"Late edge%: μ={col['late_edge_percent_mean']:.1f}% (σ={col['late_edge_percent_std']:.1f})")
        print(f"Increase: μ={col['edge_increase_mean']:.1f}pp (σ={col['edge_increase_std']:.1f})")
    
    print("\n" + "="*80)


# ============================================================================
# HELPER
# ============================================================================

def make_json_serializable(obj):
    """Recursively convert NumPy/pandas types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_all_statistics(all_stats, agg_stats):
    """Save both individual and aggregated statistics"""
    
    # Save aggregated JSON
    agg_file = Path(BASE_DIR) / "aggregated_statistics.json"
    agg_data = {
        'num_experiments': len(all_stats),
        'experiments': list(all_stats.keys()),
        'aggregated': agg_stats,
        'individual': all_stats
    }
    
    agg_data = make_json_serializable(agg_data)
    
    with open(agg_file, 'w') as f:
        json.dump(agg_data, f, indent=2)
    
    print(f"\n✓ Saved aggregated JSON: {agg_file}")
    
    # Save aggregated text
    txt_file = Path(BASE_DIR) / "aggregated_statistics.txt"
    
    with open(txt_file, 'w') as f:
        f.write(f"AGGREGATED STATISTICS (n={len(all_stats)} experiments)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Experiments: {', '.join(all_stats.keys())}\n\n")
        
        f.write("TRAINING AGGREGATED:\n")
        for key, val in agg_stats['training'].items():
            f.write(f"  {key}: {val}\n")
        
        if agg_stats['collision']:
            f.write("\nCOLLISION AGGREGATED:\n")
            for key, val in agg_stats['collision'].items():
                f.write(f"  {key}: {val}\n")
    
    print(f"✓ Saved aggregated TXT: {txt_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution - analyze multiple experiments"""
    
    print("\n" + "="*80)
    print("MULTI-EXPERIMENT STATISTICS ANALYZER")
    print("="*80)
    print(f"\nAnalyzing {len(EXPERIMENT_NAMES)} experiments:")
    for exp in EXPERIMENT_NAMES:
        print(f"  - {exp}")
    print()
    
    # Load and analyze each experiment
    all_stats = {}
    
    for exp_name in EXPERIMENT_NAMES:
        print(f"\n{'─'*80}")
        print(f"Processing: {exp_name}")
        print('─'*80)
        
        # Load data
        df_training = load_training_data(exp_name)
        
        if df_training is None:
            print(f"⚠ Skipping {exp_name} (no data)")
            continue
        
        df_collision = load_collision_summary(exp_name)
        
        # Calculate statistics
        stats = calculate_training_statistics(df_training)
        collision_stats = calculate_collision_statistics(df_collision)
        
        # Store results
        all_stats[exp_name] = {
            'training': stats,
            'collision': collision_stats
        }
        
        # Print individual summary
        print_individual_statistics(exp_name, stats, collision_stats)
    
    # Aggregate across all experiments
    if len(all_stats) > 1:
        print(f"\n{'='*80}")
        print("COMPUTING AGGREGATED STATISTICS")
        print('='*80)
        
        agg_stats = aggregate_statistics(all_stats)
        
        # Print aggregated results
        print_aggregated_statistics(agg_stats, len(all_stats))
        
        # Save everything
        save_all_statistics(all_stats, agg_stats)
    
    elif len(all_stats) == 1:
        print(f"\n⚠ Only 1 experiment loaded - no aggregation performed")
    
    else:
        print(f"\n❌ No experiments successfully loaded")
    
    print("\n✓ ANALYSIS COMPLETE\n")


if __name__ == "__main__":
    main()