# training_analytics.py
"""
Training Analytics and Visualization Module
Comprehensive logging, analysis, and publication-quality plots for RL training.

All outputs (plots, logs, CSVs) saved to 'recorded_data/' folder with consistent timestamp naming.

Handles:
- Reward component tracking
- Win rate computation
- Policy entropy calculation
- CSV logging
- Matplotlib visualization (10 plots)
- Summary statistics

Usage:
    from training_analytics import TrainingAnalytics
    analytics = TrainingAnalytics()
    # After training:
    analytics.save_summary(config_dict={...})
    analytics.generate_plots()
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import datetime


class TrainingAnalytics:
    """Comprehensive training analytics and visualization"""
    
    # ACCEPT THE CORRECT PATH AND TIMESTAMP FROM THE ORCHESTRATOR
    def __init__(self, output_dir, timestamp):
        """Initialize analytics tracker with externally provided path and timestamp"""
        
        # USE THE INJECTED PATH AND TIMESTAMP
        self.output_dir = output_dir 
        self.timestamp = timestamp
        
        # All files use same timestamp in the correct output_dir/
        self.training_csv = self.output_dir / f"{self.timestamp}.csv"
        self.evaluation_csv = self.output_dir / f"{self.timestamp}_eval.csv"
        self.summary_txt = self.output_dir / f"{self.timestamp}_summary.txt"
        self.plots_prefix = self.output_dir / f"{self.timestamp}"
        
        # Data storage
        self.episodes = []
        self.total_rewards = []
        self.score_rewards = []
        self.hit_rewards = []
        self.loss_penalties = []
        self.episode_lengths = []
        self.player_scores = []
        self.cpu_scores = []
        self.win_flags = []
        self.running_averages = []
        self.policy_entropies = []
        self.losses = []
        self.actions_taken = defaultdict(int)
        
        # Initialize CSV headers
        self._init_csv_files()
        
        print(f"✓ Analytics initialized")
        print(f"  Timestamp: {self.timestamp}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  All files saved as: {self.timestamp}*")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Training log
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'score_reward', 'hit_reward', 'loss_penalty',
                'episode_length', 'player_score', 'cpu_score', 'win_flag',
                'running_average', 'policy_entropy', 'loss'
            ])
        
        # Evaluation log
        with open(self.evaluation_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'eval_episode', 'total_reward', 'player_score', 'cpu_score',
                'episode_length', 'win_flag'
            ])
    
    def log_episode(self, episode, total_reward, reward_breakdown, episode_length,
                    player_score, cpu_score, policy_entropy=None, loss=None):
        """
        Log a training episode.
        
        Args:
            episode: Episode number
            total_reward: Total reward for episode
            reward_breakdown: Dict with keys: 'score', 'hit', 'loss'
            episode_length: Number of steps in episode
            player_score: Final player score
            cpu_score: Final CPU score
            policy_entropy: (Optional) Policy entropy measure
            loss: (Optional) Loss value
        """
        self.episodes.append(episode)
        self.total_rewards.append(total_reward)
        self.score_rewards.append(reward_breakdown.get('score', 0))
        self.hit_rewards.append(reward_breakdown.get('hit', 0))
        self.loss_penalties.append(reward_breakdown.get('loss', 0))
        self.episode_lengths.append(episode_length)
        self.player_scores.append(player_score)
        self.cpu_scores.append(cpu_score)
        
        # Determine win
        win = 1 if player_score > cpu_score else 0
        self.win_flags.append(win)
        
        # Optional metrics
        self.policy_entropies.append(policy_entropy if policy_entropy is not None else 0)
        self.losses.append(loss if loss is not None else 0)
        
        # Running average
        if len(self.running_averages) == 0:
            running_avg = total_reward
        else:
            running_avg = 0.95 * self.running_averages[-1] + 0.05 * total_reward
        self.running_averages.append(running_avg)
        
        # Write to CSV
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward,
                reward_breakdown.get('score', 0),
                reward_breakdown.get('hit', 0),
                reward_breakdown.get('loss', 0),
                episode_length, player_score, cpu_score, win,
                running_avg, policy_entropy if policy_entropy is not None else '',
                loss if loss is not None else ''
            ])
    
    def log_evaluation_episode(self, eval_episode, total_reward, player_score, cpu_score,
                               episode_length):
        """Log an evaluation episode"""
        win = 1 if player_score > cpu_score else 0
        
        with open(self.evaluation_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                eval_episode, total_reward, player_score, cpu_score,
                episode_length, win
            ])
    
    def log_action(self, action):
        """Track action frequency"""
        self.actions_taken[action] += 1
    
    def save_summary(self, config_dict=None):
        """
        Save summary statistics to text file.
        
        Args:
            config_dict: Optional dict of hyperparameters to include
        """
        if len(self.total_rewards) == 0:
            print("⚠ No training data to summarize")
            return
        
        summary_lines = [
            "=" * 60,
            "TRAINING SUMMARY",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            "",
            "TRAINING RESULTS",
            "-" * 60,
            f"Total episodes trained: {len(self.total_rewards)}",
            f"Final running average reward: {self.running_averages[-1]:.2f}",
            f"Best episode reward: {max(self.total_rewards):.2f}",
            f"Worst episode reward: {min(self.total_rewards):.2f}",
            f"Mean episode reward: {np.mean(self.total_rewards):.2f}",
            f"Std episode reward: {np.std(self.total_rewards):.2f}",
            "",
            "EVALUATION METRICS",
            "-" * 60,
            f"Win rate: {(sum(self.win_flags) / len(self.win_flags) * 100):.1f}%",
            f"Total wins: {sum(self.win_flags)} / {len(self.win_flags)}",
            f"Mean episode length: {np.mean(self.episode_lengths):.0f} steps",
            f"Mean player score: {np.mean(self.player_scores):.1f}",
            f"Mean CPU score: {np.mean(self.cpu_scores):.1f}",
            "",
            "REWARD COMPONENTS (Average)",
            "-" * 60,
            f"Score reward: +{np.mean([r for r in self.score_rewards if r > 0]):.2f}" if any(r > 0 for r in self.score_rewards) else "Score reward: N/A",
            f"Hit reward: +{np.mean([r for r in self.hit_rewards if r > 0]):.4f}" if any(r > 0 for r in self.hit_rewards) else "Hit reward: N/A",
            f"Loss penalty: {np.mean([r for r in self.loss_penalties if r < 0]):.2f}" if any(r < 0 for r in self.loss_penalties) else "Loss penalty: N/A",
            "",
            "ACTION DISTRIBUTION",
            "-" * 60,
        ]
        
        if self.actions_taken:
            total_actions = sum(self.actions_taken.values())
            summary_lines.extend([
                f"Stay: {self.actions_taken.get(0, 0)} ({self.actions_taken.get(0, 0) / total_actions * 100:.1f}%)",
                f"Up: {self.actions_taken.get(1, 0)} ({self.actions_taken.get(1, 0) / total_actions * 100:.1f}%)",
                f"Down: {self.actions_taken.get(2, 0)} ({self.actions_taken.get(2, 0) / total_actions * 100:.1f}%)",
            ])
        else:
            summary_lines.append("N/A")
        
        if config_dict:
            summary_lines.extend([
                "",
                "HYPERPARAMETERS",
                "-" * 60,
            ])
            for key, value in config_dict.items():
                summary_lines.append(f"{key}: {value}")
        
        summary_lines.extend([
            "",
            "=" * 60,
        ])
        
        summary_text = "\n".join(summary_lines)
        
        with open(self.summary_txt, 'w') as f:
            f.write(summary_text)
        
        print(f"\n✓ Summary saved to {self.summary_txt}")
        print(summary_text)
    
    def generate_plots(self):
        """Generate all visualization plots"""
        if len(self.total_rewards) == 0:
            print("⚠ No data to plot")
            return
        
        print("\n✓ Generating plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Learning Curve (Raw + Moving Average)
        self._plot_learning_curve()
        
        # Plot 2: Running Average Reward
        self._plot_running_average()
        
        # Plot 3: Reward Component Breakdown
        self._plot_reward_components()
        
        # Plot 4: Episode Length Over Time
        self._plot_episode_length()
        
        # Plot 5: Win Rate Over Training
        self._plot_win_rate()
        
        # Plot 6: Policy Entropy
        self._plot_policy_entropy()
        
        # Plot 7: Action Distribution
        self._plot_action_distribution()
        
        # Plot 8: Loss Over Time
        self._plot_loss()
        
        # Plot 9: Histogram of Rewards
        self._plot_reward_histogram()
        
        # Plot 10: Score Distribution
        self._plot_score_distribution()
        
        plt.close('all')
        print(f"✓ All plots saved to {self.output_dir}/ with prefix {self.timestamp}")
    
    def _plot_learning_curve(self):
        """Plot 1: Learning curve with raw and smoothed"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.episodes, self.total_rewards, alpha=0.3, label='Raw Episode Reward', linewidth=0.5)
        
        window = min(50, len(self.total_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.total_rewards, np.ones(window)/window, mode='valid')
            ax.plot(self.episodes[window-1:], moving_avg, label=f'Moving Avg (window={window})',
                   linewidth=2, color='orange')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Learning Curve - Episode Rewards', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_01_learning_curve.png', dpi=150)
        plt.close()
    
    def _plot_running_average(self):
        """Plot 2: Running average reward"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.episodes, self.running_averages, linewidth=2, color='green', label='Running Average')
        ax.fill_between(self.episodes, self.running_averages, alpha=0.2, color='green')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Running Average Reward', fontsize=12)
        ax.set_title('Running Average Reward (Exponential Smoothing)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_02_running_average.png', dpi=150)
        plt.close()
    
    def _plot_reward_components(self):
        """Plot 3: Reward component breakdown"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        window = min(100, len(self.episodes) // 10)
        if window > 1:
            score_smooth = np.convolve(self.score_rewards, np.ones(window)/window, mode='valid')
            hit_smooth = np.convolve(self.hit_rewards, np.ones(window)/window, mode='valid')
            loss_smooth = np.convolve(self.loss_penalties, np.ones(window)/window, mode='valid')
            
            ep_offset = self.episodes[window-1:]
            ax.plot(ep_offset, score_smooth, label='Score Reward', linewidth=2)
            ax.plot(ep_offset, hit_smooth, label='Hit Reward', linewidth=2)
            ax.plot(ep_offset, loss_smooth, label='Loss Penalty', linewidth=2)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward Component Value', fontsize=12)
        ax.set_title('Reward Component Breakdown (Smoothed)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_03_reward_components.png', dpi=150)
        plt.close()
    
    def _plot_episode_length(self):
        """Plot 4: Episode length over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.episodes, self.episode_lengths, alpha=0.5, linewidth=1, color='purple')
        
        window = 50
        if len(self.episode_lengths) >= window:
            smooth = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax.plot(self.episodes[window-1:], smooth, linewidth=2, color='darkviolet', label=f'Avg (window={window})')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Length (steps)', fontsize=12)
        ax.set_title('Episode Duration Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_04_episode_length.png', dpi=150)
        plt.close()
    
    def _plot_win_rate(self):
        """Plot 5: Win rate over training"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        window = 100
        win_rates = []
        win_episodes = []
        
        for i in range(len(self.win_flags) - window + 1):
            win_rate = np.mean(self.win_flags[i:i+window]) * 100
            win_rates.append(win_rate)
            win_episodes.append(self.episodes[i + window - 1])
        
        ax.plot(win_episodes, win_rates, linewidth=2, color='red', marker='o', markersize=3)
        ax.fill_between(win_episodes, win_rates, alpha=0.2, color='red')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title(f'Win Rate Over Training (window={window})', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_05_win_rate.png', dpi=150)
        plt.close()
    
    def _plot_policy_entropy(self):
        """Plot 6: Policy entropy over time"""
        if not any(self.policy_entropies):
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.episodes, self.policy_entropies, alpha=0.5, linewidth=1, color='cyan')
        
        window = 50
        if len(self.policy_entropies) >= window:
            smooth = np.convolve(self.policy_entropies, np.ones(window)/window, mode='valid')
            ax.plot(self.episodes[window-1:], smooth, linewidth=2, color='darkcyan', label=f'Avg (window={window})')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Policy Entropy', fontsize=12)
        ax.set_title('Policy Entropy Over Training (Exploration → Exploitation)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_06_policy_entropy.png', dpi=150)
        plt.close()
    
    def _plot_action_distribution(self):
        
        """Plot 7: Action distribution"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        actions = ['Stay', 'Up', 'Down']
        counts = [self.actions_taken.get(i, 0) for i in range(3)]
        colors = ['gray', 'blue', 'red']
        
        bars = ax.bar(actions, counts, color=colors, alpha=0.7, edgecolor='black')
        if sum(counts) == 0:
            print("Warning: Skipping action distribution plot. No action data was logged during training/evaluation.")
            return # Safely exits the function without dividing by zero
        else:
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}\n({count/sum(counts)*100:.1f}%)',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Action Distribution Over All Episodes', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{self.plots_prefix}_07_action_distribution.png', dpi=150)
            plt.close()
        
    def _plot_loss(self):
        """Plot 8: Loss over time"""
        if not any(self.losses):
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.episodes, self.losses, alpha=0.5, linewidth=1, color='orange')
        
        window = 50
        if len(self.losses) >= window:
            smooth = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            ax.plot(self.episodes[window-1:], smooth, linewidth=2, color='darkorange', label=f'Avg (window={window})')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Policy Gradient Loss Over Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_08_loss.png', dpi=150)
        plt.close()
    
    def _plot_reward_histogram(self):
        """Plot 9: Histogram of rewards"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.total_rewards, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(self.total_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.total_rewards):.2f}')
        ax.axvline(np.median(self.total_rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(self.total_rewards):.2f}')
        
        ax.set_xlabel('Total Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Episode Rewards', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_09_reward_histogram.png', dpi=150)
        plt.close()
    
    def _plot_score_distribution(self):
        """Plot 10: Player vs CPU score distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.scatter(self.player_scores, self.cpu_scores, alpha=0.5, s=30, c=self.win_flags, cmap='RdYlGn')
        ax1.plot([0, 21], [0, 21], 'k--', alpha=0.3, label='Equal score line')
        ax1.set_xlabel('Player Score', fontsize=12)
        ax1.set_ylabel('CPU Score', fontsize=12)
        ax1.set_title('Player vs CPU Scores (Green=Win, Red=Loss)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        box_data = [self.player_scores, self.cpu_scores]
        bp = ax2.boxplot(box_data, labels=['Player', 'CPU'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Final Score', fontsize=12)
        ax2.set_title('Final Score Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_prefix}_10_score_distribution.png', dpi=150)
        plt.close()
    
    def save_all(self):
        """Save all data and generate plots"""
        print("\n✓ Saving all analytics...")
        self.generate_plots()
        print(f"✓ Analytics complete!")
        print(f"  All files in: {self.output_dir}/")
        print(f"  Timestamp prefix: {self.timestamp}")
