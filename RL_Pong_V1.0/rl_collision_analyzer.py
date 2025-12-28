# rl_collision_analyzer.py
"""
Paddle Collision Analysis Module for Pong RL
Tracks collision positions, bounce angles, and strategic evolution

Demonstrates agent's exploitation of progressive angle mechanics by:
- Recording where ball hits paddle (edge vs center)
- Tracking resulting bounce angles
- Analyzing progression over training epochs

Usage:
    analyzer = CollisionAnalyzer(paddle_height=12, episodes_per_epoch=200)
    
    # During training, when ball hits player paddle:
    analyzer.record_collision(
        hit_position=hit_pos,      # 0-11 (pixel on paddle)
        bounce_angle=angle,         # degrees (-90 to 90)
        episode=current_episode
    )
    
    # At end of training:
    analyzer.generate_report(output_dir="outputs/exp_001/analysis")
    analyzer.plot_all_figures(output_dir="outputs/exp_001/analysis")
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple


class CollisionAnalyzer:
    """
    Analyzes paddle collision patterns to demonstrate strategic learning.
    
    Key Metrics:
    - Collision position distribution (edge vs center hits)
    - Bounce angle distribution and progression
    - Temporal evolution across training epochs
    """
    
    def __init__(self, paddle_height: int = 12, episodes_per_epoch: int = 200):
        """
        Initialize collision analyzer.
        
        Args:
            paddle_height: Height of paddle in pixels (default: 12)
            episodes_per_epoch: Number of episodes per analysis epoch (default: 200)
        """
        self.paddle_height = paddle_height
        self.episodes_per_epoch = episodes_per_epoch
        
        # Define edge zones (top 25% and bottom 25% of paddle)
        self.edge_threshold = int(paddle_height * 0.25)  # ~3 pixels for 12-pixel paddle
        self.top_edge_zone = (0, self.edge_threshold)
        self.bottom_edge_zone = (paddle_height - self.edge_threshold, paddle_height)
        
        # Data storage
        self.collision_positions = []  # Where ball hit paddle (0 to paddle_height-1)
        self.bounce_angles = []        # Resulting bounce angle in degrees
        self.episode_numbers = []      # Which episode this occurred in
        
        # Epoch-aggregated statistics
        self.epoch_stats = []
        
        # Current epoch tracking
        self.current_epoch_collisions = []
        self.current_epoch_angles = []
        self.current_epoch_start = 0
    
    def record_collision(self, hit_position: float, bounce_angle: float, episode: int):
        """
        Record a single paddle collision event.
        
        Args:
            hit_position: Y-position on paddle where ball hit (0 = top, paddle_height-1 = bottom)
            bounce_angle: Resulting bounce angle in degrees (positive = upward, negative = downward)
            episode: Current episode number
        """
        # Clamp hit position to valid range
        hit_position = np.clip(hit_position, 0, self.paddle_height - 1)
        
        # Store raw data
        self.collision_positions.append(hit_position)
        self.bounce_angles.append(bounce_angle)
        self.episode_numbers.append(episode)
        
        # Store in current epoch
        self.current_epoch_collisions.append(hit_position)
        self.current_epoch_angles.append(bounce_angle)
    
    def end_epoch(self, epoch_number: int):
        """
        Finalize current epoch and compute statistics.
        
        Args:
            epoch_number: The epoch number being finalized
        """
        if len(self.current_epoch_collisions) == 0:
            print(f"⚠ Warning: Epoch {epoch_number} has no collision data")
            return
        
        # Calculate epoch statistics
        stats = self._compute_epoch_stats(
            self.current_epoch_collisions,
            self.current_epoch_angles,
            epoch_number
        )
        
        self.epoch_stats.append(stats)
        
        # Reset for next epoch
        self.current_epoch_collisions = []
        self.current_epoch_angles = []
        self.current_epoch_start = len(self.collision_positions)
    
    def _compute_epoch_stats(self, positions: List[float], angles: List[float], epoch: int) -> Dict:
        """Compute statistics for a single epoch"""
        positions_array = np.array(positions)
        angles_array = np.array(angles)
        
        # Edge hit detection
        edge_hits = np.sum(
            (positions_array <= self.edge_threshold) |  # Top edge
            (positions_array >= self.paddle_height - self.edge_threshold)  # Bottom edge
        )
        total_hits = len(positions_array)
        edge_percentage = (edge_hits / total_hits * 100) if total_hits > 0 else 0
        
        # Angle statistics
        avg_angle = np.mean(np.abs(angles_array))  # Average absolute angle
        max_angle = np.max(np.abs(angles_array))
        
        # Position distribution
        top_hits = np.sum(positions_array <= self.edge_threshold)
        center_hits = np.sum(
            (positions_array > self.edge_threshold) &
            (positions_array < self.paddle_height - self.edge_threshold)
        )
        bottom_hits = np.sum(positions_array >= self.paddle_height - self.edge_threshold)
        
        return {
            'epoch': epoch,
            'total_collisions': total_hits,
            'edge_hits': edge_hits,
            'edge_percentage': edge_percentage,
            'avg_angle': avg_angle,
            'max_angle': max_angle,
            'top_hits': top_hits,
            'center_hits': center_hits,
            'bottom_hits': bottom_hits,
            'top_percentage': (top_hits / total_hits * 100) if total_hits > 0 else 0,
            'center_percentage': (center_hits / total_hits * 100) if total_hits > 0 else 0,
            'bottom_percentage': (bottom_hits / total_hits * 100) if total_hits > 0 else 0,
        }
    
    def get_summary_statistics(self) -> Dict:
        """
        Get overall summary statistics across all epochs.
        
        Returns:
            Dictionary with summary metrics
        """
        if len(self.epoch_stats) == 0:
            return {'error': 'No epoch data available'}
        
        # First epoch vs last epoch comparison
        first_epoch = self.epoch_stats[0]
        last_epoch = self.epoch_stats[-1]
        
        edge_improvement = last_epoch['edge_percentage'] - first_epoch['edge_percentage']
        angle_improvement = last_epoch['avg_angle'] - first_epoch['avg_angle']
        
        return {
            'total_epochs': len(self.epoch_stats),
            'total_collisions': sum(ep['total_collisions'] for ep in self.epoch_stats),
            'initial_edge_percentage': first_epoch['edge_percentage'],
            'final_edge_percentage': last_epoch['edge_percentage'],
            'edge_improvement': edge_improvement,
            'initial_avg_angle': first_epoch['avg_angle'],
            'final_avg_angle': last_epoch['avg_angle'],
            'angle_improvement': angle_improvement,
        }
    
    def generate_report(self, output_dir: str = "collision_analysis"):
        """
        Generate text report and save statistics to JSON.
        
        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_summary_statistics()
        
        # Create text report
        report_lines = [
            "=" * 70,
            "PADDLE COLLISION ANALYSIS REPORT",
            "=" * 70,
            "",
            "SUMMARY STATISTICS",
            "-" * 70,
            f"Total Epochs Analyzed: {summary['total_epochs']}",
            f"Total Collisions Recorded: {summary['total_collisions']}",
            "",
            "EDGE HIT PROGRESSION",
            "-" * 70,
            f"Initial Edge Hit Rate: {summary['initial_edge_percentage']:.1f}%",
            f"Final Edge Hit Rate: {summary['final_edge_percentage']:.1f}%",
            f"Improvement: +{summary['edge_improvement']:.1f} percentage points",
            "",
            "BOUNCE ANGLE PROGRESSION",
            "-" * 70,
            f"Initial Average Angle: {summary['initial_avg_angle']:.1f}°",
            f"Final Average Angle: {summary['final_avg_angle']:.1f}°",
            f"Improvement: +{summary['angle_improvement']:.1f}°",
            "",
            "INTERPRETATION",
            "-" * 70,
            "Edge hits increased from {:.1f}% to {:.1f}%, demonstrating the agent".format(
                summary['initial_edge_percentage'], summary['final_edge_percentage']
            ),
            "learned to exploit the progressive angle system. Higher edge hit rates",
            "produce steeper bounce angles, making returns harder to defend.",
            "",
            f"Average bounce angle increased from {summary['initial_avg_angle']:.1f}° to",
            f"{summary['final_avg_angle']:.1f}°, confirming strategic use of steep",
            "trajectories for offensive advantage.",
            "",
            "=" * 70,
        ]
        
        report_text = "\n".join(report_lines)
        
        # Save text report
        report_file = output_path / "collision_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to {report_file}")
        
        # Save detailed statistics to JSON
        json_data = {
            'summary': summary,
            'epoch_statistics': self.epoch_stats,
            'metadata': {
                'paddle_height': self.paddle_height,
                'edge_threshold': self.edge_threshold,
                'episodes_per_epoch': self.episodes_per_epoch,
            }
        }
        
        # Convert NumPy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            """
            Recursively convert NumPy types to native Python types for JSON serialization.
            """
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_data = convert_numpy_types(json_data)
        
        json_file = output_path / "collision_statistics.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

    
    def plot_all_figures(self, output_dir: str = "collision_analysis"):
        """
        Generate all visualization figures.
        
        Creates:
        - Figure 1: Collision position distribution (early vs late)
        - Figure 2: Bounce angle progression over epochs
        - Figure 3: Edge hit percentage over training
        
        Args:
            output_dir: Directory to save figure files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Generating collision analysis figures...")
        
        # Figure 1: Collision Distribution
        self._plot_collision_distribution(output_path)
        
        # Figure 2: Angle Progression
        self._plot_angle_progression(output_path)
        
        # Figure 3: Edge Hit Progression
        self._plot_edge_hit_progression(output_path)
        
        print(f"✓ All figures saved to {output_path}/")
    
    def _plot_collision_distribution(self, output_path: Path):
        """Figure 1: Collision position distribution"""
        if len(self.epoch_stats) < 2:
            print("⚠ Not enough epochs for collision distribution plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Early epoch (first)
        first_epoch = self.epoch_stats[0]
        early_positions = np.array(self.collision_positions[:first_epoch['total_collisions']])
        
        ax1.hist(early_positions, bins=self.paddle_height, range=(0, self.paddle_height),
                 color='lightcoral', edgecolor='black', alpha=0.7)
        ax1.axvspan(0, self.edge_threshold, alpha=0.2, color='green', label='Top Edge Zone')
        ax1.axvspan(self.paddle_height - self.edge_threshold, self.paddle_height,
                    alpha=0.2, color='green', label='Bottom Edge Zone')
        ax1.set_xlabel('Hit Position on Paddle (pixels)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Early Training (Epoch 0)\nEdge Hits: {first_epoch["edge_percentage"]:.1f}%',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Late epoch (last)
        last_epoch = self.epoch_stats[-1]
        late_start = sum(ep['total_collisions'] for ep in self.epoch_stats[:-1])
        late_positions = np.array(self.collision_positions[late_start:])
        
        ax2.hist(late_positions, bins=self.paddle_height, range=(0, self.paddle_height),
                 color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvspan(0, self.edge_threshold, alpha=0.2, color='green', label='Top Edge Zone')
        ax2.axvspan(self.paddle_height - self.edge_threshold, self.paddle_height,
                    alpha=0.2, color='green', label='Bottom Edge Zone')
        ax2.set_xlabel('Hit Position on Paddle (pixels)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Late Training (Epoch {last_epoch["epoch"]})\nEdge Hits: {last_epoch["edge_percentage"]:.1f}%',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Collision Position Distribution: Early vs Late Training',
                     fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = output_path / "figure_1_collision_distribution.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Figure 1 saved: {filepath.name}")
    
    def _plot_angle_progression(self, output_path: Path):
        """Figure 2: Average bounce angle over training"""
        if len(self.epoch_stats) == 0:
            print("⚠ No epoch data for angle progression plot")
            return
        
        epochs = [ep['epoch'] for ep in self.epoch_stats]
        avg_angles = [ep['avg_angle'] for ep in self.epoch_stats]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(epochs, avg_angles, marker='o', linewidth=2, markersize=6,
                color='darkgreen', label='Average Absolute Bounce Angle')
        ax.fill_between(epochs, avg_angles, alpha=0.2, color='green')
        
        # Add trend line
        z = np.polyfit(epochs, avg_angles, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "--", color='red', linewidth=1.5,
                alpha=0.7, label=f'Trend: +{z[0]:.2f}°/epoch')
        
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Average Absolute Bounce Angle (degrees)', fontsize=12)
        ax.set_title('Bounce Angle Progression Over Training\n(Higher angles = Steeper trajectories = Harder to defend)',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Annotate start and end
        ax.annotate(f'{avg_angles[0]:.1f}°', xy=(epochs[0], avg_angles[0]),
                    xytext=(epochs[0], avg_angles[0] - 5),
                    fontsize=10, ha='center', fontweight='bold', color='darkgreen')
        ax.annotate(f'{avg_angles[-1]:.1f}°', xy=(epochs[-1], avg_angles[-1]),
                    xytext=(epochs[-1], avg_angles[-1] + 5),
                    fontsize=10, ha='center', fontweight='bold', color='darkgreen')
        
        plt.tight_layout()
        
        filepath = output_path / "figure_2_angle_progression.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Figure 2 saved: {filepath.name}")
    
    def _plot_edge_hit_progression(self, output_path: Path):
        """Figure 3: Edge hit percentage over training"""
        if len(self.epoch_stats) == 0:
            print("⚠ No epoch data for edge hit progression plot")
            return
        
        epochs = [ep['epoch'] for ep in self.epoch_stats]
        edge_percentages = [ep['edge_percentage'] for ep in self.epoch_stats]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(epochs, edge_percentages, marker='s', linewidth=2.5, markersize=7,
                color='darkblue', label='Edge Hit Percentage')
        ax.fill_between(epochs, edge_percentages, alpha=0.2, color='blue')
        
        # Reference line at 33% (random would be ~33% if 4 pixels out of 12 are "edge")
        expected_random = (self.edge_threshold * 2) / self.paddle_height * 100
        ax.axhline(y=expected_random, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Random Baseline (~{expected_random:.0f}%)')
        
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Edge Hit Percentage (%)', fontsize=12)
        ax.set_title('Edge Hit Rate Over Training\n(Edge hits produce steeper angles for offensive advantage)',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(edge_percentages) + 10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Annotate start and end
        ax.annotate(f'{edge_percentages[0]:.1f}%', xy=(epochs[0], edge_percentages[0]),
                    xytext=(epochs[0] + 1, edge_percentages[0] + 3),
                    fontsize=10, fontweight='bold', color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=1))
        ax.annotate(f'{edge_percentages[-1]:.1f}%', xy=(epochs[-1], edge_percentages[-1]),
                    xytext=(epochs[-1] - 1, edge_percentages[-1] + 3),
                    fontsize=10, fontweight='bold', color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=1))
        
        plt.tight_layout()
        
        filepath = output_path / "figure_3_edge_hit_progression.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Figure 3 saved: {filepath.name}")


    def export_collision_csvs(self, output_dir: str = "collision_analysis"):
        """
        Export collision data as CSV files for analysis.
        
        Creates two CSV files:
        1. collision_summary.csv - Aggregated by epoch with zones (for paper)
        2. collision_distribution.csv - Full 12-position distribution (for detailed analysis)
        
        Args:
            output_dir: Directory to save CSV files
        """
        import csv
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # === CSV 1: Summary with Zones (PRIMARY - for paper) ===
        summary_csv = output_path / "collision_summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'Episodes', 'Total_Hits', 
                'Top_Edge', 'Center', 'Bottom_Edge',
                'Edge_Hits', 'Edge_Percent', 'Avg_Angle'
            ])
            
            for ep_stats in self.epoch_stats:
                episode_range = f"{ep_stats['epoch']*200+1}-{(ep_stats['epoch']+1)*200}"
                writer.writerow([
                    ep_stats['epoch'],
                    episode_range,
                    ep_stats['total_collisions'],
                    ep_stats['top_hits'],
                    ep_stats['center_hits'],
                    ep_stats['bottom_hits'],
                    ep_stats['edge_hits'],
                    f"{ep_stats['edge_percentage']:.1f}",
                    f"{ep_stats['avg_angle']:.1f}"
                ])
            
            # Add totals row
            if self.epoch_stats:
                total_collisions = sum(ep['total_collisions'] for ep in self.epoch_stats)
                total_top = sum(ep['top_hits'] for ep in self.epoch_stats)
                total_center = sum(ep['center_hits'] for ep in self.epoch_stats)
                total_bottom = sum(ep['bottom_hits'] for ep in self.epoch_stats)
                total_edge = total_top + total_bottom
                total_edge_pct = (total_edge / total_collisions * 100) if total_collisions > 0 else 0
                avg_all_angles = sum(ep['avg_angle'] * ep['total_collisions'] 
                                    for ep in self.epoch_stats) / total_collisions
                
                writer.writerow([
                    'TOTAL',
                    f"1-{len(self.epoch_stats)*200}",
                    total_collisions,
                    total_top,
                    total_center,
                    total_bottom,
                    total_edge,
                    f"{total_edge_pct:.1f}",
                    f"{avg_all_angles:.1f}"
                ])
        
        print(f"  ✓ Summary CSV: {summary_csv.name}")
        
        # === CSV 2: Full Distribution (DETAILED - all 12 positions) ===
        distribution_csv = output_path / "collision_distribution.csv"
        
        # Calculate distribution for each epoch
        with open(distribution_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Epoch', 'Episodes'] + [f'Pos_{i}' for i in range(self.paddle_height)]
            writer.writerow(header)
            
            # Data for each epoch
            epoch_start_idx = 0
            for ep_stats in self.epoch_stats:
                epoch_num = ep_stats['epoch']
                epoch_size = ep_stats['total_collisions']
                episode_range = f"{epoch_num*200+1}-{(epoch_num+1)*200}"
                
                # Get collision positions for this epoch
                epoch_positions = self.collision_positions[epoch_start_idx:epoch_start_idx + epoch_size]
                
                # Count hits per position (0-11)
                position_counts = [0] * self.paddle_height
                for pos in epoch_positions:
                    pos_idx = int(np.clip(pos, 0, self.paddle_height - 1))
                    position_counts[pos_idx] += 1
                
                # Write row
                row = [epoch_num, episode_range] + position_counts
                writer.writerow(row)
                
                epoch_start_idx += epoch_size
            
            # Add totals row
            if self.epoch_stats:
                total_position_counts = [0] * self.paddle_height
                for pos in self.collision_positions:
                    pos_idx = int(np.clip(pos, 0, self.paddle_height - 1))
                    total_position_counts[pos_idx] += 1
                
                row = ['TOTAL', f"1-{len(self.epoch_stats)*200}"] + total_position_counts
                writer.writerow(row)
        
        print(f"  ✓ Distribution CSV: {distribution_csv.name}")
        print(f"  Total CSV files: 2")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Create analyzer and simulate some data
    analyzer = CollisionAnalyzer(paddle_height=12, episodes_per_epoch=200)
    
    # Simulate 3 epochs of training with improving strategy
    np.random.seed(42)
    
    for epoch in range(3):
        print(f"\nSimulating Epoch {epoch}...")
        
        # Simulate collisions (200 episodes × ~50 collisions/episode = 10000 collisions/epoch)
        num_collisions = 10000
        
        for i in range(num_collisions):
            # Early epochs: more random (uniform distribution)
            # Late epochs: more edge hits (bimodal distribution)
            if np.random.random() < (epoch / 3):  # Increasing edge bias
                # Edge hit
                if np.random.random() < 0.5:
                    hit_pos = np.random.uniform(0, 3)  # Top edge
                else:
                    hit_pos = np.random.uniform(9, 12)  # Bottom edge
                angle = np.random.uniform(55, 75)  # Steep angle
            else:
                # Random hit
                hit_pos = np.random.uniform(0, 12)
                angle = np.random.uniform(30, 60)
            
            # Record collision
            episode = epoch * 200 + (i // 50)  # Approximate episode number
            analyzer.record_collision(hit_pos, angle, episode)
        
        # End epoch
        analyzer.end_epoch(epoch)
    
    # Generate report and plots
    analyzer.generate_report("example_output")
    analyzer.plot_all_figures("example_output")
    
    print("\n✓ Example analysis complete! Check 'example_output/' directory.")