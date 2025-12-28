# heatmap_recorder.py
"""
Advanced Heatmap Recording System for Pong RL
Tracks ball position, paddle defense, scoring patterns, and learning progression

Usage:
    recorder = AdvancedHeatmapRecorder()
    
    for episode in range(num_episodes):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Record game state
            game_info = env.get_game_info()
            recorder.record_frame(
                ball_x=game_info['ball_position'][0],
                ball_y=game_info['ball_position'][1],
                action=action,
                player_paddle_y=game_info['player_paddle_y'],
                cpu_paddle_y=game_info['cpu_paddle_y'],
                scored=info.get('last_scorer')
            )
            
            if done:
                break
        
        # End epoch every N episodes
        if (episode + 1) % episodes_per_epoch == 0:
            recorder.end_epoch()
    
    # Export for visualization
    data = recorder.get_all_epochs_data()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class HeatmapRecorder:
    """
    Advanced heatmap recorder with temporal, paddle, and scoring analysis.
    Tracks learning progression across training epochs.
    """
    
    def __init__(self, 
                 court_width: int = 160, 
                 court_height: int = 192, 
                 grid_resolution: int = 20,
                 paddle_height: int = 16):
        """
        Initialize heatmap recorder.
        
        Args:
            court_width: Width of game court (pixels)
            court_height: Height of game court (pixels)
            grid_resolution: Resolution of heatmap grid (X dimension)
            paddle_height: Height of paddle (for position tracking)
        """
        self.court_width = court_width
        self.court_height = court_height
        self.grid_res = grid_resolution
        self.grid_height = int(court_height / (court_width / grid_resolution))
        self.paddle_height = paddle_height
        
        # Current epoch data
        self.current_epoch_data = self._create_empty_epoch()
        
        # Storage for all epochs
        self.all_epochs = []
        
        # Episode counter
        self.episode_count = 0
        self.frame_count = 0
    
    def _create_empty_epoch(self) -> Dict:
        """Create empty data structure for an epoch"""
        return {
            'ball_position': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'player_scores': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'cpu_scores': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'player_paddle': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'cpu_paddle': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'action_stay': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'action_up': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'action_down': np.zeros((self.grid_height, self.grid_res), dtype=np.float32),
            'stats': {
                'player_scores': 0,
                'cpu_scores': 0,
                'total_frames': 0,
                'total_episodes': 0,
                'avg_player_paddle_y': 0.0,
                'avg_cpu_paddle_y': 0.0,
                'avg_rally_length': 0.0
            }
        }
    
    def _position_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert court coordinates to grid coordinates.
        
        Args:
            x: X position on court (0 to court_width)
            y: Y position on court (0 to court_height)
            
        Returns:
            Tuple of (grid_x, grid_y) clamped to grid bounds
        """
        grid_x = int((x / self.court_width) * self.grid_res)
        grid_y = int((y / self.court_height) * self.grid_height)
        
        grid_x = np.clip(grid_x, 0, self.grid_res - 1)
        grid_y = np.clip(grid_y, 0, self.grid_height - 1)
        
        return grid_x, grid_y
    
    def record_frame(self,
                    ball_x: float,
                    ball_y: float,
                    action: int,
                    player_paddle_y: float,
                    cpu_paddle_y: float,
                    scored: Optional[str] = None,
                    rally_length: int = 0) -> None:
        """
        Record a single frame of gameplay.
        
        Args:
            ball_x: Ball X position on court
            ball_y: Ball Y position on court
            action: Player action (0=stay, 1=up, 2=down)
            player_paddle_y: Player paddle Y position
            cpu_paddle_y: CPU paddle Y position
            scored: 'player', 'cpu', or None
            rally_length: Number of volleys in current rally
        """
        # Ball position heatmap
        ball_gx, ball_gy = self._position_to_grid(ball_x, ball_y)
        self.current_epoch_data['ball_position'][ball_gy, ball_gx] += 1
        
        # Action heatmaps
        action_keys = ['action_stay', 'action_up', 'action_down']
        if 0 <= action < 3:
            self.current_epoch_data[action_keys[action]][ball_gy, ball_gx] += 1
        
        # Paddle positions (track defense locations)
        # Player paddle is on right edge (x ≈ 150-160)
        player_gx, player_gy = self._position_to_grid(self.court_width - 5, player_paddle_y)
        self.current_epoch_data['player_paddle'][player_gy, player_gx] += 1
        
        # CPU paddle is on left edge (x ≈ 0-10)
        cpu_gx, cpu_gy = self._position_to_grid(5, cpu_paddle_y)
        self.current_epoch_data['cpu_paddle'][cpu_gy, cpu_gx] += 1
        
        # Scoring heatmaps - record WHERE on court point was scored
        if scored == 'player':
            self.current_epoch_data['player_scores'][ball_gy, ball_gx] += 1
            self.current_epoch_data['stats']['player_scores'] += 1
        elif scored == 'cpu':
            self.current_epoch_data['cpu_scores'][ball_gy, ball_gx] += 1
            self.current_epoch_data['stats']['cpu_scores'] += 1
        
        # Update running statistics
        self.current_epoch_data['stats']['total_frames'] += 1
        self.current_epoch_data['stats']['avg_player_paddle_y'] += player_paddle_y
        self.current_epoch_data['stats']['avg_cpu_paddle_y'] += cpu_paddle_y
        self.current_epoch_data['stats']['avg_rally_length'] += rally_length
        
        self.frame_count += 1
    
    def end_episode(self) -> None:
        """Call at end of each episode"""
        self.episode_count += 1
        self.current_epoch_data['stats']['total_episodes'] += 1
    
    def end_epoch(self) -> None:
        """
        Save current epoch and reset for next.
        Call this after N episodes to mark epoch boundary.
        """
        # Finalize statistics
        stats = self.current_epoch_data['stats']
        if stats['total_frames'] > 0:
            stats['avg_player_paddle_y'] /= stats['total_frames']
            stats['avg_cpu_paddle_y'] /= stats['total_frames']
            stats['avg_rally_length'] /= stats['total_frames']
        
        # Save epoch
        self.all_epochs.append(self.current_epoch_data)
        
        # Reset for next epoch
        self.current_epoch_data = self._create_empty_epoch()
    
    def get_normalized_heatmap(self, 
                              epoch_idx: int,
                              heatmap_type: str) -> Optional[np.ndarray]:
        """
        Get normalized heatmap for visualization.
        
        Args:
            epoch_idx: Index of epoch (0 to num_epochs-1)
            heatmap_type: Type of heatmap
                ('ball_position', 'player_scores', 'cpu_scores',
                 'player_paddle', 'cpu_paddle', 'action_stay', 'action_up', 'action_down')
        
        Returns:
            Normalized heatmap array or None if invalid
        """
        if epoch_idx < 0 or epoch_idx >= len(self.all_epochs):
            return None
        
        epoch = self.all_epochs[epoch_idx]
        
        if heatmap_type not in epoch:
            return None
        
        heatmap = epoch[heatmap_type]
        max_val = np.max(heatmap)
        
        if max_val > 0:
            return heatmap / max_val
        return heatmap
    
    def get_epoch_stats(self, epoch_idx: int) -> Optional[Dict]:
        """
        Get statistics for a specific epoch.
        
        Args:
            epoch_idx: Index of epoch
            
        Returns:
            Dictionary with epoch statistics
        """
        if epoch_idx < 0 or epoch_idx >= len(self.all_epochs):
            return None
        
        return self.all_epochs[epoch_idx]['stats'].copy()
    
    def get_all_epochs_data(self) -> Dict:
        """
        Get all epochs data for export/visualization.
        
        Returns:
            Dictionary containing all epoch data
        """
        return {
            'num_epochs': len(self.all_epochs),
            'epochs': self.all_epochs,
            'metadata': {
                'court_width': self.court_width,
                'court_height': self.court_height,
                'grid_resolution': self.grid_res,
                'grid_height': self.grid_height,
                'total_frames': self.frame_count,
                'total_episodes': self.episode_count
            }
        }
    
    def get_temporal_analysis(self) -> Dict[str, List]:
        """
        Get temporal analysis across all epochs.
        
        Returns:
            Dictionary with temporal metrics
        """
        analysis = {
            'epochs': [],
            'player_win_rate': [],
            'player_scores': [],
            'cpu_scores': [],
            'avg_rally_length': [],
            'player_paddle_center': [],
            'cpu_paddle_center': []
        }
        
        for idx, epoch in enumerate(self.all_epochs):
            stats = epoch['stats']
            
            total_points = stats['player_scores'] + stats['cpu_scores']
            win_rate = (stats['player_scores'] / total_points * 100) if total_points > 0 else 0
            
            analysis['epochs'].append(idx)
            analysis['player_win_rate'].append(win_rate)
            analysis['player_scores'].append(stats['player_scores'])
            analysis['cpu_scores'].append(stats['cpu_scores'])
            analysis['avg_rally_length'].append(stats['avg_rally_length'])
            analysis['player_paddle_center'].append(stats['avg_player_paddle_y'])
            analysis['cpu_paddle_center'].append(stats['avg_cpu_paddle_y'])
        
        return analysis
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export heatmap data to JSON for external analysis.
        
        Args:
            filepath: Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        data = self.get_all_epochs_data()
        serializable = convert_to_serializable(data)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"✓ Exported heatmap data to {filepath}")
        
    """
    ADD THIS METHOD TO heatmap_recorder.py CLASS
    
    This adds PNG visualization export to the HeatmapRecorder class.
    Converts heatmap numpy arrays to beautiful PNG images.
    """
    
    # Add these imports at the top of heatmap_recorder.py:
    # import matplotlib.pyplot as plt
    # import matplotlib.cm as cm
    # from matplotlib.colors import Normalize
    
    # Then add this method to the HeatmapRecorder class:
    
    
    def export_to_png(self, output_dir: str) -> None:
        """
        Export heatmaps as PNG images for visualization and papers.
        Creates one PNG per heatmap type per epoch.
        
        Args:
            output_dir: Directory to save PNG files
            
        Example output:
            outputs/pong_baseline_20251107_143522/heatmap_images/
            ├── epoch_0_ball_position.png
            ├── epoch_0_player_scores.png
            ├── epoch_0_cpu_scores.png
            ├── epoch_1_ball_position.png
            ├── epoch_1_player_scores.png
            └── ...
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        
        # Create output directory
        output_path = Path(output_dir) / "heatmap_images"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Exporting heatmaps as PNG images to {output_path}/")
        
        # Heatmap types to export
        heatmap_types = [
            'ball_position',
            'player_scores',
            'cpu_scores',
            'player_paddle',
            'cpu_paddle',
            'action_stay',
            'action_up',
            'action_down'
        ]
        
        # Titles for visualization
        titles = {
            'ball_position': 'Ball Position Heatmap',
            'player_scores': 'Player Scoring Locations',
            'cpu_scores': 'CPU Scoring Locations (Player Weaknesses)',
            'player_paddle': 'Player Paddle Defense Position',
            'cpu_paddle': 'CPU Paddle Position',
            'action_stay': 'Action: STAY',
            'action_up': 'Action: UP',
            'action_down': 'Action: DOWN'
        }
        
        # Export for each epoch
        for epoch_idx in range(len(self.all_epochs)):
            for heatmap_type in heatmap_types:
                # Get normalized heatmap
                heatmap = self.get_normalized_heatmap(epoch_idx, heatmap_type)
                
                if heatmap is None:
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
                
                # Plot heatmap
                im = ax.imshow(heatmap, cmap='hot', aspect='auto', origin='upper')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, label='Frequency (normalized)')
                
                # Labels and title
                ax.set_xlabel('X Position (Court Width)', fontsize=12)
                ax.set_ylabel('Y Position (Court Height)', fontsize=12)
                ax.set_title(f"Epoch {epoch_idx} - {titles[heatmap_type]}", 
                            fontsize=14, fontweight='bold')
                
                # Add grid for reference
                ax.grid(True, alpha=0.2, color='white', linestyle='--')
                
                # Save figure
                filename = f"epoch_{epoch_idx}_{heatmap_type}.png"
                filepath = output_path / filename
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Saved: {filename}")
        
        # Create summary visualization (all epochs for ball position)
        self._export_temporal_heatmap(output_path, titles)
        
        print(f"\n✓ Heatmap PNG export complete!")
        print(f"  Total images: {len(self.all_epochs)} epochs × {len(heatmap_types)} types")
        print(f"  Location: {output_path}/")


    def _export_temporal_heatmap(self, output_path: Path, titles: dict) -> None:
        """
        Create temporal progression visualization showing how gameplay evolves.
        Shows ball position heatmap for each epoch side-by-side.
        
        Args:
            output_path: Directory to save
            titles: Dictionary of heatmap titles
        """
        import matplotlib.pyplot as plt
        
        num_epochs = len(self.all_epochs)
        
        if num_epochs == 0:
            return
        
        # Create figure with subplots (one row per heatmap type, one col per epoch)
        heatmap_types = [
            'ball_position',
            'player_scores',
            'cpu_scores',
            'player_paddle'
        ]
        
        fig, axes = plt.subplots(len(heatmap_types), num_epochs, 
                                 figsize=(4*num_epochs, 4*len(heatmap_types)), 
                                 dpi=100)
        
        # Handle single epoch case (axes won't be 2D)
        if num_epochs == 1:
            axes = axes.reshape(-1, 1)
        if len(heatmap_types) == 1:
            axes = axes.reshape(1, -1)
        
        # Fill subplots
        for hm_idx, heatmap_type in enumerate(heatmap_types):
            for epoch_idx in range(num_epochs):
                heatmap = self.get_normalized_heatmap(epoch_idx, heatmap_type)
                
                if heatmap is None:
                    continue
                
                ax = axes[hm_idx, epoch_idx]
                
                im = ax.imshow(heatmap, cmap='hot', aspect='auto', origin='upper')
                
                # Title only for first row
                if hm_idx == 0:
                    ax.set_title(f'Epoch {epoch_idx}', fontsize=10, fontweight='bold')
                
                # Y-label only for first column
                if epoch_idx == 0:
                    ax.set_ylabel(f'{titles[heatmap_type]}', fontsize=10)
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle('Temporal Evolution of Gameplay Heatmaps Across Training Epochs', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = output_path / "temporal_progression.png"
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: temporal_progression.png (4×{num_epochs} grid)")
    
    def get_scoring_weakness_map(self, epoch_idx: int) -> np.ndarray:
        """
        Get a heatmap showing where player is weakest (CPU scores most).
        This is the inverse of player_scores - shows defensive gaps.
        
        Args:
            epoch_idx: Index of epoch
            
        Returns:
            Normalized heatmap of player weaknesses
        """
        cpu_score_map = self.get_normalized_heatmap(epoch_idx, 'cpu_scores')
        if cpu_score_map is None:
            return None
        
        return cpu_score_map
    
    def get_scoring_strength_map(self, epoch_idx: int) -> np.ndarray:
        """
        Get a heatmap showing where player scores most often.
        Shows CPU's defensive weaknesses.
        
        Args:
            epoch_idx: Index of epoch
            
        Returns:
            Normalized heatmap of player scoring strength
        """
        player_score_map = self.get_normalized_heatmap(epoch_idx, 'player_scores')
        if player_score_map is None:
            return None
        
        return player_score_map
    
    def reset(self) -> None:
        """Reset all recorded data"""
        self.current_epoch_data = self._create_empty_epoch()
        self.all_epochs = []
        self.episode_count = 0
        self.frame_count = 0