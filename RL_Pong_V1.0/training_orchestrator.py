# training_orchestrator.py
"""
Training Orchestrator (Agent & Trainer Agnostic) with Heatmap Recording and Collision Analysis
Manages training workflow, analytics logging, heatmap recording, collision analysis, and result saving independently.

Key Design:
- Does NOT import specific trainer or agent classes
- Works with ANY trainer that implements the Trainer interface
- Works with ANY agent that has .save() method
- NOW includes HeatmapRecorder for automatic gameplay data collection
- NOW includes CollisionAnalyzer for paddle hit analysis
- Purely orchestrates the pipeline
"""

import os
import json
import datetime
from pathlib import Path
from training_analytics import TrainingAnalytics
from config import TrainingConfig, PolicyGradientAgentConfig, EnvConfig
from rl_collision_analyzer import CollisionAnalyzer

# =============================================================================
# TOP-LEVEL DYNAMIC PATH DEFINITION (Single Source of Truth for Timestamp)
# =============================================================================

# Calculate the timestamp once when the script starts
GLOBAL_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_output_root():
    """Generates the unique, timestamped root directory using TrainingConfig.EXPERIMENT_NAME."""
    
    # Access EXPERIMENT_NAME directly from the imported config module
    experiment_name = TrainingConfig.EXPERIMENT_NAME 
    
    # Use the globally defined timestamp
    dir_name = f"{experiment_name}_{GLOBAL_TIMESTAMP}"
    return Path("outputs") / dir_name

# The Orchestrator's main output path is defined here
OUTPUT_ROOT_DIR = _get_output_root()

# Define Sub-Directories relative to the root
SAVED_MODELS_DIR = OUTPUT_ROOT_DIR / "saved_models"
RECORDED_DATA_DIR = OUTPUT_ROOT_DIR / "recorded_data"
HEATMAP_DIR = RECORDED_DATA_DIR / "heatmaps"
COLLISION_DIR = RECORDED_DATA_DIR / "collision_analysis"


# =============================================================================
# TRAINING ORCHESTRATOR CLASS
# =============================================================================

class TrainingOrchestrator:
    """Agent and trainer agnostic orchestrator with heatmap recording and collision analysis"""
    
    def __init__(self, agent, trainer, heatmap_recorder=None):
        """
        Initialize orchestrator. All paths are now derived from the top-level
        OUTPUT_ROOT_DIR defined above.
        """
        self.agent = agent
        self.trainer = trainer
        
        # Assign the experiment name from config
        self.experiment_name = TrainingConfig.EXPERIMENT_NAME
        
        # Connect heatmap recorder to trainer
        self.heatmap_recorder = heatmap_recorder
        self.trainer.heatmap_recorder = heatmap_recorder
        
        # NEW: Initialize collision analyzer
        self.collision_analyzer = CollisionAnalyzer(
            paddle_height=12,
            episodes_per_epoch=200
        )
        self.trainer.collision_analyzer = self.collision_analyzer
        
        # NEW: Set collision analyzer in environment module
        import env_custom_pong_simulator
        env_custom_pong_simulator.COLLISION_ANALYZER = self.collision_analyzer
        
        # CRITICAL FIX: Use the final, unique directory path
        self.output_dir = OUTPUT_ROOT_DIR 
        
        # CRITICAL FIX: Use the globally defined timestamp to ensure consistency
        self.timestamp = GLOBAL_TIMESTAMP 
        
        # Create directories based on top-level Path objects
        self.output_dir.mkdir(parents=True, exist_ok=True)
        SAVED_MODELS_DIR.mkdir(exist_ok=True)
        RECORDED_DATA_DIR.mkdir(exist_ok=True)
        HEATMAP_DIR.mkdir(exist_ok=True)
        COLLISION_DIR.mkdir(exist_ok=True)
        
        # Setup final paths based on the top-level definitions
        # NOTE: All these file names now use the single, consistent GLOBAL_TIMESTAMP
        self.model_path = SAVED_MODELS_DIR / f"{self.experiment_name}_{self.timestamp}.npz"
        self.config_path = SAVED_MODELS_DIR / f"{self.experiment_name}_{self.timestamp}_config.json"
        self.log_path = SAVED_MODELS_DIR / f"{self.experiment_name}_{self.timestamp}.log"
        self.heatmap_path = HEATMAP_DIR / f"{self.experiment_name}_{self.timestamp}_heatmap.json"
        
        # Initialize analytics - Pass the consistent timestamp
        self.analytics = TrainingAnalytics(
            output_dir=RECORDED_DATA_DIR, 
            timestamp=self.timestamp
        )
        
        # Training state
        self.start_episode_offset = 0
        self.model_loaded = False
        self.eval_results = None
        self.start_time = None
        self.end_time = None
        self._check_for_continuation()
    
    
    def _check_for_continuation(self):
        """Check if we're continuing from a previous run"""
        
        # Check for existing "current" model (from TrainingConfig.MODEL_SAVE_PATH)
        current_model_path = Path(TrainingConfig.MODEL_SAVE_PATH)
        
        if TrainingConfig.LOAD_EXISTING_MODEL and current_model_path.exists():
            print(f"\n✓ Found existing model at {current_model_path}")
            print(f"  Loading model to continue training...")
            self.agent.load(str(current_model_path))
            self.model_loaded = True
        
        # Check for existing CSV data
        existing_csv = self.analytics.training_csv
        
        if os.path.exists(existing_csv):
            print(f"\n✓ Found existing training data: {existing_csv}")
            print(f"  Continuing from previous run...")
            
            # Count existing episodes
            with open(existing_csv, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # More than just header
                    self.start_episode_offset = len(lines) - 1
                    print(f"  Found {self.start_episode_offset} existing episodes")
                    print(f"  New episodes will start at episode {self.start_episode_offset + 1}")
    
    
    def _log_training_to_analytics(self):
        """Log all training episodes to analytics"""
        print("\n✓ Logging training data to analytics...")
        training_data = self.trainer.get_training_data()
        
        for ep in range(len(training_data['episode_rewards'])):
            episode_num = self.start_episode_offset + ep + 1
            total_reward = training_data['episode_rewards'][ep]
            episode_length = training_data['episode_lengths'][ep]
            player_score = training_data['episode_player_scores'][ep]
            cpu_score = training_data['episode_cpu_scores'][ep]
            score_reward = training_data['episode_score_rewards'][ep]
            hit_reward = training_data['episode_hit_rewards'][ep]
            loss_penalty = training_data['episode_loss_penalties'][ep]
            
            self.analytics.log_episode(
                episode=episode_num,
                total_reward=total_reward,
                reward_breakdown={
                    'score': score_reward,
                    'hit': hit_reward,
                    'loss': loss_penalty
                },
                episode_length=episode_length,
                player_score=player_score,
                cpu_score=cpu_score,
                policy_entropy=None,
                loss=None
            )
    
    
    def _log_evaluation_to_analytics(self):
        """Log all evaluation episodes to analytics"""
        print("\n✓ Logging evaluation data to analytics...")
        for eval_data in self.eval_results['eval_results']:
            self.analytics.log_evaluation_episode(
                eval_episode=self.start_episode_offset + eval_data['episode'],
                total_reward=eval_data['reward'],
                player_score=eval_data['player_score'],
                cpu_score=eval_data['cpu_score'],
                episode_length=eval_data['length']
            )
    
    
    def _save_heatmap_data(self):
        """
        Save heatmap data to JSON and PNG files.
        Called automatically at end of training.
        """
        if self.heatmap_recorder is None:
            print("\n⚠ Heatmap recorder not provided - skipping heatmap save")
            return
        
        print(f"\n✓ Saving heatmap data to {self.heatmap_path}...")
        
        # Export JSON
        self.heatmap_recorder.export_to_json(str(self.heatmap_path))
        
        # Export PNG images
        self.heatmap_recorder.export_to_png(str(HEATMAP_DIR))
        
        # Print heatmap summary
        self._print_heatmap_summary()
    
    
    def _print_heatmap_summary(self):
        """Print summary of collected heatmap data"""
        if self.heatmap_recorder is None:
            return
        
        data = self.heatmap_recorder.get_all_epochs_data()
        temporal = self.heatmap_recorder.get_temporal_analysis()
        
        print("\n" + "=" * 60)
        print("HEATMAP DATA SUMMARY")
        print("=" * 60)
        print(f"Total Epochs: {data['num_epochs']}")
        print(f"Total Episodes Recorded: {data['metadata']['total_episodes']}")
        print(f"Total Frames Recorded: {data['metadata']['total_frames']:,}")
        print(f"Grid Resolution: {data['metadata']['grid_resolution']}×"
              f"{data['metadata']['grid_height']}")
        print(f"Court Size: {data['metadata']['court_width']}×"
              f"{data['metadata']['court_height']}")
        
        if temporal['epochs'] and len(temporal['epochs']) > 0:
            print(f"\nPerformance Progression (across epochs):")
            print(f"{'Epoch':<8} {'Episodes':<12} {'Player Wins':<15} {'CPU Wins':<15} {'Win %':<12}")
            print("-" * 60)
            
            for epoch_idx in range(data['num_epochs']):
                stats = self.heatmap_recorder.get_epoch_stats(epoch_idx)
                total_points = stats['player_scores'] + stats['cpu_scores']
                
                if total_points > 0:
                    win_rate = (stats['player_scores'] / total_points * 100)
                else:
                    win_rate = 0.0
                
                print(f"{epoch_idx:<8} {stats['total_episodes']:<12} "
                      f"{stats['player_scores']:<15} {stats['cpu_scores']:<15} "
                      f"{win_rate:>10.1f}%")
        
        print("=" * 60)
    
    
    def _save_collision_data(self):
        """
        Save collision analysis data and generate plots.
        Called automatically at end of training.
        """
        if self.collision_analyzer is None:
            print("\n⚠ Collision analyzer not provided - skipping collision analysis")
            return
        
        print(f"\n✓ Saving collision analysis to {COLLISION_DIR}...")
        
        # Generate report and plots
        self.collision_analyzer.generate_report(str(COLLISION_DIR))
        self.collision_analyzer.plot_all_figures(str(COLLISION_DIR))
        
        # Export CSV files
        self.collision_analyzer.export_collision_csvs(str(COLLISION_DIR))
        
        print(f"✓ Collision analysis complete!")
        print(f"  Location: {COLLISION_DIR}/")
    
    
    def _save_model(self):
        """Save trained model to both current location AND timestamped backup"""
        
        # Save to "current" location (for resuming training)
        current_model_path = Path(TrainingConfig.MODEL_SAVE_PATH)
        current_model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Saving current model to {current_model_path}...")
        self.agent.save(str(current_model_path))
        
        # Save timestamped backup (for version history)
        print(f"✓ Saving timestamped backup to {self.model_path}...")
        self.agent.save(str(self.model_path))
    
    
    def _save_config(self):
        """Save configuration JSON"""
        print(f"\n✓ Saving configuration to {self.config_path}...")
        
        try:
            summary = self.agent.get_network_summary()
        except:
            summary = {}
        
        config_dict = {
            'timestamp': self.timestamp,
            'experiment_name': self.experiment_name,
            'continued_from_episode': self.start_episode_offset,
            'environment': {
                'width': EnvConfig.WIDTH,
                'height': EnvConfig.HEIGHT,
                'max_score': EnvConfig.MAX_SCORE,
                'cpu_difficulty': EnvConfig.OPPONENT_DIFFICULTY,
            },
            'agent': summary if summary else {'note': 'Agent-specific configuration'},
            'training': {
                'max_episodes': TrainingConfig.MAX_EPISODES,
                'batch_size': TrainingConfig.BATCH_SIZE,
                'running_reward_decay': TrainingConfig.RUNNING_REWARD_DECAY,
                'print_frequency': TrainingConfig.PRINT_EVERY_N_EPISODES,
            },
            'rewards': {
                'ball_hit': TrainingConfig.REWARD_BALL_HIT,
                'score': TrainingConfig.REWARD_SCORE,
                'opponent_score': TrainingConfig.REWARD_OPPONENT_SCORE,
                'win': TrainingConfig.REWARD_WIN,
                'loss': TrainingConfig.REWARD_LOSS,
            },
            'training_results': {
                'final_running_average': float(self.trainer.running_reward),
                'best_episode_reward': float(self.trainer.best_reward),
                'total_episodes_trained': len(self.trainer.episode_rewards),
                'total_episodes_cumulative': self.start_episode_offset + len(self.trainer.episode_rewards),
            },
            'evaluation_results': {
                'mean_reward': float(self.eval_results['mean_reward']),
                'std_reward': float(self.eval_results['std_reward']),
                'win_rate': float(self.eval_results['win_rate']),
                'eval_episodes': TrainingConfig.EVAL_EPISODES,
            },
            'heatmap_data': {
                'recording_enabled': self.heatmap_recorder is not None,
                'heatmap_file': str(self.heatmap_path.name) if self.heatmap_recorder else None,
                'epochs_collected': self.heatmap_recorder.get_all_epochs_data()['num_epochs']
                                     if self.heatmap_recorder else 0,
            },
            'collision_data': {
                'recording_enabled': self.collision_analyzer is not None,
                'epochs_collected': len(self.collision_analyzer.epoch_stats)
                                   if self.collision_analyzer else 0,
                'total_collisions': sum(ep['total_collisions'] for ep in self.collision_analyzer.epoch_stats)
                                   if self.collision_analyzer and self.collision_analyzer.epoch_stats else 0,
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    
    def run(self):
        """Execute full training pipeline"""
        self.start_time = datetime.datetime.now()
        
        print("\n" + "=" * 70)
        print("TRAINING ORCHESTRATOR - UNIFIED PIPELINE")
        print("=" * 70)
        print(f"Experiment: {self.experiment_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Root Directory: {self.output_dir}/")
        print(f"Output Sub-Directories:")
        print(f"  - Models: {SAVED_MODELS_DIR}/")
        print(f"  - Analytics: {RECORDED_DATA_DIR}/")
        if self.heatmap_recorder:
            print(f"  - Heatmaps: {HEATMAP_DIR}/")
        if self.collision_analyzer:
            print(f"  - Collision Analysis: {COLLISION_DIR}/")
        print(f"Heatmap Recording: {'✓ ENABLED' if self.heatmap_recorder else '✗ DISABLED'}")
        print(f"Collision Analysis: {'✓ ENABLED' if self.collision_analyzer else '✗ DISABLED'}")
        
        # Run training
        print("\n✓ Starting trainer...")
        self.trainer.train(verbose_freq=TrainingConfig.PRINT_EVERY_N_EPISODES)
        
        # Log training to analytics
        self._log_training_to_analytics()
        
        # Evaluate
        print("\n✓ Evaluating...")
        self.eval_results = self.trainer.evaluate(num_episodes=TrainingConfig.EVAL_EPISODES)
        
        # Log evaluation to analytics
        self._log_evaluation_to_analytics()
        
        # Save all results
        self._save_model()
        self._save_config()
        
        # Generate analytics
        print("\n✓ Saving analytics summary...")
        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
        self.analytics.save_summary(config_dict=config_dict)
        
        print("\n✓ Generating analytics visualizations...")
        self.analytics.generate_plots()
        
        # Save heatmap data
        if self.heatmap_recorder:
            self._save_heatmap_data()
        
        # NEW: Save collision analysis
        if self.collision_analyzer:
            self._save_collision_data()
        
        self.end_time = datetime.datetime.now()
        duration = self.end_time - self.start_time
        
        # Final output summary:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Total Training Time: {duration}")
        print("=" * 70)
        print(f"\n✓ All results saved to: {self.output_dir}/")
        print(f"\n  Models ({SAVED_MODELS_DIR}/):")
        print(f"    - Model: {self.model_path.name}")
        print(f"    - Config: {self.config_path.name}")
        print(f"    - Log: {self.log_path.name}")
        print(f"\n  Analytics ({RECORDED_DATA_DIR}/):")
        print(f"    - Training CSV: {self.analytics.timestamp}.csv") 
        print(f"    - Evaluation CSV: {self.analytics.timestamp}_eval.csv")
        print(f"    - Summary: {self.analytics.timestamp}_summary.txt")
        print(f"    - Plots (10): {self.analytics.timestamp}_0*.png")
        if self.heatmap_recorder:
            print(f"\n  Heatmaps ({HEATMAP_DIR}/):")
            print(f"    - Heatmap Data: {self.heatmap_path.name}")
            print(f"    - Epochs: {self.heatmap_recorder.get_all_epochs_data()['num_epochs']}")
            print(f"    - Frames: {self.heatmap_recorder.get_all_epochs_data()['metadata']['total_frames']:,}")
        if self.collision_analyzer:
            print(f"\n  Collision Analysis ({COLLISION_DIR}/):")
            print(f"    - Report: collision_report.txt")
            print(f"    - Statistics: collision_statistics.json")
            print(f"    - Figures: figure_1_collision_distribution.png")
            print(f"    - Figures: figure_2_angle_progression.png")
            print(f"    - Figures: figure_3_edge_hit_progression.png")
            if self.collision_analyzer.epoch_stats:
                summary = self.collision_analyzer.get_summary_statistics()
                print(f"    - Total Collisions: {summary['total_collisions']:,}")
                print(f"    - Edge Hit Rate: {summary['initial_edge_percentage']:.1f}% → {summary['final_edge_percentage']:.1f}%")
                print(f"    - Avg Angle: {summary['initial_avg_angle']:.1f}° → {summary['final_avg_angle']:.1f}°")
        print("=" * 70 + "\n")
    
    
    def get_results(self):
        """Return all results as dictionary"""
        return {
            'trainer': self.trainer,
            'analytics': self.analytics,
            'heatmap_recorder': self.heatmap_recorder,
            'collision_analyzer': self.collision_analyzer,
            'eval_results': self.eval_results,
            'duration': self.end_time - self.start_time if self.end_time else None,
            'timestamp': self.timestamp,
            'model_path': str(self.model_path),
            'config_path': str(self.config_path),
            'heatmap_path': str(self.heatmap_path) if self.heatmap_recorder else None,
            'collision_path': str(COLLISION_DIR) if self.collision_analyzer else None,
            'output_dir': str(self.output_dir),
        }