# rl_pong_training_complete.py

"""
Complete Non-Visual Pong Agent Training - All Parameters from Config
Contains FeatureExtractor, PolicyGradientAgent, and Training Loop in one file.

All hyperparameters are pulled from TrainingConfig for centralized management.
Enhanced with per-episode tracking for analytics, heatmap recording, and collision analysis.

Usage:
    python pong_training_complete.py

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from env_custom_pong_simulator import CustomPongSimulator
from rl_policy_gradient_agent import PolicyGradientAgent
from config import EnvConfig, TrainingConfig, PolicyGradientAgentConfig


# =============================================================================
# TRAINER
# =============================================================================

class PongTrainer:
    """Manages training loop for Pong agent - all config from TrainingConfig"""
    
    def __init__(self, agent, env, feature_extractor, max_episodes=None):
        self.agent = agent
        self.env = env
        self.feature_extractor = feature_extractor
        self.max_episodes = max_episodes or TrainingConfig.MAX_EPISODES
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.running_reward = None
        self.best_reward = -float('inf')
        
        # NEW: Track per-episode details for analytics
        self.episode_player_scores = []
        self.episode_cpu_scores = []
        self.episode_wins = []
        self.episode_score_rewards = []
        self.episode_hit_rewards = []
        self.episode_loss_penalties = []
        
        # NEW: Heatmap recorder (will be set by orchestrator)
        self.heatmap_recorder = None
        
        # NEW: Collision analyzer (will be set by orchestrator)
        self.collision_analyzer = None
        
        # NEW: Episodes per epoch
        self.episodes_per_epoch = 200
    
    def play_episode(self, render=False):
        """Play one episode and collect trajectory with reward tracking"""
        state = self.env.reset()
        trajectory = []
        total_reward = 0
        step_count = 0
        
        # Track reward components for this episode
        score_reward_sum = 0
        hit_reward_sum = 0
        loss_penalty_sum = 0
        win_bonus = 0
        
        while True:
            features = self.feature_extractor.extract()
            features_normalized = self.feature_extractor.normalize(features)
            
            action, action_prob = self.agent.select_action(features_normalized)
            state, reward, done, info = self.env.step(action)
            
            # NEW: Record frame to heatmap (if recorder available)
            if self.heatmap_recorder:
                game_info = self.env.get_game_info()
                self.heatmap_recorder.record_frame(
                    ball_x=game_info['ball_position'][0],
                    ball_y=game_info['ball_position'][1],
                    action=action,
                    player_paddle_y=game_info['player_paddle_y'],
                    cpu_paddle_y=game_info['cpu_paddle_y'],
                    scored=info.get('last_scorer')
                )
            
            # Track reward components (before win/loss bonus)
            if reward > 0:
                if reward == TrainingConfig.REWARD_SCORE:
                    score_reward_sum += reward
                elif reward == TrainingConfig.REWARD_BALL_HIT:
                    hit_reward_sum += reward
            elif reward < 0 and reward == TrainingConfig.REWARD_OPPONENT_SCORE:
                loss_penalty_sum += reward
            
            # Add win bonus if game ended
            if done:
                if self.env.player_score >= 21:
                    reward += TrainingConfig.REWARD_WIN  # Bonus for winning
                    win_bonus = TrainingConfig.REWARD_WIN
                elif self.env.cpu_score >= 21:
                    reward += TrainingConfig.REWARD_LOSS  # Penalty for losing
                    win_bonus = TrainingConfig.REWARD_LOSS
            
            trajectory.append((features_normalized, action, reward))
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Store episode metadata
        self.episode_player_scores.append(self.env.player_score)
        self.episode_cpu_scores.append(self.env.cpu_score)
        self.episode_wins.append(1 if self.env.player_score > self.env.cpu_score else 0)
        self.episode_score_rewards.append(score_reward_sum)
        self.episode_hit_rewards.append(hit_reward_sum)
        self.episode_loss_penalties.append(loss_penalty_sum)
        
        return total_reward, step_count, trajectory
    
    def train(self, verbose_freq=None):
        """Main training loop - uses TrainingConfig for all parameters"""
        verbose_freq = verbose_freq or TrainingConfig.PRINT_EVERY_N_EPISODES
        
        print(f"\n{'='*60}")
        print(f"TRAINING PONG AGENT")
        print(f"{'='*60}")
        print(f"Agent: {self.agent.get_network_summary()}")
        print(f"Learning Rate: {self.agent.LEARNING_RATE}")
        print(f"Discount Factor: {self.agent.DISCOUNT_FACTOR}")
        print(f"Max Episodes: {self.max_episodes}")
        print(f"RMSprop Decay: {self.agent.RMSPROP_DECAY}")
        print(f"Running Reward Decay: {TrainingConfig.RUNNING_REWARD_DECAY}")
        print(f"Reward Structure: Score={TrainingConfig.REWARD_SCORE}, Hit={TrainingConfig.REWARD_BALL_HIT}, Loss={TrainingConfig.REWARD_OPPONENT_SCORE}")
        print(f"Win Reward: {TrainingConfig.REWARD_WIN}, Loss Penalty: {TrainingConfig.REWARD_LOSS}")
        
        # Show heatmap configuration
        if self.heatmap_recorder:
            print(f"Heatmap Recording: ENABLED")
            print(f"Episodes per heatmap epoch: {self.episodes_per_epoch}")
        else:
            print(f"Heatmap Recording: DISABLED")
        
        # NEW: Show collision analyzer status
        if self.collision_analyzer:
            print(f"Collision Analysis: ENABLED")
            print(f"Episodes per collision epoch: {self.episodes_per_epoch}")
        else:
            print(f"Collision Analysis: DISABLED")
        
        print(f"{'='*60}\n")
        
        for episode in range(1, self.max_episodes + 1):
            # Track current episode for collision analyzer
            self.env._current_episode = episode
            
            total_reward, episode_length, trajectory = self.play_episode()
            
            self.agent.train_on_episode(trajectory)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            
            if self.running_reward is None:
                self.running_reward = total_reward
            else:
                self.running_reward = (TrainingConfig.RUNNING_REWARD_DECAY * self.running_reward + 
                                       (1 - TrainingConfig.RUNNING_REWARD_DECAY) * total_reward)
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
            
            # End episode in heatmap recorder
            if self.heatmap_recorder:
                self.heatmap_recorder.end_episode()
            
            # End heatmap epoch every N episodes
            if self.heatmap_recorder and (episode % self.episodes_per_epoch == 0):
                self.heatmap_recorder.end_epoch()
                print(f"  ✓ Heatmap epoch saved (episodes {episode - self.episodes_per_epoch + 1}-{episode})")
            
            # NEW: End collision epoch every N episodes
            if self.collision_analyzer and (episode % self.episodes_per_epoch == 0):
                epoch_num = (episode // self.episodes_per_epoch) - 1
                self.collision_analyzer.end_epoch(epoch_num)
                print(f"  ✓ Collision epoch {epoch_num} saved")
            
            if episode % verbose_freq == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Running Avg: {self.running_reward:7.2f} | "
                      f"Best: {self.best_reward:7.2f} | "
                      f"Length: {episode_length}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Final Running Average: {self.running_reward:.2f}")
        print(f"Best Episode Reward: {self.best_reward:.2f}")
        if self.heatmap_recorder:
            print(f"Total heatmap epochs: {len(self.heatmap_recorder.all_epochs)}")
        if self.collision_analyzer:
            print(f"Total collision epochs: {len(self.collision_analyzer.epoch_stats)}")
        print(f"{'='*60}\n")
    
    def plot_learning_curve(self, window=None):
        """Plot learning progress - uses TrainingConfig for window size"""
        window = window or TrainingConfig.PLOT_WINDOW_SIZE
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Raw rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, 
                                     np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                    label=f'Moving Avg (window={window})', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Curve - Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Episode length
        ax = axes[0, 1]
        ax.plot(self.episode_lengths, alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Duration Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of rewards
        ax = axes[1, 0]
        ax.hist(self.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(self.episode_rewards), color='r', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Training loss
        ax = axes[1, 1]
        if len(self.agent.loss_history) > 0:
            ax.plot(self.agent.loss_history, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Policy Gradient Loss')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = TrainingConfig.PLOT_SAVE_PATH
        plt.savefig(save_path, dpi=100)
        print(f"✓ Learning curve saved to '{save_path}'")
        plt.show()
    
    def evaluate(self, num_episodes=None):
        """Evaluate trained agent - returns detailed results for analytics"""
        num_episodes = num_episodes or TrainingConfig.EVAL_EPISODES
        
        print(f"\n{'='*60}")
        print(f"EVALUATING TRAINED AGENT ({num_episodes} episodes)")
        print(f"{'='*60}\n")
        
        eval_rewards = []
        eval_wins = 0
        eval_lengths = []
        eval_player_scores = []
        eval_cpu_scores = []
        eval_results = []
        
        for ep in range(num_episodes):
            reward, length, _ = self.play_episode()
            eval_rewards.append(reward)
            eval_lengths.append(length)
            eval_player_scores.append(self.env.player_score)
            eval_cpu_scores.append(self.env.cpu_score)
            
            is_win = 1 if self.env.player_score > self.env.cpu_score else 0
            if is_win:
                eval_wins += 1
            
            eval_results.append({
                'episode': ep + 1,
                'reward': reward,
                'player_score': self.env.player_score,
                'cpu_score': self.env.cpu_score,
                'length': length,
                'win': is_win
            })
            
            print(f"Eval Episode {ep+1:2d}: Reward={reward:7.2f}, "
                  f"Player={self.env.player_score:2d}, CPU={self.env.cpu_score:2d}, "
                  f"Length={length}")
        
        print(f"\n{'='*60}")
        print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Reward Range: [{np.min(eval_rewards):.2f}, {np.max(eval_rewards):.2f}]")
        print(f"Win Rate: {eval_wins/num_episodes:.1%}")
        print(f"Mean Episode Length: {np.mean(eval_lengths):.0f} steps")
        print(f"{'='*60}\n")
        
        # Return results for analytics logging
        return {
            'eval_rewards': eval_rewards,
            'eval_wins': eval_wins,
            'eval_lengths': eval_lengths,
            'eval_player_scores': eval_player_scores,
            'eval_cpu_scores': eval_cpu_scores,
            'eval_results': eval_results,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'win_rate': eval_wins / num_episodes
        }
    
    def get_training_data(self):
        """Return all tracked training data for analytics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_player_scores': self.episode_player_scores,
            'episode_cpu_scores': self.episode_cpu_scores,
            'episode_wins': self.episode_wins,
            'episode_score_rewards': self.episode_score_rewards,
            'episode_hit_rewards': self.episode_hit_rewards,
            'episode_loss_penalties': self.episode_loss_penalties,
        }