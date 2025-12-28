# rl_training_loop.py
"""
Training Loop for Non-Visual Pong Agent
Trains the Policy Gradient agent and tracks learning progress.

Run this to train the agent and see it learn to play Pong!

Usage:
    python train_pong_agent.py

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from env_custom_pong_simulator import CustomPongSimulator
from rl_feature_extractor import FeatureExtractor
from rl_policy_gradient_agent import PolicyGradientAgent
from config import EnvConfig, TrainingConfig


class PongTrainer:
    """Manages training loop for Pong agent"""
    
    def __init__(self, agent, env, feature_extractor, max_episodes=1000):
        """
        Initialize trainer.
        
        Args:
            agent: PolicyGradientAgent instance
            env: CustomPongSimulator instance
            feature_extractor: FeatureExtractor instance
            max_episodes: Maximum number of episodes to train
        """
        self.agent = agent
        self.env = env
        self.feature_extractor = feature_extractor
        self.max_episodes = max_episodes
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.running_reward = None
        self.best_reward = -float('inf')
    
    def play_episode(self, render=False, verbose=False):
        """
        Play one episode and collect trajectory.
        
        Args:
            render: If True, print ASCII rendering each step
            verbose: If True, print detailed info during episode
        
        Returns:
            tuple: (total_reward, episode_length, trajectory)
        """
        state = self.env.reset()
        trajectory = []
        total_reward = 0
        step_count = 0
        
        while True:
            # Extract features
            features = self.feature_extractor.extract()
            features_normalized = self.feature_extractor.normalize(features)
            
            # Agent selects action
            action, action_prob = self.agent.select_action(features_normalized)
            
            # Take action in environment
            state, reward, done, info = self.env.step(action)
            
            # Store in trajectory
            trajectory.append((features_normalized, action, reward))
            total_reward += reward
            step_count += 1
            
            if verbose and step_count % 100 == 0:
                print(f"  Step {step_count}: action={self.agent.get_action_names()[action]}, "
                      f"reward={reward:.2f}, cumulative={total_reward:.2f}")
            
            if render and step_count % 50 == 0:
                print(self.env.render('ascii'))
            
            if done:
                break
        
        return total_reward, step_count, trajectory
    
    def train(self, verbose_freq=50, render_freq=None):
        """
        Main training loop.
        
        Args:
            verbose_freq: Print progress every N episodes
            render_freq: Render every N episodes (None to disable)
        """
        print(f"\n{'='*60}")
        print(f"TRAINING NON-VISUAL PONG AGENT")
        print(f"{'='*60}")
        print(f"Agent: {self.agent.get_network_summary()}")
        print(f"Learning Rate: {self.agent.learning_rate}")
        print(f"Discount Factor: {self.agent.discount}")
        print(f"Max Episodes: {self.max_episodes}")
        print(f"{'='*60}\n")
        
        for episode in range(1, self.max_episodes + 1):
            # Play episode
            should_render = render_freq and episode % render_freq == 0
            total_reward, episode_length, trajectory = self.play_episode(
                render=should_render,
                verbose=False
            )
            
            # Train on trajectory
            self.agent.train_on_episode(trajectory)
            
            # Track metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            
            # Calculate running average
            if self.running_reward is None:
                self.running_reward = total_reward
            else:
                self.running_reward = 0.99 * self.running_reward + 0.01 * total_reward
            
            # Track best reward
            if total_reward > self.best_reward:
                self.best_reward = total_reward
            
            # Print progress
            if episode % verbose_freq == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Running Avg: {self.running_reward:7.2f} | "
                      f"Best: {self.best_reward:7.2f} | "
                      f"Length: {episode_length}")
            
            # Check for convergence
            if episode > 100 and self.running_reward > 0.5:
                print(f"\n✓ Agent learning well! Running average > 0.5")
                print(f"  Consider training longer or adjusting difficulty")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Final Running Average: {self.running_reward:.2f}")
        print(f"Best Episode Reward: {self.best_reward:.2f}")
        print(f"{'='*60}\n")
    
    def plot_learning_curve(self, window=50):
        """
        Plot learning progress.
        
        Args:
            window: Window size for moving average
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Raw rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        
        # Moving average
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
        ax.plot(self.episode_lengths, alpha=0.5, label='Episode Length')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Duration Over Time')
        ax.legend()
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
            ax.plot(self.agent.loss_history, alpha=0.7, label='Policy Loss')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Policy Gradient Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss history', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('pong_training_results.png', dpi=100)
        print("✓ Learning curve saved to 'pong_training_results.png'")
        plt.show()
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: If True, render episodes
        
        Returns:
            dict: Evaluation statistics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING TRAINED AGENT ({num_episodes} episodes)")
        print(f"{'='*60}\n")
        
        eval_rewards = []
        eval_wins = 0
        eval_lengths = []
        
        for ep in range(num_episodes):
            reward, length, _ = self.play_episode(render=render, verbose=False)
            eval_rewards.append(reward)
            eval_lengths.append(length)
            
            # Agent wins if score > 0
            if self.env.player_score > self.env.cpu_score:
                eval_wins += 1
            
            print(f"Eval Episode {ep+1:2d}: Reward={reward:7.2f}, "
                  f"Player={self.env.player_score:2d}, CPU={self.env.cpu_score:2d}, "
                  f"Length={length}")
        
        stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'win_rate': eval_wins / num_episodes,
            'mean_length': np.mean(eval_lengths),
        }
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"Win Rate: {stats['win_rate']:.1%}")
        print(f"Mean Episode Length: {stats['mean_length']:.0f} steps")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main training script"""
    # Create environment
    print("\n✓ Initializing Pong environment...")
    env = CustomPongSimulator(**EnvConfig.get_env_params_with_rewards())
    
    # Create feature extractor
    print("✓ Creating feature extractor...")
    feature_extractor = FeatureExtractor(env)
    print(f"  Features: {', '.join(feature_extractor.feature_names)}")
    
    # Create agent
    print("✓ Creating Policy Gradient agent...")
    agent = PolicyGradientAgent(
        input_size=feature_extractor.num_features,
        hidden_size=64,
        learning_rate=1e-3,
        discount=0.99
    )
    print(f"  Network: {agent.get_network_summary()['input_size']} → "
          f"{agent.get_network_summary()['hidden_size']} → "
          f"{agent.get_network_summary()['output_size']}")
    print(f"  Total Parameters: {agent.get_network_summary()['total_params']}")
    
    # Create trainer
    trainer = PongTrainer(agent, env, feature_extractor, 
                          max_episodes=TrainingConfig.MAX_EPISODES)
    
    # Train
    trainer.train(verbose_freq=50, render_freq=None)
    
    # Plot results
    print("\n✓ Generating training plots...")
    trainer.plot_learning_curve(window=50)
    
    # Evaluate
    print("\n✓ Evaluating trained agent...")
    trainer.evaluate(num_episodes=10, render=False)


if __name__ == "__main__":
    main()