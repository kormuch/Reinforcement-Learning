# rl_policy_gradient_agent.py
"""
Policy Gradient Agent for Non-Visual Pong Learning
Simple neural network agent using REINFORCE algorithm.

Author: Korbinian Much
Course: IU reinforcement learning
Date: 2025
"""
import numpy as np
import os
# NOTE: The import below is crucial for config access
# It should be changed to import the specific config class needed.
# Assuming PolicyGradientAgentConfig is available in the 'config' module:
from config import PolicyGradientAgentConfig 

class PolicyGradientAgent:
    """
    Policy Gradient Agent using REINFORCE algorithm.
    """
    
    def __init__(self): 
        """
        Initialize agent with neural network, relying ABSOLUTELY on the 
        PolicyGradientAgentConfig class parameters.
        """
        
        # ABSOLUTE assignment from the centralized configuration source
        # *** FIX: Changed ALL self attributes to be UPPERCASE to match the main script's access. ***
        self.INPUT_SIZE = PolicyGradientAgentConfig.INPUT_SIZE
        self.HIDDEN_SIZE = PolicyGradientAgentConfig.HIDDEN_SIZE
        self.OUTPUT_SIZE = PolicyGradientAgentConfig.OUTPUT_SIZE
        self.LEARNING_RATE = PolicyGradientAgentConfig.LEARNING_RATE
        self.DISCOUNT_FACTOR = PolicyGradientAgentConfig.DISCOUNT_FACTOR # Fixed attribute name mismatch

        # RMSprop parameters
        self.RMSPROP_DECAY = PolicyGradientAgentConfig.RMSPROP_DECAY
        self.RMSPROP_EPSILON = PolicyGradientAgentConfig.RMSPROP_EPSILON
        
        # Initialize network weights (small random values)
        self.W1 = np.random.randn(self.INPUT_SIZE, self.HIDDEN_SIZE) * 0.01
        self.b1 = np.zeros((1, self.HIDDEN_SIZE))
        self.W2 = np.random.randn(self.HIDDEN_SIZE, self.OUTPUT_SIZE) * 0.01
        self.b2 = np.zeros((1, self.OUTPUT_SIZE))
        
        # RMSprop accumulators
        self.W1_acc = np.zeros_like(self.W1)
        self.b1_acc = np.zeros_like(self.b1)
        self.W2_acc = np.zeros_like(self.W2)
        self.b2_acc = np.zeros_like(self.b2)
        
        # History for learning analysis
        self.loss_history = []
    
    def forward(self, features):
        """Forward pass through network."""
        # *** FIX: Using uppercase attributes ***
        self.hidden = np.maximum(0, np.dot(features, self.W1) + self.b1)
        logits = np.dot(self.hidden, self.W2) + self.b2
        self.action_probs = self._softmax(logits)
        return self.action_probs
    
    def _softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def select_action(self, features):
        """Select action based on current policy."""
        probs = self.forward(features)
        # *** FIX: Using uppercase attributes ***
        action = np.random.choice(self.OUTPUT_SIZE, p=probs[0])
        action_prob = probs[0, action]
        return action, action_prob
    
    def train_on_episode(self, trajectory):
        """Train agent on completed episode using REINFORCE."""
        if len(trajectory) == 0:
            return
        
        discounted_rewards = self._calculate_discounted_rewards(trajectory)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        dW1, db1, dW2, db2 = 0, 0, 0, 0
        
        for i, (features, action, _) in enumerate(trajectory):
            probs = self.forward(features)
            action_prob = probs[0, action]
            log_prob = np.log(action_prob + 1e-8)
            loss = -log_prob * discounted_rewards[i]
            
            action_one_hot = np.zeros((1, self.OUTPUT_SIZE)) # *** FIX: Using uppercase attributes ***
            action_one_hot[0, action] = 1
            
            d_logits = (probs - action_one_hot) * discounted_rewards[i]
            dW2 += np.dot(self.hidden.T, d_logits)
            db2 += d_logits
            d_hidden = np.dot(d_logits, self.W2.T)
            d_hidden[self.hidden <= 0] = 0
            dW1 += np.dot(features.T, d_hidden)
            db1 += d_hidden
        
        dW1 /= len(trajectory)
        db1 /= len(trajectory)
        dW2 /= len(trajectory)
        db2 /= len(trajectory)
        
        self._rmsprop_update(dW1, db1, dW2, db2)
        
        mean_loss = np.mean([-np.log(self.forward(trajectory[i][0])[0, trajectory[i][1]] + 1e-8) * discounted_rewards[i]
                              for i in range(len(trajectory))])
        self.loss_history.append(mean_loss)
    
    def _calculate_discounted_rewards(self, trajectory):
        """Calculate discounted cumulative rewards."""
        discounted_rewards = np.zeros(len(trajectory))
        cumulative_reward = 0
        for t in reversed(range(len(trajectory))):
            _, _, reward = trajectory[t]
            # *** FIX: Using uppercase attributes ***
            cumulative_reward = reward + self.DISCOUNT_FACTOR * cumulative_reward 
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards
    
    def _rmsprop_update(self, dW1, db1, dW2, db2):
        """RMSprop optimization step."""
        # *** FIX: Using uppercase attributes ***
        self.W1_acc = self.RMSPROP_DECAY * self.W1_acc + (1 - self.RMSPROP_DECAY) * (dW1 ** 2)
        self.b1_acc = self.RMSPROP_DECAY * self.b1_acc + (1 - self.RMSPROP_DECAY) * (db1 ** 2)
        self.W2_acc = self.RMSPROP_DECAY * self.W2_acc + (1 - self.RMSPROP_DECAY) * (dW2 ** 2)
        self.b2_acc = self.RMSPROP_DECAY * self.b2_acc + (1 - self.RMSPROP_DECAY) * (db2 ** 2)
        
        self.W1 -= self.LEARNING_RATE * dW1 / (np.sqrt(self.W1_acc) + self.RMSPROP_EPSILON)
        self.b1 -= self.LEARNING_RATE * db1 / (np.sqrt(self.b1_acc) + self.RMSPROP_EPSILON)
        self.W2 -= self.LEARNING_RATE * dW2 / (np.sqrt(self.W2_acc) + self.RMSPROP_EPSILON)
        self.b2 -= self.LEARNING_RATE * db2 / (np.sqrt(self.b2_acc) + self.RMSPROP_EPSILON)
    
    def get_action_names(self):
        """Return action names for debugging"""
        return ['stay', 'up', 'down']
    
    def get_network_summary(self):
        """Get summary of network architecture"""
        # *** FIX: Using uppercase attributes ***
        return {
            'input_size': self.INPUT_SIZE,
            'hidden_size': self.HIDDEN_SIZE,
            'output_size': self.OUTPUT_SIZE,
            'total_params': (self.INPUT_SIZE * self.HIDDEN_SIZE + 
                             self.HIDDEN_SIZE + 
                             self.HIDDEN_SIZE * self.OUTPUT_SIZE + 
                             self.OUTPUT_SIZE)
        }
    
    # === ✅ Added Save / Load Methods ===
    def save(self, path):
        """
        Save model weights and parameters to a .npz file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W1_acc=self.W1_acc,
            b1_acc=self.b1_acc,
            W2_acc=self.W2_acc,
            b2_acc=self.b2_acc,
            loss_history=np.array(self.loss_history)
        )
        print(f"✓ Agent parameters saved to '{path}'")

    def load(self, path):
        """
        Load model weights and parameters from a .npz file.
        """
        data = np.load(path, allow_pickle=True)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W1_acc = data["W1_acc"]
        self.b1_acc = data["b1_acc"]
        self.W2_acc = data["W2_acc"]
        self.b2_acc = data["b2_acc"]
        self.loss_history = data["loss_history"].tolist()
        print(f"✓ Agent parameters loaded from '{path}'")