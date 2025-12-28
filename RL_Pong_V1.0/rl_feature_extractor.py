# rl_feature_extractor.py

import numpy as np

# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================
class FeatureExtractor:
    """Extracts features from Pong game state for RL agent with memory"""
    
    def __init__(self, env):
        self.env = env
        self.feature_names = [
            'player_y', 'cpu_y', 'ball_x', 'ball_y',
            'ball_vx', 'ball_vy', 'player_score', 'cpu_score',
        ]
        
        # This is the size of the single state (8)
        self.num_features_single = len(self.feature_names) 
        
        # The reported size must be 16 (8 current + 8 previous)
        self.num_features = self.num_features_single * 2 
        
        # Stores the previous 8 features, initialized to zeros.
        self.previous_features = np.zeros((1, self.num_features_single), dtype=np.float32)

    
    def extract(self):
        """Extract feature vector from current game state and stack with previous"""
        game_info = self.env.get_game_info()
        ball_pos = game_info['ball_position']
        ball_vel = game_info['ball_velocity']
        
        # 1. Get current (unstacked) features (1x8 array)
        current_features = np.array([
            self.env.player_y,
            self.env.cpu_y,
            ball_pos[0],
            ball_pos[1],
            ball_vel[0],
            ball_vel[1],
            self.env.player_score,
            self.env.cpu_score,
        ], dtype=np.float32).reshape(1, -1)
        
        # 2. Stack features (CURRENT | PREVIOUS) -> This generates the (1, 16) array
        # The agent sees [State(t), State(t-1)]
        stacked_features = np.hstack([current_features, self.previous_features])
        
        # 3. Update memory for the next step (t+1)
        self.previous_features = current_features.copy()
        return stacked_features
    
    
    def normalize(self, features):
        """Normalize features to [-1, 1] range for better learning"""
        normalized = features.copy()
        
        # Helper function to apply normalization to a 1x8 slice of the features
        def apply_norm(arr_slice):
            arr_slice[0] = (arr_slice[0] - self.env.height/2) / (self.env.height/2)  # player_y
            arr_slice[1] = (arr_slice[1] - self.env.height/2) / (self.env.height/2)  # cpu_y
            arr_slice[2] = (arr_slice[2] - self.env.width/2) / (self.env.width/2)    # ball_x
            arr_slice[3] = (arr_slice[3] - self.env.height/2) / (self.env.height/2)  # ball_y
            arr_slice[4] = arr_slice[4] / 4.0                                        # ball_vx (assuming max vel is 4.0)
            arr_slice[5] = arr_slice[5] / 4.0                                        # ball_vy (assuming max vel is 4.0)
            arr_slice[6] = (arr_slice[6] - 10.5) / 10.5                              # player_score (assuming max score is 21)
            arr_slice[7] = (arr_slice[7] - 10.5) / 10.5                              # cpu_score
            return arr_slice

        # 1. Apply normalization to the CURRENT state (indices 0-7)
        normalized[0, 0:8] = apply_norm(features[0, 0:8])
        
        # 2. Apply normalization to the PREVIOUS state (indices 8-15)
        normalized[0, 8:16] = apply_norm(features[0, 8:16])
        
        return normalized

    def get_feature_description(self, features):
        """Get human-readable description of features"""
        lines = ["Game Features:"]
        # Only show the current features (first 8) for readability
        for i, name in enumerate(self.feature_names):
            value = features[0, i]
            lines.append(f"  {name:15s}: {value:7.2f}")
        return "\n".join(lines)