# config_agent_cnn.py
# =============================================================================
# CNN AGENT CONFIGURATION - VISUAL (FUTURE)
# =============================================================================

from config_agent_base import AgentConfig
from config_environment import EnvConfig


class CNNAgentConfig(AgentConfig):
    """Visual CNN Agent configuration (for pixel-based learning)"""
    
    # Network architecture (for pixel-based agent)
    INPUT_SIZE = EnvConfig.WIDTH * EnvConfig.HEIGHT  # 30,720 for 160×192
    HIDDEN_SIZE = 200             # Hidden layer neurons
    
    # Learning parameters
    LEARNING_RATE = 1e-3          # Step size for gradient descent
    
    # Optimization (RMSprop)
    RMSPROP_DECAY = 0.99          # Decay rate for squared gradient accumulator
    RMSPROP_EPSILON = 1e-5        # Small constant for numerical stability
    
    @classmethod
    def get_agent_params(cls):
        """Returns dictionary of CNN-specific parameters"""
        params = super().get_agent_params()
        params.update({
            'input_size': cls.INPUT_SIZE,
            'hidden_size': cls.HIDDEN_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'rmsprop_decay': cls.RMSPROP_DECAY,
            'rmsprop_epsilon': cls.RMSPROP_EPSILON,
        })
        return params
    
    @classmethod
    def get_total_parameters(cls):
        """Calculate total number of network parameters"""
        # Input -> Hidden: INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE (bias)
        # Hidden -> Output: HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE (bias)
        return (cls.INPUT_SIZE * cls.HIDDEN_SIZE + cls.HIDDEN_SIZE + 
                cls.HIDDEN_SIZE * cls.OUTPUT_SIZE + cls.OUTPUT_SIZE)


# Auto-load from JSON when module is imported (inherits from AgentConfig)
CNNAgentConfig.load_from_json()