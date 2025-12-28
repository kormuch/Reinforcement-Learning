# config_agent_policy_gradient.py
# =============================================================================
# POLICY GRADIENT AGENT CONFIGURATION - NON-VISUAL
# =============================================================================

from config_agent_base import AgentConfig

# IMPORTANT: Define the single source of truth for the config file path
json_path_agent = "config/config_agent.json"

class PolicyGradientAgentConfig(AgentConfig):
    """Non-Visual Policy Gradient Agent configuration (REINFORCE)"""
    
    # Network architecture (These are SAFE DEFAULTS, overwritten by JSON load below)
    INPUT_SIZE = 0                # Feature vector size
    HIDDEN_SIZE = 0              # Hidden layer neurons
    
    # Learning parameters
    # Set a default value, though it will be overwritten by JSON
    LEARNING_RATE = 0.003
    
    # Optimization (RMSprop)
    RMSPROP_DECAY = 0.99          # Decay rate for squared gradient accumulator
    RMSPROP_EPSILON = 1e-8        # Small constant for numerical stability
    
    @classmethod
    def get_agent_params(cls):
        """Returns dictionary of PolicyGradient-specific parameters"""
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
PolicyGradientAgentConfig.load_from_json()