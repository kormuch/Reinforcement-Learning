# config_agent_base.py
# =============================================================================
# BASE AGENT CONFIGURATION
# =============================================================================

import json
import os


class AgentConfig:
    """Base agent configuration - common parameters for all agents"""

    # Output size (same for all agents - Pong has 3 actions)
    OUTPUT_SIZE = 3  # Actions: stay, up, down

    # Common learning parameters
    DISCOUNT_FACTOR = 0.99  # Reward discount factor (gamma)

    @classmethod
    def get_agent_params(cls):
        """Returns dictionary of common agent parameters"""
        return {
            'output_size': cls.OUTPUT_SIZE,
            'discount_factor': cls.DISCOUNT_FACTOR,
        }

    @classmethod
    def load_from_json(cls, json_path="config/config_agent.json"):
        """
        Load agent parameters from JSON file.

        This method only loads parameters that are defined as attributes 
        in the *current* class (cls) or its base classes.
        """
        if not os.path.exists(json_path):
            print(f"⚠ No agent config file found at {json_path}. Using defaults.")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            key_upper = key.upper()
            
            # --- ADAPTATION: Check for key existence in the current class ---
            # This ensures we only load attributes that the class itself defines
            # or inherits (and thus already has a default value for).
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)
                print(f"✓ Loaded {key_upper} = {value}")
            elif key_upper == 'LEARNING_RATE': 
                # Temporary check for a common sub-class parameter to suppress 
                # the warning for the base class load.
                # NOTE: This line is a band-aid. The correct solution is below.
                pass 
            else:
                # This warning is harmless for subclass-specific keys, but good 
                # for catching typos in common/base keys.
                print(f"⚠ Unknown agent config key: {key}")

