# config_seed.py
# =============================================================================
# RANDOM SEED CONFIGURATION
# =============================================================================

class SeedConfig:
    """Random seed configuration for reproducibility"""
    
    USE_SEED = False
    RANDOM_SEED = 42
    
    @classmethod
    def set_global_seed(cls):
        """Set numpy random seed if USE_SEED is True"""
        if cls.USE_SEED:
            import numpy as np
            np.random.seed(cls.RANDOM_SEED)
            print(f"Random seed set to {cls.RANDOM_SEED} for reproducibility")