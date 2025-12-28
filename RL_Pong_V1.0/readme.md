# Policy Gradient Pong RL - Reproduction Package

Reinforcement learning agent trained with REINFORCE algorithm on custom Pong environment with progressive angle mechanics.

## Requirements

```bash
pip install numpy>=1.21.0 matplotlib>=3.5.0
```

**Optional (for GUI demo):**
```bash
pip install pygame>=2.1.0
```

## Quick Start

**Train the agent:**
```bash
python main_train_pong_gradient_agent.py
```

Training runs for 20,000 episodes (approximately 4-6 hours on CPU). All results save automatically to `outputs/exp_NAME_TIMESTAMP/`.

**Watch trained agent play (GUI):**
```bash
python main_demo_agent_gradient.py
```

## Configuration

Edit hyperparameters in JSON files:

**`config/config_training.json`** - Training settings
```json
{
  "EXPERIMENT_NAME": "exp_03",
  "MAX_EPISODES": 20000,
  "BATCH_SIZE": 5,
  "REWARD_SCORE": 20.0,
  "REWARD_BALL_HIT": 0.02,
  "REWARD_OPPONENT_SCORE": -1.0,
  "REWARD_WIN": 5000.0
}
```

**`config/config_agent.json`** - Network architecture
```json
{
  "INPUT_SIZE": 16,
  "HIDDEN_SIZE": 128,
  "LEARNING_RATE": 0.0015,
  "DISCOUNT_FACTOR": 0.99
}
```

## Output Structure

After training:
```
outputs/
└── exp_03_20251130_192852/
    ├── saved_models/
    │   ├── exp_03_20251130_192852.npz         # Trained weights
    │   └── exp_03_20251130_192852_config.json # Full configuration
    └── recorded_data/
        ├── 20251130_192852.csv                # Training metrics
        ├── 20251130_192852_eval.csv           # Evaluation results
        ├── 20251130_192852_summary.txt        # Statistics summary
        ├── 20251130_192852_01_learning_curve.png
        ├── ... (10 total plots)
        └── collision_analysis/
            ├── collision_summary.csv          # Edge-hit progression
            ├── collision_distribution.csv     # Full 12-position data
            ├── collision_report.txt
            └── figure_*.png (3 figures)
```

## Environment API

The custom Pong environment follows OpenAI Gym conventions:

```python
from env_custom_pong_simulator import CustomPongSimulator
from config import EnvConfig

# Initialize environment
env = CustomPongSimulator(**EnvConfig.get_env_params())

# Reset to initial state
state = env.reset()

# Game loop
for step in range(1000):
    action = agent.select_action(features)  # 0=STAY, 1=UP, 2=DOWN
    next_state, reward, done, info = env.step(action)
    
    if done:
        break

# Get game information
game_info = env.get_game_info()
# Returns: ball position, velocities, paddle positions, scores

# Render (optional)
rgb_array = env.render('rgb_array')  # or 'ascii'
```

**State space:** Dictionary with ball position, paddle positions, velocities  
**Action space:** Discrete(3) - {0: STAY, 1: UP, 2: DOWN}  
**Reward structure:** Configurable via `TrainingConfig`

## Key Features

**Progressive Angle System:** Edge hits produce steeper bounce angles (harder to defend)  
**Collision Analysis:** Automatic tracking of paddle hit positions and bounce angles  
**Reproducible:** All hyperparameters in JSON configs, no random seed fixing  
**Modular Design:** Swap agents/trainers by changing imports in `main_train_pong_gradient_agent.py`

## Reproducing Paper Results

Default configuration produces:
- **Win rate:** 8-10% (agent learns defense but not full strategy)
- **Edge-hit rate:** 60-65% (agent camps at edges)
- **Score:** ~16-17 vs 20-21 (competitive but loses)

Training takes ~4-6 hours on modern CPU. Exact numbers vary (no fixed random seed).

## File Descriptions

**Core Training:**
- `main_train_pong_gradient_agent.py` - Main training entry point
- `training_orchestrator.py` - Coordinates training pipeline
- `rl_policy_gradient_agent.py` - REINFORCE agent (2-layer network)
- `rl_pong_training_complete.py` - Training loop and evaluation

**Environment:**
- `env_custom_pong_simulator.py` - Main environment (Gym-like API)
- `env_physics_engine.py` - Ball movement and collisions
- `env_collision_detector.py` - Paddle collision detection
- `env_opponent_controller.py` - CPU opponent AI
- `env_ball_mechanics.py` - Progressive angle system

**Feature Extraction:**
- `rl_feature_extractor.py` - Converts game state to 16-feature vector

**Analysis:**
- `training_analytics.py` - Metrics logging and visualization
- `rl_collision_analyzer.py` - Paddle hit analysis and plots

**Configuration:**
- `config.py` - Main config import
- `config_training.py` - Training hyperparameters
- `config_agent_policygradient.py` - Agent architecture
- `config_environment.py` - Environment parameters

**Demo:**
- `main_demo_agent_gradient.py` - Watch trained agent play (Pygame GUI)

## Troubleshooting

**ImportError:** Install dependencies with `pip install -r requirements.txt`

**No config files:** Create `config/` directory and add JSON files (see Configuration section)

**Training too slow:** Reduce `MAX_EPISODES` in `config_training.json`

**Demo won't run:** Install pygame: `pip install pygame`

## Citation

If you use this code, please cite:

```
[Your Name]. (2025). Training a Policy Gradient Agent on Custom Pong with Progressive Angles.
[University], [Course].
```

## Contact

[Korbinian Much] - [korbinian.much@gmail.com]  
Project: Policy Gradient Methods for Atari Pong with Strategic Analysis

---

**License:** MIT