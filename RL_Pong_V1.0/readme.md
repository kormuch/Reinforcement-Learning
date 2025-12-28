# Policy Gradient Pong RL - Reproduction Package

Reinforcement learning agent trained with REINFORCE algorithm on custom Pong environment with progressive angle mechanics.

## Requirements

```bash
pip install numpy>=1.21.0 matplotlib>=3.5.0 pandas>=1.3.0
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

Training runs for 20,000 episodes (approximately 6 hours on Intel i7 3.6 GHz CPU). All results save automatically to `outputs/exp_NAME_TIMESTAMP/`.

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
        ├── 20251130_192852.csv                # Episode-by-episode training metrics
        ├── 20251130_192852_eval.csv           # Evaluation results
        ├── 20251130_192852_summary.txt        # Statistics summary
        ├── 20251130_192852_01_learning_curve.png
        ├── ... (10 total plots)
        └── collision_analysis/
            ├── collision_summary.csv          # Edge-hit progression by epoch
            ├── collision_distribution.csv     # Full 12-position hit data
            ├── collision_statistics.json      # Detailed collision metrics
            ├── collision_report.txt
            └── figure_*.png (3 figures)
```

## Aggregated Results (Multi-Run Analysis)

To analyze results across multiple experiments:

**Generate aggregated statistics:**
```bash
python evaluation/aggregate_statistics.py
```

This produces:
```
evaluation/
├── aggregated_statistics.json          # Mean and std across all runs
├── aggregated_statistics.txt           # Human-readable summary
└── average_results_TIMESTAMP/
    ├── learning_curve.png              # Training reward over 20,000 episodes
    ├── edge_hit_progression.png        # Edge-hit % across all epochs (line)
    ├── collision_distribution.png      # Early vs Late positions (bar chart)
    ├── episode_length_progression.png  # Episode length over time
    ├── performance_table_simple.png    # Win rate & score comparison
    └── performance_table_detailed.png  # Complete agent metrics
```

**Paper figures script:**
```bash
python generate_paper_figures.py
```

Generates publication-ready figures (300 DPI PNG) from aggregated data across 5 independent runs.

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

**Progressive Angle System:** Edge hits produce steeper bounce angles (up to 75°), making returns harder to defend  
**Collision Analysis:** Automatic tracking of paddle hit positions across 12 segments and bounce angles  
**Multi-Run Aggregation:** Built-in tools to analyze statistics across multiple independent training runs  
**Reproducible:** All hyperparameters stored in JSON configs  
**Modular Design:** Easy to swap agents/trainers by changing imports  
**Small Network:** Only 2,563 parameters - trains quickly on CPU without GPU

## Reproducing Paper Results

The paper reports results averaged over **5 independent runs** with different random initializations (exp_03 through exp_07).

**Expected performance (n=5):**
- **Win rate:** μ = 8.0% (σ = 3.36%)
- **Mean score:** μ = 15.7 (σ = 1.09) vs μ = 20.9 (σ = 0.03) for CPU
- **Edge-hit rate (early):** μ = 59.6% (σ = 2.2%)
- **Edge-hit rate (late):** μ = 98.9% (σ = 0.4%)
- **Mean episode length:** μ = 4,329 steps (σ = 36)

Training takes approximately 6 hours per run on a standard CPU (Intel i7, 3.6 GHz). Variance across runs is expected due to stochastic initialization and opponent behavior.

**Key finding:** Agent successfully learns to exploit edge-hitting (98.9% of contacts) but fails to develop serve reception strategy, explaining the low win rate despite strong offensive play.

## File Descriptions

**Core Training:**
- `main_train_pong_gradient_agent.py` - Main training entry point
- `training_orchestrator.py` - Coordinates training pipeline
- `rl_policy_gradient_agent.py` - REINFORCE agent (2-layer network, 2,563 params)
- `rl_pong_training_complete.py` - Training loop and evaluation

**Environment:**
- `env_custom_pong_simulator.py` - Main environment (Gym-like API)
- `env_physics_engine.py` - Ball movement and collisions
- `env_collision_detector.py` - Paddle collision detection (12 segments)
- `env_opponent_controller.py` - Rule-based CPU opponent
- `env_ball_mechanics.py` - Progressive angle system

**Feature Extraction:**
- `rl_feature_extractor.py` - Converts game state to 16-feature vector

**Analysis:**
- `training_analytics.py` - Metrics logging and visualization
- `rl_collision_analyzer.py` - Paddle hit analysis and plots
- `evaluation/aggregate_statistics.py` - Multi-run statistics aggregation
- `generate_paper_figures.py` - Publication-ready figure generation

**Configuration:**
- `config.py` - Main config import
- `config_training.py` - Training hyperparameters
- `config_agent_policygradient.py` - Agent architecture
- `config_environment.py` - Environment parameters

**Demo:**
- `main_demo_agent_gradient.py` - Watch trained agent play (Pygame GUI)

## Experimental Setup

For reproducibility, the paper uses:
- **5 independent training runs** (exp_03, exp_04, exp_05, exp_06, exp_07)
- **20,000 episodes** per run (~6 hours each)
- **Network:** 2 hidden layers (128 units each), 2,563 total parameters
- **Learning rate:** 0.0015
- **Batch size:** 5 episodes
- **Discount factor:** 0.99
- **No fixed random seed** (evaluates robustness to initialization)

Results are reported as mean (μ) and standard deviation (σ) across the 5 runs.

## Troubleshooting

**ImportError:** Install dependencies with `pip install numpy matplotlib pandas`

**No config files:** Ensure `config/` directory exists with JSON config files (see Configuration section)

**Training too slow:** Reduce `MAX_EPISODES` in `config_training.json` or run on faster CPU

**Demo won't run:** Install pygame: `pip install pygame`

**Aggregation script fails:** Ensure all experiment folders (exp_03 through exp_07) exist in `evaluation/` directory with correct CSV files

## Citation

If you use this code, please cite:

```
Much, K. (2025). Training a Policy Gradient Agent on Custom Pong with Progressive Angles.
IU International University of Applied Sciences, Reinforcement Learning Course.
```

## Contact

Korbinian Much - korbinian.much@gmail.com  
Project: Policy Gradient Methods for Atari Pong with Strategic Analysis

---

**License:** MIT  
**Paper:** Full technical report available in `paper/` directory
