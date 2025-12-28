# test_suite.py
"""
Physics Validation Test Suite for Atari-Authentic Pong
Tests all critical fixes and ensures authentic Pong behavior.

Updated for refactored architecture with modular components.

Run this after any changes to verify physics correctness.

Usage:
    python test_pong_physics.py
"""

import numpy as np
import sys
from env_custom_pong_simulator import CustomPongSimulator
from config import EnvConfig, TrainingConfig

class Colors:
    """ANSI color codes for pretty output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(test_name):
    """Print test header"""
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {test_name}")

def print_pass(message):
    """Print success message"""
    print(f"  {Colors.GREEN}✓ PASS:{Colors.END} {message}")

def print_fail(message):
    """Print failure message"""
    print(f"  {Colors.RED}✗ FAIL:{Colors.END} {message}")

def print_info(message):
    """Print info message"""
    print(f"  {Colors.YELLOW}ℹ INFO:{Colors.END} {message}")


def test_paddle_positions():
    """Test that paddles use config values, not hardcoded positions"""
    print_test("Paddle Position Configuration")
    
    env = CustomPongSimulator()
    
    # Check player paddle
    expected_player_x = EnvConfig.WIDTH - EnvConfig.PADDLE_OFFSET_FROM_EDGE
    if env.player_paddle_x == expected_player_x:
        print_pass(f"Player paddle X = {env.player_paddle_x} (using PADDLE_OFFSET_FROM_EDGE)")
    else:
        print_fail(f"Player paddle X = {env.player_paddle_x}, expected {expected_player_x}")
        return False
    
    # Check CPU paddle
    expected_cpu_x = EnvConfig.PADDLE_OFFSET_FROM_EDGE
    if env.cpu_paddle_x == expected_cpu_x:
        print_pass(f"CPU paddle X = {env.cpu_paddle_x} (using PADDLE_OFFSET_FROM_EDGE)")
    else:
        print_fail(f"CPU paddle X = {env.cpu_paddle_x}, expected {expected_cpu_x}")
        return False
    
    return True


def test_constant_ball_speed():
    """Test that ball maintains constant speed throughout game"""
    print_test("Constant Ball Speed (Atari-Authentic Physics)")
    
    if not EnvConfig.BALL_SPEED_CONSTANT:
        print_info("BALL_SPEED_CONSTANT is False - skipping test")
        return True
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    target_speed = env.physics.target_ball_speed
    print_info(f"Target ball speed: {target_speed:.4f}")
    
    speeds = []
    max_steps = 2000
    
    for step in range(max_steps):
        action = np.random.randint(0, 3)
        state, reward, done, info = env.step(action)
        
        game_info = env.get_game_info()
        current_speed = game_info['ball_speed']
        speeds.append(current_speed)
        
        if done:
            break
    
    speeds = np.array(speeds)
    min_speed = speeds.min()
    max_speed = speeds.max()
    mean_speed = speeds.mean()
    std_speed = speeds.std()
    
    print_info(f"Observed speeds over {len(speeds)} steps:")
    print_info(f"  Min: {min_speed:.4f}, Max: {max_speed:.4f}")
    print_info(f"  Mean: {mean_speed:.4f}, Std: {std_speed:.4f}")
    
    # Speed should be very close to target (tolerance: 0.01)
    tolerance = 0.01
    speed_deviation = abs(mean_speed - target_speed)
    
    if speed_deviation < tolerance and std_speed < 0.1:
        print_pass(f"Ball speed constant within tolerance (deviation: {speed_deviation:.6f})")
        return True
    else:
        print_fail(f"Ball speed not constant (deviation: {speed_deviation:.6f}, std: {std_speed:.6f})")
        return False


def test_distinct_state_values():
    """Test that state uses distinct values for different objects"""
    print_test("Distinct State Values")
    
    if not EnvConfig.USE_DISTINCT_VALUES:
        print_info("USE_DISTINCT_VALUES is False - skipping test")
        return True
    
    env = CustomPongSimulator()
    state = env.reset()
    
    unique_values = np.unique(state)
    print_info(f"Unique state values: {unique_values}")
    
    # Should have 4 distinct values: 0 (empty), 0.5 (CPU), 0.75 (ball), 1.0 (player)
    expected_values = {0.0, EnvConfig.STATE_VALUE_AI, 
                      EnvConfig.STATE_VALUE_BALL, EnvConfig.STATE_VALUE_PLAYER}
    
    if set(unique_values) == expected_values:
        print_pass("All expected state values present")
        print_info(f"  Empty: 0.0, CPU: {EnvConfig.STATE_VALUE_AI}, " +
                  f"Ball: {EnvConfig.STATE_VALUE_BALL}, Player: {EnvConfig.STATE_VALUE_PLAYER}")
        return True
    else:
        print_fail(f"State values don't match expected: {expected_values}")
        return False


def test_loser_serves():
    """Test that loser serves mechanism works correctly"""
    print_test("Loser Serves Mechanism (Atari-Authentic)")
    
    if not EnvConfig.LOSER_SERVES:
        print_info("LOSER_SERVES is False - skipping test")
        return True
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    # Test 1: CPU scores (player loses, player serves)
    # When CPU scores, player lost, so player serves
    # Ball should go toward CPU (negative vx)
    env.last_scorer = 'cpu'
    env._reset_ball_for_new_point()
    
    if env.physics.ball_vx < 0:
        print_pass("After CPU scores, ball serves toward CPU (negative vx) - Player serves")
    else:
        print_fail(f"After CPU scores, ball vx = {env.physics.ball_vx}, expected negative")
        return False
    
    # Test 2: Player scores (CPU loses, CPU serves)
    # When player scores, CPU lost, so CPU serves
    # Ball should go toward player (positive vx)
    env.last_scorer = 'player'
    env._reset_ball_for_new_point()
    
    if env.physics.ball_vx > 0:
        print_pass("After player scores, ball serves toward player (positive vx) - CPU serves")
    else:
        print_fail(f"After player scores, ball vx = {env.physics.ball_vx}, expected positive")
        return False
    
    return True


def test_collision_detection():
    """Test that paddle collisions are detected correctly"""
    print_test("Collision Detection")
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    # Force ball to hit player paddle
    env.physics.ball_x = float(env.player_paddle_x - 1)
    env.physics.ball_y = float(env.player_y + 5)
    env.physics.ball_vx = 2.0  # Moving right toward paddle
    env.physics.ball_vy = 0.0
    
    initial_vx = env.physics.ball_vx
    state, reward, done, info = env.step(0)  # No movement
    
    # Ball should have bounced (vx should reverse)
    if env.physics.ball_vx < 0 and reward > 0:
        print_pass(f"Player paddle collision detected (vx: {initial_vx:.1f} → {env.physics.ball_vx:.1f}, reward: {reward})")
    else:
        print_fail(f"Player paddle collision not detected properly (vx: {env.physics.ball_vx}, reward: {reward})")
        return False
    
    # Reset and test CPU paddle
    env.reset(seed=43)
    env.physics.ball_x = float(env.cpu_paddle_x + env.paddle_width + 1)
    env.physics.ball_y = float(env.cpu_y + 5)
    env.physics.ball_vx = -2.0  # Moving left toward CPU paddle
    env.physics.ball_vy = 0.0
    
    initial_vx = env.physics.ball_vx
    state, reward, done, info = env.step(0)
    
    # Ball should have bounced (vx should reverse to positive)
    if env.physics.ball_vx > 0:
        print_pass(f"CPU paddle collision detected (vx: {initial_vx:.1f} → {env.physics.ball_vx:.1f})")
    else:
        print_fail(f"CPU paddle collision not detected properly (vx: {env.physics.ball_vx})")
        return False
    
    return True


def test_scoring():
    """Test that scoring works correctly"""
    print_test("Scoring Mechanism")
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    # Force ball past CPU paddle (player scores)
    env.physics.ball_x = -5.0
    env.physics.ball_vx = -2.0
    
    initial_player_score = env.player_score
    state, reward, done, info = env.step(0)
    
    if env.player_score == initial_player_score + 1 and reward == TrainingConfig.REWARD_SCORE:
        print_pass(f"Player scoring works (score: {initial_player_score} → {env.player_score}, reward: {reward})")
    else:
        print_fail(f"Player scoring failed (score: {env.player_score}, reward: {reward})")
        return False
    
    # Reset and test CPU scoring
    env.reset(seed=43)
    env.physics.ball_x = float(env.width + 5)
    env.physics.ball_vx = 2.0
    
    initial_cpu_score = env.cpu_score
    state, reward, done, info = env.step(0)
    
    if env.cpu_score == initial_cpu_score + 1 and reward == TrainingConfig.REWARD_OPPONENT_SCORE:
        print_pass(f"CPU scoring works (score: {initial_cpu_score} → {env.cpu_score}, reward: {reward})")
    else:
        print_fail(f"CPU scoring failed (score: {env.cpu_score}, reward: {reward})")
        return False
    
    return True


def test_episode_termination():
    """Test that episodes terminate correctly"""
    print_test("Episode Termination")
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    # Test max score termination (player wins)
    env.player_score = EnvConfig.MAX_SCORE - 1
    env.physics.ball_x = -5.0
    env.physics.ball_vx = -2.0
    
    state, reward, done, info = env.step(0)
    
    if done and info['done_reason'] == 'player_won':
        print_pass(f"Episode ends when player reaches max score ({EnvConfig.MAX_SCORE})")
    else:
        print_fail(f"Episode should end when player wins (done={done}, reason={info.get('done_reason')})")
        return False
    
    # Test CPU wins
    env.reset(seed=43)
    env.cpu_score = EnvConfig.MAX_SCORE - 1
    env.physics.ball_x = float(env.width + 5)
    env.physics.ball_vx = 2.0
    
    state, reward, done, info = env.step(0)
    
    if done and info['done_reason'] == 'cpu_won':
        print_pass(f"Episode ends when CPU reaches max score ({EnvConfig.MAX_SCORE})")
    else:
        print_fail(f"Episode should end when CPU wins (done={done}, reason={info.get('done_reason')})")
        return False
    
    return True


def test_frame_stacking():
    """Test frame stacking if enabled"""
    print_test("Frame Stacking")
    
    if not EnvConfig.USE_FRAME_STACKING:
        print_info("USE_FRAME_STACKING is False - skipping test")
        return True
    
    env = CustomPongSimulator()
    state = env.reset(seed=42)
    
    # Check shape
    expected_shape = (EnvConfig.HEIGHT, EnvConfig.WIDTH, EnvConfig.FRAME_STACK_SIZE)
    
    if state.shape == expected_shape:
        print_pass(f"Frame stacking shape correct: {state.shape}")
    else:
        print_fail(f"Frame stacking shape: {state.shape}, expected {expected_shape}")
        return False
    
    # Take a step and verify frames update
    old_frame = state[:, :, -1].copy()
    state, _, _, _ = env.step(1)
    new_frame = state[:, :, -1].copy()
    
    if not np.array_equal(old_frame, new_frame):
        print_pass("Frame stack updates correctly after step")
    else:
        print_fail("Frame stack not updating")
        return False
    
    return True


def test_progressive_angles():
    """Test that progressive angle system works"""
    print_test("Progressive Angle Difficulty System")
    
    if not EnvConfig.USE_PROGRESSIVE_ANGLES:
        print_info("USE_PROGRESSIVE_ANGLES is False - skipping test")
        return True
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    # Check initial state
    initial_multiplier = env.ball_mechanics.current_angle_multiplier
    initial_volleys = env.ball_mechanics.volley_count
    
    if initial_multiplier == 1.0 and initial_volleys == 0:
        print_pass(f"Progressive angle system initializes correctly (multiplier: {initial_multiplier}, volleys: {initial_volleys})")
    else:
        print_fail(f"Initial state incorrect (multiplier: {initial_multiplier}, volleys: {initial_volleys})")
        return False
    
    # Simulate volleys to reach first threshold
    first_threshold = EnvConfig.ANGLE_INCREASE_VOLLEYS[0]
    
    for i in range(first_threshold):
        env.ball_mechanics.on_paddle_hit()
    
    if env.ball_mechanics.current_angle_multiplier > 1.0:
        print_pass(f"Angle multiplier increased at volley {first_threshold}: {env.ball_mechanics.current_angle_multiplier:.3f}")
    else:
        print_fail(f"Angle multiplier should increase at threshold (got {env.ball_mechanics.current_angle_multiplier})")
        return False
    
    # Test reset
    env.ball_mechanics.reset_rally()
    
    if env.ball_mechanics.current_angle_multiplier == 1.0 and env.ball_mechanics.volley_count == 0:
        print_pass("Progressive angle system resets correctly after point")
    else:
        print_fail(f"Reset failed (multiplier: {env.ball_mechanics.current_angle_multiplier}, volleys: {env.ball_mechanics.volley_count})")
        return False
    
    return True


def test_cpu_opponent_behavior():
    """Test that CPU opponent behaves reasonably"""
    print_test("CPU Opponent Behavior")
    
    env = CustomPongSimulator()
    env.reset(seed=42)
    
    # Run a short game
    cpu_movements = 0
    steps = 500
    
    # Force ball to move toward CPU and cross midline for testing
    for step_num in range(steps):
        old_cpu_y = env.cpu_y
        
        # Occasionally force ball toward CPU side for testing
        if step_num % 50 == 0:
            env.physics.ball_x = 40  # Left of midline
            env.physics.ball_vx = -2  # Moving toward CPU
            env.cpu_opponent.ball_crossed_midline = True
        
        state, reward, done, info = env.step(1)  # Player moves up
        
        if env.cpu_y != old_cpu_y:
            cpu_movements += 1
        
        if done:
            break
    
    movement_rate = cpu_movements / steps
    
    # CPU should move at least 5% of the time
    if movement_rate > 0.05:
        print_pass(f"CPU moves appropriately ({cpu_movements}/{steps} steps = {movement_rate:.1%})")
    else:
        print_fail(f"CPU barely moves ({cpu_movements}/{steps} steps = {movement_rate:.1%})")
        return False
    
    print_info(f"CPU difficulty: {EnvConfig.COMPUTER_DIFFICULTY} (speed: {env.cpu_speed})")
    print_info(f"CPU reaction rule enabled: {env.cpu_reaction_enabled}")
    
    return True


def test_reward_configuration():
    """Test that rewards are correctly configured from TrainingConfig"""
    print_test("Reward Configuration (from TrainingConfig)")
    
    env = CustomPongSimulator()
    
    if env.reward_score == TrainingConfig.REWARD_SCORE:
        print_pass(f"Reward for score: {env.reward_score}")
    else:
        print_fail(f"Score reward mismatch: {env.reward_score} vs {TrainingConfig.REWARD_SCORE}")
        return False
    
    if env.reward_opponent_score == TrainingConfig.REWARD_OPPONENT_SCORE:
        print_pass(f"Reward for opponent score: {env.reward_opponent_score}")
    else:
        print_fail(f"Opponent reward mismatch: {env.reward_opponent_score} vs {TrainingConfig.REWARD_OPPONENT_SCORE}")
        return False
    
    if env.reward_ball_hit == TrainingConfig.REWARD_BALL_HIT:
        print_pass(f"Reward for ball hit: {env.reward_ball_hit}")
    else:
        print_fail(f"Ball hit reward mismatch: {env.reward_ball_hit} vs {TrainingConfig.REWARD_BALL_HIT}")
        return False
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}ATARI-AUTHENTIC PONG - PHYSICS VALIDATION TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}(Refactored Architecture){Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    print(f"\n{Colors.YELLOW}Configuration:{Colors.END}")
    print(f"  Court: {EnvConfig.WIDTH}×{EnvConfig.HEIGHT}")
    print(f"  Ball Speed Constant: {EnvConfig.BALL_SPEED_CONSTANT}")
    print(f"  Progressive Angles: {EnvConfig.USE_PROGRESSIVE_ANGLES}")
    print(f"  Distinct Values: {EnvConfig.USE_DISTINCT_VALUES}")
    print(f"  Loser Serves: {EnvConfig.LOSER_SERVES}")
    print(f"  Frame Stacking: {EnvConfig.USE_FRAME_STACKING}")
    
    tests = [
        ("Paddle Positions", test_paddle_positions),
        ("Constant Ball Speed", test_constant_ball_speed),
        ("Distinct State Values", test_distinct_state_values),
        ("Loser Serves", test_loser_serves),
        ("Collision Detection", test_collision_detection),
        ("Scoring", test_scoring),
        ("Episode Termination", test_episode_termination),
        ("Frame Stacking", test_frame_stacking),
        ("Progressive Angle System", test_progressive_angles),
        ("CPU Opponent Behavior", test_cpu_opponent_behavior),
        ("Reward Configuration", test_reward_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print_fail(f"Test crashed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"  {status} - {test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED! Your Pong is Atari-authentic!{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}{total - passed} test(s) failed. Please review.{Colors.END}")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)