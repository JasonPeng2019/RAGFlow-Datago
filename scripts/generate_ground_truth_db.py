"""
Generate synthetic ground truth database for uncertainty tuning.
Creates 10,000 positions with realistic shallow vs deep MCTS comparisons.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def generate_policy_distribution(num_moves: int = 6, concentration: float = 1.0) -> List[float]:
    """Generate a realistic policy distribution using Dirichlet"""
    # Higher concentration = more uniform, lower = more peaked
    alpha = np.ones(num_moves) * concentration
    policy = np.random.dirichlet(alpha)
    return policy.tolist()


def generate_value_distribution(mean_value: float, sharpness: float = 2.0) -> List[float]:
    """Generate value distribution (6 buckets for value ranges)"""
    # Value buckets: [-1, -0.6], [-0.6, -0.2], [-0.2, 0.2], [0.2, 0.6], [0.6, 1.0]
    # Map mean_value to bucket centers
    centers = np.array([-0.8, -0.4, 0.0, 0.4, 0.8, 0.95])
    
    # Distance from mean value to each center
    distances = np.abs(centers - mean_value)
    
    # Softmax with temperature (higher sharpness = more peaked)
    probs = np.exp(-sharpness * distances)
    probs = probs / probs.sum()
    
    # Add some noise
    noise = np.random.dirichlet(np.ones(6) * 0.5)
    probs = 0.8 * probs + 0.2 * noise
    probs = probs / probs.sum()
    
    return probs.tolist()


def compute_policy_entropy(policy: List[float]) -> float:
    """Compute Shannon entropy of policy distribution"""
    p = np.array(policy)
    p = p[p > 1e-10]  # Avoid log(0)
    return float(-np.sum(p * np.log(p)))


def compute_value_entropy(value_dist: List[float]) -> float:
    """Compute Shannon entropy of value distribution"""
    v = np.array(value_dist)
    v = v[v > 1e-10]
    return float(-np.sum(v * np.log(v)))


def compute_kl_divergence(p: List[float], q: List[float]) -> float:
    """KL divergence KL(p||q)"""
    p_arr = np.array(p)
    q_arr = np.array(q)
    
    # Avoid log(0) and division by zero
    mask = (p_arr > 1e-10) & (q_arr > 1e-10)
    
    if not np.any(mask):
        return 0.0
    
    return float(np.sum(p_arr[mask] * np.log(p_arr[mask] / q_arr[mask])))


def compute_cross_entropy(p: List[float], q: List[float]) -> float:
    """Cross entropy H(p, q) = -sum(p * log(q))"""
    p_arr = np.array(p)
    q_arr = np.array(q)
    
    mask = (p_arr > 1e-10) & (q_arr > 1e-10)
    
    if not np.any(mask):
        return 0.0
    
    return float(-np.sum(p_arr[mask] * np.log(q_arr[mask])))


def compute_sparseness(dist: List[float]) -> float:
    """Compute sparseness: 1 - normalized entropy"""
    entropy = compute_value_entropy(dist)
    max_entropy = np.log(len(dist))
    
    if max_entropy == 0:
        return 0.0
    
    normalized_entropy = entropy / max_entropy
    return float(1.0 - normalized_entropy)


def generate_position(game_id: int, pos_id: int, stones_on_board: int) -> Dict:
    """Generate a single position with shallow and deep MCTS results"""
    
    # Determine uncertainty level based on game phase
    # Mid-game tends to have more uncertainty
    game_progress = stones_on_board / 361.0
    
    # Peak uncertainty in mid-game (around 0.4-0.6 progress)
    mid_game_factor = 1.0 - abs(game_progress - 0.5) * 2
    base_uncertainty = 0.3 + 0.4 * mid_game_factor
    
    # Add random variation
    uncertainty_level = base_uncertainty + np.random.normal(0, 0.15)
    uncertainty_level = np.clip(uncertainty_level, 0.1, 0.9)
    
    # Generate shallow MCTS results (more uncertain)
    shallow_concentration = 2.0 - uncertainty_level * 1.5  # Lower = more spread out
    shallow_policy = generate_policy_distribution(num_moves=6, concentration=shallow_concentration)
    
    shallow_value = np.random.uniform(-0.9, 0.9)
    shallow_value_sharpness = 2.0 - uncertainty_level * 1.5
    shallow_value_dist = generate_value_distribution(shallow_value, shallow_value_sharpness)
    
    # Generate deep MCTS results (more certain, converges toward truth)
    # Deep MCTS should be similar to shallow but with less spread
    
    # For policy: deep focuses more on top moves
    deep_concentration = shallow_concentration + 0.5 + uncertainty_level * 1.0
    
    # Start with shallow policy and adjust toward more concentrated
    deep_policy = np.array(shallow_policy)
    
    # Add noise proportional to uncertainty
    noise_scale = uncertainty_level * 0.3
    noise = np.random.normal(0, noise_scale, size=6)
    deep_policy = deep_policy + noise
    deep_policy = np.maximum(deep_policy, 0.01)  # Ensure positive
    deep_policy = deep_policy / deep_policy.sum()  # Renormalize
    
    # Make top move more prominent in deep search
    top_idx = np.argmax(deep_policy)
    deep_policy[top_idx] *= (1.0 + uncertainty_level * 0.5)
    deep_policy = deep_policy / deep_policy.sum()
    
    deep_policy = deep_policy.tolist()
    
    # For value: deep converges with adjustment based on uncertainty
    value_shift = np.random.normal(0, uncertainty_level * 0.15)
    deep_value = shallow_value + value_shift
    deep_value = np.clip(deep_value, -1.0, 1.0)
    
    deep_value_sharpness = shallow_value_sharpness + 1.0 + uncertainty_level * 0.5
    deep_value_dist = generate_value_distribution(deep_value, deep_value_sharpness)
    
    # Compute features
    policy_entropy = compute_policy_entropy(shallow_policy)
    value_entropy = compute_value_entropy(shallow_value_dist)
    policy_cross_entropy = compute_cross_entropy(shallow_policy, deep_policy)
    value_sparseness = compute_sparseness(shallow_value_dist)
    
    # Compute errors
    policy_kl = compute_kl_divergence(shallow_policy, deep_policy)
    value_abs_error = abs(shallow_value - deep_value)
    policy_l2 = float(np.sqrt(np.sum((np.array(shallow_policy) - np.array(deep_policy))**2)))
    
    # Combined error (weighted)
    combined_error = 0.6 * policy_kl + 0.4 * value_abs_error
    
    position = {
        "id": f"game_{game_id}_pos_{pos_id}",
        "game_id": f"game_{game_id}",
        "move_number": stones_on_board,
        "stones_on_board": stones_on_board,
        "shallow_mcts": {
            "policy": [round(p, 6) for p in shallow_policy],
            "value": round(shallow_value, 6),
            "value_distribution": [round(v, 6) for v in shallow_value_dist],
            "visits": 800
        },
        "deep_mcts": {
            "policy": [round(p, 6) for p in deep_policy],
            "value": round(deep_value, 6),
            "value_distribution": [round(v, 6) for v in deep_value_dist],
            "visits": 5000
        },
        "features": {
            "policy_entropy": round(policy_entropy, 6),
            "value_entropy": round(value_entropy, 6),
            "policy_cross_entropy": round(policy_cross_entropy, 6),
            "value_sparseness": round(value_sparseness, 6)
        },
        "errors": {
            "policy_kl_divergence": round(policy_kl, 6),
            "value_absolute_error": round(value_abs_error, 6),
            "policy_l2_error": round(policy_l2, 6),
            "combined_error": round(combined_error, 6)
        }
    }
    
    return position


def generate_full_database(num_positions: int = 10000, num_games: int = 200) -> Dict:
    """Generate complete ground truth database"""
    
    print(f"Generating {num_positions} positions from {num_games} games...")
    
    positions = []
    positions_per_game = num_positions // num_games
    
    for game_id in range(num_games):
        if game_id % 20 == 0:
            print(f"  Game {game_id}/{num_games}...")
        
        for pos_id in range(positions_per_game):
            # Sample stones_on_board from realistic distribution
            # Early game (0-80): 20%
            # Mid game (80-200): 50%
            # Late game (200-320): 30%
            
            rand = np.random.random()
            if rand < 0.2:
                stones = np.random.randint(5, 80)
            elif rand < 0.7:
                stones = np.random.randint(80, 200)
            else:
                stones = np.random.randint(200, 320)
            
            position = generate_position(game_id, pos_id, stones)
            positions.append(position)
    
    database = {
        "metadata": {
            "description": "Ground truth database for uncertainty tuning",
            "total_positions": len(positions),
            "collection_date": "2025-11-12",
            "shallow_visits": 800,
            "deep_visits": 5000,
            "games_analyzed": num_games,
            "positions_per_game": positions_per_game,
            "generation_method": "synthetic",
            "error_combination": "0.6 * policy_kl + 0.4 * value_abs_error"
        },
        "positions": positions
    }
    
    print(f"Generated {len(positions)} positions")
    
    # Compute statistics
    errors = [p["errors"]["combined_error"] for p in positions]
    print(f"\nError statistics:")
    print(f"  Mean: {np.mean(errors):.6f}")
    print(f"  Std: {np.std(errors):.6f}")
    print(f"  Min: {np.min(errors):.6f}")
    print(f"  Max: {np.max(errors):.6f}")
    print(f"  Median: {np.median(errors):.6f}")
    print(f"  95th percentile: {np.percentile(errors, 95):.6f}")
    
    return database


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truth database")
    parser.add_argument("--num-positions", type=int, default=10000)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--output", type=str, default="./data/ground_truth_full.json")
    
    args = parser.parse_args()
    
    # Generate database
    database = generate_full_database(args.num_positions, args.num_games)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2)
    
    print(f"Done! Database size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
