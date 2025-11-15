"""
Test script for Phase 1b: Relevance Weight Tuning

Creates a synthetic RAG database with duplicate sym_hash entries for testing.
"""

import json
import numpy as np
from pathlib import Path

def generate_test_database(output_path: str, num_unique_positions: int = 50, duplicates_per_position: int = 5):
    """
    Generate synthetic RAG database with duplicate sym_hash entries.
    
    Each unique game state (sym_hash) has multiple entries with:
    - Same game state (same sym_hash)
    - Different contexts (varied komi, winrate, score_lead)
    - Perturbed policy/value to simulate MCTS variance
    """
    print(f"Generating test database: {num_unique_positions} positions × {duplicates_per_position} duplicates...")
    
    positions = []
    
    for pos_idx in range(num_unique_positions):
        # Base sym_hash for this position
        sym_hash = f"test_pos_{pos_idx:04d}"
        
        # Base policy (random distribution)
        base_policy = np.random.dirichlet(np.ones(361) * 0.5)
        base_winrate = np.random.uniform(0.3, 0.7)
        base_score_lead = np.random.uniform(-10, 10)
        
        # Generate duplicates with perturbations
        for dup_idx in range(duplicates_per_position):
            # Perturb context features
            komi = 7.5 + np.random.choice([-1.5, 0, 1.5])  # Vary komi
            winrate = base_winrate + np.random.normal(0, 0.05)  # Small winrate variation
            winrate = np.clip(winrate, 0.0, 1.0)
            score_lead = base_score_lead + np.random.normal(0, 2.0)  # Score lead variation
            stone_count = np.random.randint(40, 150)
            
            # Perturb policy slightly (MCTS variance)
            policy = base_policy + np.random.dirichlet(np.ones(361) * 10.0) * 0.1
            policy = policy / policy.sum()  # Renormalize
            
            # Extend to 362 (including pass)
            policy_full = np.append(policy, [0.01])
            policy_full = policy_full / policy_full.sum()
            
            # Generate child nodes with visit distribution
            num_children = np.random.randint(5, 15)
            child_nodes = []
            for child_idx in range(num_children):
                child_nodes.append({
                    'hash': f"child_{pos_idx}_{dup_idx}_{child_idx}",
                    'value': np.random.uniform(0.3, 0.7),
                    'pUCT': np.random.uniform(0.5, 2.0),
                    'visits': np.random.randint(10, 200),
                    'move': np.random.randint(0, 361)
                })
            
            # Create position entry
            position = {
                'sym_hash': sym_hash,
                'state_hash': f"{sym_hash}_dup_{dup_idx}",
                'policy': policy_full.tolist(),
                'ownership': np.random.uniform(-1, 1, 361).tolist(),
                'winrate': float(winrate),
                'score_lead': float(score_lead),
                'komi': float(komi),
                'stone_count': int(stone_count),
                'child_nodes': child_nodes,
                'move_infos': [],
                'query_id': f"query_{pos_idx}_{dup_idx}"
            }
            
            positions.append(position)
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(positions, f, indent=2)
    
    print(f"✓ Generated {len(positions)} positions")
    print(f"  Unique sym_hashes: {num_unique_positions}")
    print(f"  Duplicates per position: {duplicates_per_position}")
    print(f"  Saved to: {output_path}")
    
    return output_path


def verify_database(db_path: str):
    """Verify the test database has correct structure"""
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nVerifying database: {db_path}")
    print(f"Total positions: {len(data)}")
    
    # Count sym_hash duplicates
    from collections import Counter
    sym_hashes = [p['sym_hash'] for p in data]
    hash_counts = Counter(sym_hashes)
    
    print(f"Unique sym_hashes: {len(hash_counts)}")
    print(f"Average duplicates per sym_hash: {np.mean(list(hash_counts.values())):.1f}")
    print(f"Min duplicates: {min(hash_counts.values())}")
    print(f"Max duplicates: {max(hash_counts.values())}")
    
    # Check feature variance within groups
    print("\nFeature variance within sym_hash groups:")
    example_hash = list(hash_counts.keys())[0]
    example_positions = [p for p in data if p['sym_hash'] == example_hash]
    
    komis = [p['komi'] for p in example_positions]
    winrates = [p['winrate'] for p in example_positions]
    score_leads = [p['score_lead'] for p in example_positions]
    
    print(f"  Example group (sym_hash={example_hash}):")
    print(f"    Komi range: {min(komis):.1f} - {max(komis):.1f}")
    print(f"    Winrate range: {min(winrates):.3f} - {max(winrates):.3f}")
    print(f"    Score lead range: {min(score_leads):.1f} - {max(score_leads):.1f}")
    
    print("\n✓ Database structure verified")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test database for Phase 1b")
    parser.add_argument("--output", type=str, default="./data/test_rag_phase1b.json",
                       help="Output path for test database")
    parser.add_argument("--num-positions", type=int, default=50,
                       help="Number of unique positions (sym_hashes)")
    parser.add_argument("--duplicates", type=int, default=5,
                       help="Duplicates per position")
    
    args = parser.parse_args()
    
    # Generate database
    db_path = generate_test_database(
        output_path=args.output,
        num_unique_positions=args.num_positions,
        duplicates_per_position=args.duplicates
    )
    
    # Verify it
    verify_database(db_path)
    
    print("\n" + "="*80)
    print("Test database ready!")
    print("="*80)
    print(f"\nRun Phase 1b with:")
    print(f"  python tuning/phase1/phase1b_relevance_weights.py \\")
    print(f"    --rag-database {db_path} \\")
    print(f"    --output-dir ./tuning_results/phase1b_test \\")
    print(f"    --method grid \\")
    print(f"    --min-group-size 3")
    print("="*80)
