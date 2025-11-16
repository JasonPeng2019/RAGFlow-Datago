#!/bin/bash
# Test script to verify the fix

echo "==================================="
echo "Testing Move History Fix"
echo "==================================="

# Clean up old test data
rm -rf /scratch2/f004h1v/alphago_project/test_rag_output
mkdir -p /scratch2/f004h1v/alphago_project/test_rag_output

# Run a quick self-play game (just a few moves)
cd /scratch2/f004h1v/alphago_project/katago_repo/KataGo/cpp/build-opencl

# Create a minimal config for testing
cat > test_config.cfg << 'EOF'
maxVisits = 100
numSearchThreads = 1
nnMaxBatchSize = 16
nnCacheSizePowerOfTwo = 18
nnMutexPoolSizePowerOfTwo = 14
openclUseFP16Storage = true
openclUseFP16Compute = false
openclUseFP16TensorCores = false
EOF

echo "Running short selfplay game..."
./katago selfplay \
  -config test_config.cfg \
  -model ../tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz \
  -output-dir /scratch2/f004h1v/alphago_project/test_rag_output \
  -max-games-total 1 \
  -max-moves-per-game 20 2>&1 | head -100

echo ""
echo "==================================="
echo "Checking generated RAG data..."
echo "==================================="

# Find the generated JSON file
RAG_FILE=$(find ./rag_data* -name "RAG_rawdata_*.json" 2>/dev/null | head -1)

if [ -z "$RAG_FILE" ]; then
    echo "ERROR: No RAG data file found!"
    exit 1
fi

echo "Found RAG file: $RAG_FILE"
echo ""

# Extract and check first flagged position
python3 << 'PYTHON_SCRIPT'
import json
import sys

rag_file = sys.argv[1] if len(sys.argv) > 1 else None
if not rag_file:
    print("ERROR: No RAG file specified")
    sys.exit(1)

try:
    with open(rag_file, 'r') as f:
        data = json.load(f)

    print("=" * 60)
    print("RAG DATA SUMMARY:")
    print("=" * 60)
    print(f"Game ID: {data['game_id']}")
    print(f"Total flagged positions: {data['summary']['flagged_count']}")
    print(f"Total moves in game: {data['summary']['total_moves']}")
    print()

    if data['flagged_positions']:
        pos = data['flagged_positions'][0]
        print("=" * 60)
        print("FIRST FLAGGED POSITION:")
        print("=" * 60)
        print(f"Move number: {pos['move_number']}")
        print(f"Player to move: {pos['player_to_move']}")
        print(f"Moves history length: {len(pos['moves_history'])}")
        print(f"Expected moves history length: {pos['move_number']}")
        print()

        # Check if they match
        if len(pos['moves_history']) == pos['move_number']:
            print("✓ SUCCESS: moves_history length matches move_number!")
        else:
            print(f"✗ FAILURE: moves_history has {len(pos['moves_history'])} moves but move_number is {pos['move_number']}")
            print()
            print("Moves history:", pos['moves_history'][:10])

except Exception as e:
    print(f"ERROR reading RAG file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

echo ""
echo "Test complete!"