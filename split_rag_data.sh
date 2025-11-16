#!/bin/bash
# Script to split RAG data into 5 equal folders for parallel processing

set -e

RAG_SOURCE="/scratch2/f004h1v/alphago_project/raw_games_data/rag_data"
OUTPUT_BASE="/scratch2/f004h1v/alphago_project/raw_games_data"
NUM_SPLITS=5

echo "=========================================="
echo "RAG Data Splitting - 5 Equal Parts"
echo "=========================================="
echo ""

# Check if source directory exists
if [ ! -d "$RAG_SOURCE" ]; then
    echo "ERROR: RAG data directory not found at $RAG_SOURCE"
    exit 1
fi

# Count total files
total_files=$(find "$RAG_SOURCE" -name "RAG_rawdata_*.json" | wc -l)
if [ $total_files -eq 0 ]; then
    echo "ERROR: No RAG data files found in $RAG_SOURCE"
    exit 1
fi

files_per_split=$((total_files / NUM_SPLITS))
remainder=$((total_files % NUM_SPLITS))

echo "Total RAG files: $total_files"
echo "Files per split: $files_per_split (with $remainder extra in last split)"
echo ""

# Create split directories
for i in {1..5}; do
    split_dir="$OUTPUT_BASE/rag_data_split_$i"
    rm -rf "$split_dir"
    mkdir -p "$split_dir"
    echo "Created: $split_dir"
done

echo ""
echo "Distributing files..."

# Get all files into an array
mapfile -t all_files < <(find "$RAG_SOURCE" -name "RAG_rawdata_*.json" | sort)

# Split files across directories
split_num=1
files_in_current_split=0

for file in "${all_files[@]}"; do
    # Determine target split directory
    if [ $split_num -lt $NUM_SPLITS ]; then
        max_files=$files_per_split
    else
        # Last split gets remainder files
        max_files=$((files_per_split + remainder))
    fi

    # Copy file to current split
    cp "$file" "$OUTPUT_BASE/rag_data_split_$split_num/"

    ((files_in_current_split++))

    # Move to next split if current is full
    if [ $files_in_current_split -ge $max_files ] && [ $split_num -lt $NUM_SPLITS ]; then
        echo "Split $split_num: $files_in_current_split files"
        ((split_num++))
        files_in_current_split=0
    fi
done

# Print final split info
echo "Split $split_num: $files_in_current_split files"

echo ""
echo "Verifying splits..."
for i in {1..5}; do
    split_dir="$OUTPUT_BASE/rag_data_split_$i"
    count=$(find "$split_dir" -name "RAG_rawdata_*.json" 2>/dev/null | wc -l)
    echo "  Split $i: $count files"
done

echo ""
echo "=========================================="
echo "✓ RAG data split complete!"
echo "=========================================="
echo ""
echo "Split directories:"
for i in {1..5}; do
    echo "  $OUTPUT_BASE/rag_data_split_$i"
done
echo ""