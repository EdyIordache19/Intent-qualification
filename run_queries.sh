#!/bin/bash

# Define the input and output file names
INPUT_FILE="queries.txt"
OUTPUT_FILE="results.txt"

# Clear the output file if it already exists
> "$OUTPUT_FILE"

# Check if the input file exists before starting
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Cannot find $INPUT_FILE"
    exit 1
fi

# Read the input file line by line
while IFS= read -r query; do
    # Skip empty lines
    if [[ -z "$query" ]]; then
        continue
    fi

    echo ""
    echo "========================================="
    echo "Running query: $query"

    # Write a header for this query into the results file
    echo -e "\n=========================================" >> "$OUTPUT_FILE"
    echo "QUERY: $query" >> "$OUTPUT_FILE"

    # Run solution. Output goes to file, logs stay on terminal
    python3 solution.py \
        --query "$query" \
        --mode "cloud" \
        --model "meta-llama/llama-4-scout-17b-16e-instruct" \
        --top_k 10 >> "$OUTPUT_FILE"

    echo "Query complete. Sleeping for 30 seconds to reset API limits..."
    sleep 30

done < "$INPUT_FILE"

echo ""
echo "All queries processed. Clean results saved to $OUTPUT_FILE."