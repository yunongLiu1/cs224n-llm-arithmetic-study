#!/bin/bash

# Source the configuration file
source config.sh

# Function to generate a comma-separated string of numbers with a given stride
generate_sequence() {
  local start=$1
  local end=$2
  local stride=$3
  numbers=$(seq "$start" "$stride" "$end")
  result=$(echo "$numbers" | paste -s -d, -)
  echo "$result"
}

# Iterate over the lengths
for length in "${lengths[@]}"; do
    # Iterate over the dataset suffixes
    for suffix in "${dataset_suffixes[@]}"; do
        dataset_name="${dataset_base}_${suffix}"
        save_dataset="$suffix"

        config_key="${length}_${suffix}"
        if [[ -z "${ops_config[$config_key]}" ]]; then
            # echo "Error: No ops configuration found for $config_key. Using default values."
            # ops=$(generate_sequence "$ops_start" "$ops_end" "$ops_stride")
            echo "Skipping ${dataset_name} because no ops configuration found for $config_key."
            continue  # Skip to the next iteration
        else
            ops_start=4
            ops_end=20
            ops_stride=$(echo "${ops_config[$config_key]}" | jq -r '.stride')
            ops=$(generate_sequence "$ops_start" "$ops_end" "$ops_stride")
        fi

        echo "Running with length: $length, dataset: $dataset_name, save-dataset: $save_dataset"

        python3 pred/pred.py \
            --dataset-name "$dataset_name" \
            --model-name "$model_name" \
            --save-dataset "$save_dataset" \
            --save-name "$save_name" \
            --backend-type "$backend_type" \
            --num-samples "$num_samples" \
            --temperature "$temperature" \
            --max-tokens "$max_tokens" \
            --length "$length" \
            --op-range "$ops" \
            --batch-size "$batch_size" \
            --limit "$limit" \
            --filter-config "$filter_config"

        python3 pred/eval_realistic.py \
            --save-dataset "$save_dataset" \
            --save-name "$save_name" \
            --num-samples "$num_samples" \
            --length "$length" \
            --filter-config "$filter_config"
    done
done
