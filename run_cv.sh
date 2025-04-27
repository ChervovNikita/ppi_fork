#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Experiment name and npy_file path are required."
  echo "Usage: $0 <name> <npy_file> [cv_count]"
  exit 1
fi

name="$1"
npy_file="$2"
cv_count="${3:-5}"
log_dir="log_dir/$name"

mkdir -p "$log_dir"
echo "Log directory created/ensured: $log_dir"

echo "Starting $cv_count cross-validation runs for experiment '$name'..."

for (( i=0; i<cv_count; i++ )); do
# for (( i=0; i<1; i++ )); do
  log_file="$log_dir/$i.log"
  echo "--- Running Fold $i/$cv_count ---"
  echo "Output will be logged to: $log_file"

  python train.py --cv_fold "$cv_count" --cv_fold_idx "$i" --npy_file "$npy_file" > "$log_file" 2>&1

  # Check exit status of the command (optional but recommended)
  if [ $? -eq 0 ]; then
    echo "Fold $i finished successfully."
  else
    echo "Error during Fold $i. Check log: $log_file"
    # exit 1
  fi
  echo "--------------------------"
done

echo "All $cv_count CV runs completed for experiment '$name'."
echo "Logs are available in $log_dir"
