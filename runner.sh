#!/bin/sh

#ONLY FOR TRAINING
for i in $(seq "$1" "$2")
do
  echo "Running experiment_num: $i"
  python3 run_training.py --experiment_number "$i" --device_num "$3" --experiment_repeat "$4"
done
