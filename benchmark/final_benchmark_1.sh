#!/usr/bin/env bash

# Lists of parameters
N_VALUES=(10000 17783 100000 316228 1000000)
DIM_VALUES=(2 4 6 8 10 12 14 16)

# Number of parallel jobs
JOBS=32

parallel -j "$JOBS"  --line-buffer --tag  \
  cargo run -r bench --only-last-iteration \
  --n {1} --dim {2} --deg 15 --ple 2.2 --alpha "Inf" \
  --structures "atree" --structures "line-lsh" \
  --dynamic-download --seed 42 --store \
  ::: "${N_VALUES[@]}" ::: "${DIM_VALUES[@]}"
