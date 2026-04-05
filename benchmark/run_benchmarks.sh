#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Distribution Benchmark Runner
# Runs NN, Clustering (all-to-all), and POI benchmarks
# =============================================================================

# ── Configuration ────────────────────────────────────────────────────────────

# Data structure selections per benchmark type (comma-separated)
NN_STRUCTURES="atree,dyn_atree,sklearn_balltree,brute-force,kiddo,py_snn,wembed_snn"
CLUSTERING_STRUCTURES="atree,dyn_atree,sklearn_balltree,brute-force,kiddo,py_snn,wembed_snn"
POI_STRUCTURES="atree,dyn_atree,brute-force,kiddo,py_snn,wembed_snn,quadtree"

# Toggle --fast mode (set to "" to disable)
FAST="--fast"

# Maximum query points per benchmark (set to "" to disable sampling)
MAX_QUERY_POINTS="--max-query-points 1000"

# Output directory for results
OUTDIR="output"

# Dataset paths (relative to benchmark/ working directory)
NN_PATH="snn_datasets/export"
CLUSTERING_PATH="snn_datasets/export_clustering"
POI_PATH="poi/csv_output"

# Base command
CARGO="cargo run -r --features python -- bench-distributions"

# ── Helpers ──────────────────────────────────────────────────────────────────

mkdir -p "$OUTDIR"

run_nn() {
    local name="$1" train="$2" query="$3" radii="$4"
    echo "=== NN: $name ==="
    $CARGO \
        --benchmarksets "$train" \
        --benchmarksets-path "$NN_PATH" \
        --querysets "$query" \
        --radiuses "$radii" \
        --structures "$NN_STRUCTURES" \
        $FAST $MAX_QUERY_POINTS \
        -o "$OUTDIR/nn_${name}.csv"
}

run_clustering() {
    local name="$1" file="$2" eps="$3"
    echo "=== Clustering: $name ==="
    $CARGO \
        --benchmarksets "$file" \
        --benchmarksets-path "$CLUSTERING_PATH" \
        --all-to-all \
        --radiuses "$eps" \
        --structures "$CLUSTERING_STRUCTURES" \
        $FAST $MAX_QUERY_POINTS \
        -o "$OUTDIR/clustering_${name}.csv"
}

run_poi() {
    local name="$1" benchmarkset="$2" queryset="$3" radii="$4"
    echo "=== POI: $name ==="
    $CARGO \
        --benchmarksets "$benchmarkset" \
        --benchmarksets-path "$POI_PATH" \
        --querysets "$queryset" \
        --radiuses "$radii" \
        --structures "$POI_STRUCTURES" \
        $FAST $MAX_QUERY_POINTS \
        -o "$OUTDIR/poi_${name}.csv"
}

# ── Section 1: NN Dataset Benchmarks ─────────────────────────────────────────
# Radii from snn_nn.tex

run_nn fmnist    fmn_train.csv   fmn_query.csv   "800,900,1000,1100,1200"
run_nn sift10k   sift_train.csv sift_test.csv "210,230,250,270,290"
run_nn sift1m    sift_train_large.csv  sift_query_large.csv  "210,230,250,270,290"
run_nn gist      gist_train.csv  gist_query.csv  "0.8,0.85,0.9,0.95,1.0"
run_nn glove100  glo_train.csv   glo_query.csv   "0.94,0.97,1.01,1.04,1.07"
run_nn deep1b    deep_train.csv  deep_query.csv  "0.69,0.75,0.82,0.88,0.94"

# ── Section 2: Clustering (All-to-All) Benchmarks ───────────────────────────
# Eps from snn_clustering.tex

run_clustering banknote     banknote.csv     "0.1,0.2,0.3,0.4,0.5"
run_clustering dermatology  dermatology.csv  "5.0,5.1,5.2,5.3,5.4"
run_clustering ecoli        ecoli.csv        "0.5,0.6,0.7,0.8,0.9"
run_clustering phoneme      phoneme.csv      "8.5,8.6,8.7,8.8,8.9"
run_clustering wine         wine.csv         "2.2,2.3,2.4,2.5,2.6"

# ── Section 3: POI Benchmarks ───────────────────────────────────────────────
# Radii in meters

POI_RADII="500,1000,2000,5000"

run_poi parking_hospital     parking.csv     hospital.csv      "$POI_RADII"
run_poi restaurant_trainstation restaurant.csv trainstation.csv "$POI_RADII"
run_poi pharmacy_hospital    pharmacy.csv    hospital.csv      "$POI_RADII"
run_poi atm_supermarket      atm.csv         supermarket.csv   "$POI_RADII"

echo "=== All benchmarks complete. Results in $OUTDIR/ ==="
