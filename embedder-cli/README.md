# embedder-cli

CLI tool for graph embedding using force-directed layout with multiple spatial index backends.

## Usage

```bash
cargo run -p embedder-cli -- -i <graph-file> -d <dimension> [OPTIONS]
```

## Examples

```bash
# 2D embedding with progress output
cargo run -p embedder-cli -- \
  -i data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400 \
  -d 2 --progress --f1-mode end

# 8D embedding, save positions to CSV
cargo run -p embedder-cli -- \
  -i data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400 \
  -d 8 -o positions.csv --max-iterations 500

# High-dimensional embedding (uses dynamic/heap-allocated fallback for D > 16)
cargo run -p embedder-cli -- \
  -i data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400 \
  -d 20 --progress

# Use plain Sprk index (no query caching)
cargo run -p embedder-cli -- \
  -i data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400 \
  -d 4 --index sprk
```

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i, --input` | String | *required* | Path to input graph edge list file |
| `-d, --dim` | usize | *required* | Embedding dimension |
| `--dim-hint` | usize | dim | Latent dimension hint |
| `-o, --output` | String | - | Output path for positions CSV |
| `--progress` | flag | false | Print iteration progress every 100 steps |
| `--f1-mode` | enum | `end` | `never` / `end` / `every` |
| `--index` | enum | `sprk-dynamic` | `sprk` / `sprk-dynamic` |
| `--learning-rate` | f64 | 10.0 | Learning rate for Adam optimizer |
| `--cooling-factor` | f64 | 0.99 | Cooling factor per iteration |
| `--max-iterations` | usize | 1000 | Maximum number of iterations |
| `--min-position-change` | f64 | 1e-8 | Minimum relative position change for convergence |
| `--attraction-scale` | f64 | 1.0 | Scale factor for attraction forces |
| `--repulsion-scale` | f64 | 1.0 | Scale factor for repulsion forces |
| `--print-timings` | flag | false | Print per-iteration timing breakdown |
| `--seed` | u64 | 42 | Random seed for initial positions |

## Dimension dispatch

Dimensions 1-16 are monomorphized (compile-time const generics for maximum performance). Dimensions > 16 use a dynamic heap-allocated fallback via `DynVec` and `DynSprk`.

