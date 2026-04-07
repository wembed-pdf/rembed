# rembed

A graph embedding benchmark framework for comparing spatial indexing data structures on nearest neighbor queries in high-dimensional embeddings.

## What It Does

1. **Embeds graphs** into D-dimensional space using a force-directed algorithm with Adam optimization
2. **Indexes embeddings** with multiple spatial data structures (✨ SPRK-tree, KD-tree, VP-tree, LSH, grid, quadtree, etc.)
3. **Benchmarks** radius and nearest neighbor query performance across all structures
4. **Measures correctness** via precision, recall, and F1 score against brute-force ground truth

## Building

```sh
cargo build --release
```

### Optional C++ Backends

```sh
cargo build --release --features nanoflann   # nanoflann KD-tree
cargo build --release --features boost-rtree # Boost R-tree
cargo build --release --features cgal        # CGAL KD-tree
```

## Spatial Index Implementations

| Structure | Module | Notes |
|-----------|--------|-------|
| SPRK-tree | `sprk` | Sorted Projection Radius KD-tree: SIMD-vectorized with LUT leaves ([separate crate](sprk/)) |
| Kiddo | `kiddo` | Immutable KD-tree via `kiddo` crate |
| SIF KD-tree | `sif` | KD-tree via `sif-kdtree` crate |
| VP-tree | `vptree` | Vantage point tree via `acap` crate |
| Nabo | `nabo` | Approximate nearest neighbors |
| Grid | `grid` | 2D uniform grid |
| AGrid | `agrid` | Axis-aligned grid |
| Quadtree | `quadtree` | Quadtree spatial index |
| LSH | `measured_lsh`, `random_projection_lsh` | Locality-sensitive hashing |
| nanoflann | `nanoflann` | C++ KD-tree (feature-gated) |
| Boost R-tree | `boost_rtree` | C++ R-tree (feature-gated) |
| CGAL | `cgal_kdtree` | C++ KD-tree (feature-gated) |

All implementations share the `SpatialIndex<D>` trait, enabling uniform benchmarking.

## Benchmark System

Requires PostgreSQL. Set `DATABASE_URL`, `DATA_DIRECTORY`, and optionally `RSYNC_DESTINATION`.

```sh
cargo run --bin benchmark bench              # run benchmarks
cargo run --bin benchmark test               # correctness tests
cargo run --bin benchmark generate-graphs    # generate GIRG test graphs
cargo run --bin benchmark generate-positions # compute embeddings
cargo run --bin benchmark status             # check job queue
```

## License

MIT
