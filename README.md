# rembed

A graph embedding benchmark framework for comparing spatial indexing data structures on nearest neighbor queries in high-dimensional embeddings.

## SPRK-tree

This repository was developed in conjunction with the paper "Benchmarking and Engineering Data Structures for Spherical Range Queries". The [SPRK-tree](https://github.com/wembed-pdf/sprk) is a SIMD-vectorized KD-tree with LUT leaves that was developed as part of this work. The SPRK-tree is implemented in a separate crate and is used as one of the spatial index implementations in this repository.

> **Looking to reproduce the results from the paper?** \
If you are mainly interested in reproducing the results from the paper, you can use the [reproducibility-cli](https://github.com/wembed-pdf/rembed/tree/reproducability/reproducibility-cli).

## What It Does

1. **Embeds graphs** into D-dimensional space using a force-directed algorithm with Adam optimization
2. **Indexes embeddings** with multiple spatial data structures (✨ SPRK-tree, KD-tree, VP-tree, LSH, grid, quadtree, etc.)
3. **Benchmarks** radius and nearest neighbor query performance across all structures
4. **Measures correctness** via precision, recall, and F1 score against brute-force ground truth

## Building

```sh
cargo build --release
```

### Optional Backends

```sh
cargo build --release --features nanoflann   # nanoflann KD-tree
cargo build --release --features boost-rtree # Boost R-tree
cargo build --release --features cgal        # CGAL KD-tree
cargo build --release --features wembed-snn  # Wembed SNN
cargo build --release --features sklearn     # scikit-learn KD-tree
cargo build --release --features py-snn      # Python SNN
```

## Spatial Index Implementations

| Structure | Module | Notes |
|-----------|--------|-------|
| SPRK-tree | `sprk` | Sorted Projection Radius KD-tree: SIMD-vectorized with LUT leaves ([separate crate](https://github.com/wembed-pdf/sprk)) |
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
| Wembed SNN | `wembed_snn` | C++ [library](https://github.com/Vraier/wembed) SNN (feature-gated) |
| scikit-learn | `sklearn_kdtree` | Python KD-tree (feature-gated) |
| Python SNN | `py_snn` | Python [SNN](https://github.com/nla-group/snn/) library (feature-gated) |

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

## Bug Reports

We encourage you to report any problems with rembed via the [github issue tracking system](https://github.com/wembed-pdf/rembed/issues). 
For issues regarding the SPRK crate, please use the [sprk github issue tracking system](https://github.com/wembed-pdf/sprk)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use Rembed in your research, please cite the following paper:

```
@misc{bläsius2026benchmarkingengineeringdatastructures,
      title={Benchmarking and Engineering Data Structures for Spherical Range Queries}, 
      author={Thomas Bläsius and Jean-Pierre von der Heydt and Tobias Kempf and Dennis Kobert and Nikolai Maas},
      year={2026},
      eprint={2607.07367},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2607.07367}, 
}
```
