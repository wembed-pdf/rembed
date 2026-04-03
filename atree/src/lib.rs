//! A high-performance spatial index for radius queries in D-dimensional Euclidean space.
//!
//! The core data structure is [`ATree`], which combines KD-tree-like axis-aligned
//! partitioning with SIMD-vectorized leaf scans and lookup-table-based pruning.
//! For cases where the dimensionality is not known at compile time, [`DynATree`]
//! provides the same functionality with a runtime dimension parameter.
//!
//! # Quick Start
//!
//! ```
//! use atree::ATree;
//!
//! let positions = vec![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
//! let tree = ATree::new(&positions);
//!
//! let mut results: Vec<u32> = Vec::new();
//! tree.query_radius(&[0.5, 0.5], 1.5, &mut results);
//! assert_eq!(results.len(), 3); // indices 0, 1, 2
//! ```
//!
//! The output type is controlled by the [`QueryOutput`](output::QueryOutput) trait.
//! Use integer types (`u32`, `usize`) for index-only results, or tuple types
//! (`(u32, f32)`, `(usize, f32)`) to also get squared distances.

pub mod dynamic;
pub mod output;
pub mod scalar;
pub mod simd;
pub mod svd;

mod iter;
mod query;
mod tree;

pub use dynamic::DynATree;
pub use tree::ATree;
