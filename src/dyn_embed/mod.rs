//! Dynamic-dimension embedding support.
//!
//! This module provides:
//! - [`EmbedIndex`] — a unified trait for spatial indices, bridging const-generic
//!   and dynamic variants.
//! - [`DynVec`] — a heap-allocated vector for runtime-dimension embeddings.
//! - [`DynDynSprk`] — a dynamic-dimension spatial index wrapping `sprk::DynSprk`.
//! - [`DynDynamicQuery`] — a caching query wrapper implementing [`EmbedIndex`].

pub mod dyn_dynamic_query;
pub mod dyn_sprk;
pub mod dyn_vec;

pub use dyn_dynamic_query::DynDynamicQuery;
pub use dyn_sprk::DynDynSprk;
pub use dyn_vec::DynVec;

use crate::NodeId;
use crate::dvec::Vector;

/// Trait unifying const-generic spatial indices and the dynamic variant.
///
/// Implemented via blanket impl for all `Embedder<'a, D>` types, and
/// directly for [`DynDynamicQuery`].
pub trait EmbedIndex: Clone + Sync {
    type Vec: Vector;

    fn position(&self, index: NodeId) -> &Self::Vec;
    fn num_nodes(&self) -> usize;
    fn weight(&self, index: NodeId) -> f64;
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool;
    fn neighbors(&self, index: NodeId) -> &[NodeId];
    fn update_positions(&mut self, positions: &[Self::Vec], last_delta: Option<f64>);
    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>);
    fn graph_statistics(&self) -> (f64, f64);
}

/// Implements [`EmbedIndex`] for a const-generic type that implements `Embedder<'a, D>`.
///
/// Usage: `impl_embed_index!(Sprk<'a, D>);`
#[macro_export]
macro_rules! impl_embed_index {
    ($ty:ty) => {
        impl<'a, const D: usize> $crate::dyn_embed::EmbedIndex for $ty {
            type Vec = $crate::dvec::DVec<D>;

            fn position(&self, index: $crate::NodeId) -> &$crate::dvec::DVec<D> {
                $crate::query::Position::position(self, index)
            }

            fn num_nodes(&self) -> usize {
                $crate::query::Position::num_nodes(self)
            }

            fn weight(&self, index: $crate::NodeId) -> f64 {
                $crate::query::Graph::weight(self, index)
            }

            fn is_connected(
                &self,
                first: $crate::NodeId,
                second: $crate::NodeId,
            ) -> bool {
                $crate::query::Graph::is_connected(self, first, second)
            }

            fn neighbors(&self, index: $crate::NodeId) -> &[$crate::NodeId] {
                $crate::query::Graph::neighbors(self, index)
            }

            fn update_positions(
                &mut self,
                positions: &[$crate::dvec::DVec<D>],
                last_delta: Option<f64>,
            ) {
                $crate::query::Update::update_positions(self, positions, last_delta);
            }

            fn repelling_nodes(&self, index: usize, result: &mut Vec<$crate::NodeId>) {
                $crate::query::Embedder::repelling_nodes(self, index, result);
            }

            fn graph_statistics(&self) -> (f64, f64) {
                $crate::query::Embedder::graph_statistics(self)
            }
        }
    };
}

// Implement EmbedIndex for the spatial indices used in the CLI.
// To add more, use: impl_embed_index!(crate::your_module::YourType<'a, D>);
impl_embed_index!(crate::sprk::Sprk<'a, D>);
impl_embed_index!(crate::dynamic_queries::DynamicQuery<'a, D, crate::sprk::Sprk<'a, D>>);
impl_embed_index!(crate::embedding::Embedding<'a, D>);
impl_embed_index!(crate::measured_lsh::MeasuredLSH<'a, D>);
impl_embed_index!(crate::random_projection_lsh::RandomProjectionLsh<'a, D>);
impl_embed_index!(crate::lossy_queries::LossyQuery<'a, D, crate::sprk::Sprk<'a, D>>);
// impl_embed_index!(crate::kiddo::Kiddo<'a, D>);
// impl_embed_index!(crate::vptree::VPTree<'a, D>);
// impl_embed_index!(crate::quadtree::Quadtree<'a, D>);
// impl_embed_index!(crate::grid::Grid<'a, D>);
// impl_embed_index!(crate::nabo::Nabo<'a, D>);
// impl_embed_index!(crate::naive_sprk::NaiveSprk<'a, D>);
// impl_embed_index!(crate::dyn_sprk::DynSprk<'a, D>);
// impl_embed_index!(crate::agrid::AGrid<'a, D>);
// impl_embed_index!(crate::sif::SIF<'a, D>);
// impl_embed_index!(crate::neighbourhood::Neihbourhood<'a, D>);
// impl_embed_index!(crate::random_projection_lsh::RandomProjectionLsh<'a, D>);
// impl_embed_index!(crate::random_projection_lsh::RandomProjectionLsh<'a, D>);
