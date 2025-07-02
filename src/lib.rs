pub use embedding::Embedding;
use query::IndexClone;
pub use query::Query;
pub use std::io;
use std::ops::Deref;

pub mod atree;
pub mod ball_tree;
pub mod dim_reduction;
pub mod dvec;
pub mod embedding;
pub mod gpu_brute_force;
pub mod graph;
pub mod kd_tree;
pub mod kiddo;
pub mod lsh;
#[cfg(feature = "nanoflann")]
pub mod nanoflann;
pub mod neighbourhood;
pub mod parsing;
pub mod query;
pub mod sif;
pub mod snn;
pub mod vptree;
pub mod wrtree;

pub mod embedder;

pub type NodeId = usize;

pub fn convert_to_embeddings<'a, const D: usize>(
    iterations: &parsing::Iterations<D>,
    graph: &'a graph::Graph,
) -> impl DoubleEndedIterator<Item = Embedding<'a, D>> {
    iterations
        .iterations()
        .iter()
        .map(move |x| Embedding::<'a, D> {
            positions: x.positions.deref().clone(),
            graph,
        })
}

pub fn data_structures<'a, const D: usize>(
    embedding: &Embedding<'a, D>,
) -> impl ExactSizeIterator<Item = Box<dyn IndexClone<D> + 'a>> {
    let iter = [
        Box::new(atree::ATree::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(dim_reduction::LayeredLsh::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(kiddo::Kiddo::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(embedding.clone()) as Box<dyn IndexClone<D> + 'a>,
        Box::new(lsh::Lsh::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(snn::SNN::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(kd_tree::KDTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(ball_tree::WBallTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(neighbourhood::Neihbourhood::<D>::new(embedding.clone()))
            as Box<dyn IndexClone<D> + 'a>,
        Box::new(sif::SIF::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(vptree::VPTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(gpu_brute_force::GpuBruteForce::<D>::new(embedding.clone()))
            as Box<dyn IndexClone<D> + 'a>,
    ]
    .into_iter();

    #[cfg(feature = "nanoflann")]
    let iter = iter.chain(
        Box::new(nanoflann::NanoflannIndexWrapper::<D>::new(embedding))
            as Box<dyn IndexClone<D> + 'a>,
    );
    iter
}
