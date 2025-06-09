pub use embedding::Embedding;
use query::IndexClone;
pub use query::Query;
pub use std::io;
use std::ops::Deref;

pub mod dvec;
pub mod embedding;
pub mod graph;
pub mod lsh;
pub mod parsing;
pub mod query;
pub mod snn;
pub mod wrtree;

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
    [
        Box::new(embedding.clone()) as Box<dyn IndexClone<D> + 'a>,
        Box::new(lsh::Lsh::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(wrtree::WRTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(snn::SNN::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
    ]
    .into_iter()
}
