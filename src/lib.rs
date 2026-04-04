pub use embedding::Embedding;
use query::IndexClone;
pub use query::Query;
pub use std::io;
use std::ops::Deref;

pub mod agrid;
pub mod atree;
#[cfg(feature = "boost-rtree")]
pub mod boost_rtree;
#[cfg(feature = "cgal")]
pub mod cgal_kdtree;
pub mod dvec;
pub mod dyn_atree;
pub mod dynamic_queries;
pub mod embedding;
pub mod graph;
pub mod grid;
pub mod kiddo;
pub mod lossy_queries;
pub mod measured_lsh;
pub mod nabo;
#[cfg(feature = "nanoflann")]
pub mod nanoflann;
pub mod neighbourhood;
pub mod parsing;
#[cfg(feature = "py-snn")]
pub mod py_snn;
pub mod quadtree;
pub mod query;
pub mod random_projection_lsh;
pub mod sif;
#[cfg(feature = "sklearn")]
pub mod sklearn;
pub mod vptree;
#[cfg(feature = "wembed-snn")]
pub mod wembed_snn;

pub use atree::ATree;
pub use dynamic_queries::DynamicQuery;
pub use embedder::WEmbedder;
pub use kiddo::Kiddo;
pub use lossy_queries::LossyQuery;
pub use measured_lsh::MeasuredLSH;
pub use random_projection_lsh::RandomProjectionLsh;

pub mod intrinsic_dimension;

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
        Box::new(dyn_atree::DynATree::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(agrid::AGrid::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(kiddo::Kiddo::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(nabo::Nabo::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(embedding.clone()) as Box<dyn IndexClone<D> + 'a>,
        Box::new(neighbourhood::Neihbourhood::<D>::new(embedding.clone()))
            as Box<dyn IndexClone<D> + 'a>,
        // Box::new(sif::SIF::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(vptree::VPTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(quadtree::Quadtree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(grid::Grid::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
    ]
    .into_iter();

    #[cfg(feature = "nanoflann")]
    let iter = iter.chain(Some(
        Box::new(nanoflann::NanoflannIndexWrapper::<D>::new(embedding))
            as Box<dyn IndexClone<D> + 'a>,
    ));

    #[cfg(feature = "boost-rtree")]
    let iter = iter.chain(Some(
        Box::new(boost_rtree::BoostRTreeWrapper::<D>::new(embedding))
            as Box<dyn IndexClone<D> + 'a>,
    ));

    #[cfg(feature = "cgal")]
    let iter = iter.chain(Some(
        Box::new(cgal_kdtree::CgalKdTreeWrapper::<D>::new(embedding))
            as Box<dyn IndexClone<D> + 'a>,
    ));

    #[cfg(feature = "wembed-snn")]
    let iter = iter.chain(Some(
        Box::new(wembed_snn::WembedSnnWrapper::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
    ));

    #[cfg(feature = "sklearn")]
    let iter = iter.chain([
        Box::new(sklearn::SklearnKDTree::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(sklearn::SklearnBallTree::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
    ]);

    #[cfg(feature = "py-snn")]
    let iter = iter.chain(Some(
        Box::new(py_snn::PySnn::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>
    ));

    iter.collect::<Vec<_>>().into_iter()
}
