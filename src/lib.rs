pub use embedding::Embedding;
use query::IndexClone;
pub use query::Query;
pub use std::io;
use std::ops::Deref;

pub mod atree;
pub mod ball_tree;
#[cfg(feature = "boost-rtree")]
pub mod boost_rtree;
#[cfg(feature = "cgal")]
pub mod cgal_kdtree;
pub mod dim_reduction;
pub mod dvec;
pub mod dynamic_queries;
pub mod embedding;
pub mod graph;
pub mod hnsw;
pub mod kd_tree;
pub mod kiddo;
pub mod lossy_queries;
pub mod lsh;
pub mod measured_lsh;
pub mod nabo;
#[cfg(feature = "nanoflann")]
pub mod nanoflann;
pub mod neighbourhood;
pub mod parsing;
pub mod query;
pub mod random_projection_lsh;
pub mod sif;
pub mod snn;
pub mod vptree;
pub mod wrtree;

pub use atree::ATree;
pub use dim_reduction::LayeredLsh;
pub use dynamic_queries::DynamicQuery;
pub use measured_lsh::MeasuredLSH;

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
        // Box::new(hnsw::HNSWTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(atree::ATree::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(dim_reduction::LayeredLsh::<D>::new(embedding)) as Box<dyn IndexClone<D> + 'a>,
        Box::new(kiddo::Kiddo::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        // Box::new(nabo::Nabo::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(embedding.clone()) as Box<dyn IndexClone<D> + 'a>,
        Box::new(lsh::Lsh::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        // Box::new(random_projection_lsh::RandomProjectionLsh::<D>::new(embedding.clone()))
        // as Box<dyn IndexClone<D> + 'a>,
        Box::new(snn::SNN::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(kd_tree::KDTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(ball_tree::WBallTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(neighbourhood::Neihbourhood::<D>::new(embedding.clone()))
            as Box<dyn IndexClone<D> + 'a>,
        Box::new(sif::SIF::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
        Box::new(vptree::VPTree::<D>::new(embedding.clone())) as Box<dyn IndexClone<D> + 'a>,
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

    // #[cfg(any(feature = "cgal", feature = "boost-rtree", feature = "nanoflann"))]
    iter.collect::<Vec<_>>().into_iter()
}
