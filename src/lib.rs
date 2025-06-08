pub use embedding::Embedding;
use query::IndexClone;
pub use query::Query;
pub use std::io;

pub mod dvec;
pub mod embedding;
pub mod graph;
pub mod lsh;
pub mod parsing;
pub mod query;
pub mod snn;
pub mod wrtree;

pub type NodeId = usize;

pub fn load_graph(
    graph_name: &str,
    dim: usize,
    dim_hint: usize,
) -> Result<(graph::Graph, parsing::Iterations<8>), io::Error> {
    load_graph_from_path(&format!("data/{}/graph", graph_name), dim, dim_hint)
}
pub fn load_graph_from_path(
    graph_path: &str,
    dim: usize,
    dim_hint: usize,
) -> Result<(graph::Graph, parsing::Iterations<8>), io::Error> {
    let graph = graph::Graph::parse_from_edge_list_file(graph_path, dim, dim_hint)?;
    let positions_path = format!("data/{}/positions_{}_{}.log", graph_path, dim, dim_hint);
    let iterations = parsing::parse_positions_file(positions_path)?;
    Ok((graph, iterations))
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
