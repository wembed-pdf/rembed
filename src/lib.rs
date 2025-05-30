pub use embedding::Embedding;
pub use query::Query;
pub use std::io;

pub mod dvec;
pub mod embedding;
pub mod graph;
pub mod lsh;
pub mod parsing;
pub mod query;

type NodeId = usize;

pub fn load_graph(
    graph_name: &str,
    dim: usize,
    dim_hint: usize,
) -> Result<(graph::Graph, Vec<parsing::Iteration<8>>), io::Error> {
    let graph = graph::Graph::parse_from_edge_list_file(
        &format!("data/{}/graph", graph_name),
        dim,
        dim_hint,
    )?;
    let positions_path = format!("data/{}/positions_{}_{}.log", graph_name, dim, dim_hint);
    let iterations = parsing::parse_positions_file(positions_path)?;
    Ok((graph, iterations))
}
