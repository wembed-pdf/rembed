use uwuembed::{query::Embedder, *};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";
    let dim = 8;
    let dim_hint = 8;
    let graph = graph::Graph::parse_from_edge_list_file(
        &format!("data/{}/graph", graph_name),
        dim,
        dim_hint,
    )?;

    // Parse the positions file

    let positions_path = format!("data/{}/positions_{}_{}.log", graph_name, dim, dim_hint);

    let iterations = parsing::parse_positions_file(positions_path)?;
    let embeddings: Vec<Embedding<8>> = iterations
        .iter()
        .map(|x| Embedding {
            positions: x.coordinates().collect(),
            graph: &graph,
        })
        .collect();

    // Print summary
    println!("Parsed {} iterations", iterations.len());
    println!("Total of  {} nodes", graph.nodes.len());

    for embedding in &embeddings {
        embedding.repelling_nodes(0);
    }

    Ok(())
}
