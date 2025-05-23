use uwuembed::{
    lsh::Lsh,
    query::{Embedder, Update},
    *,
};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    let dim = 8;
    let dim_hint = 8;

    let (graph, iterations) = load_graph(graph_name, dim, dim_hint)?;
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

    println!("Building Data structure");
    let mut lsh = Lsh::new(embeddings[35].clone());

    for embedding in embeddings.iter().skip(35) {
        println!("Updating positions");
        lsh.update_positions(&embedding.positions);
        println!("Query all nodes");
        for node in 0..embedding.positions.len() {
            lsh.repelling_nodes(node);
        }
    }

    Ok(())
}
