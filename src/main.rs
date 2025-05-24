use uwuembed::{
    lsh::Lsh,
    query::{Embedder, Graph, Position, Update},
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
        // lsh.update_positions(&embedding.positions);
        println!("Query all nodes");
        for node in 0..embedding.positions.len() {
            let weight = embedding.weight(node);
            if weight < 1. {
                let lsh_result = lsh.repelling_nodes(node);
                continue;
                let naive_result = embedding.repelling_nodes(node);

                for naive_node in naive_result {
                    let other_weight = embedding.weight(naive_node);
                    if other_weight >= 1. {
                        continue;
                    }
                    if !lsh_result.contains(&naive_node) {
                        let p1 = *embedding.position(naive_node) * 2.;
                        let p2 = *embedding.position(node) * 2.;
                        let p1_rounded = p1.map(|x| x.floor());
                        let p2_rounded = p2.map(|x| x.floor());
                        dbg!(
                            naive_node,
                            node,
                            p1,
                            p1_rounded.to_int_array(),
                            p2,
                            p2_rounded.to_int_array(),
                            p1.distance_squared(&p2)
                        );
                        assert_eq!(p1_rounded.to_int_array(), p2_rounded.to_int_array());
                        panic!("foo",)
                    }
                }
            }
        }
        break;
    }

    Ok(())
}
