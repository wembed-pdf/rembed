use rembed::{dim_reduction::LayeredLsh, embedder::EmbedderOptions, *};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    let dim = 8;
    let dim_hint = 8;
    let iterations =
        crate::parsing::parse_positions_file::<_, 8>("../../cpp/wembed/spatial_log").unwrap();

    println!("test");

    let graph_path = format!("data/{}/graph", graph_name);
    let graph = graph::Graph::parse_from_edge_list_file(&graph_path, dim, dim_hint)?;
    // let (graph, iterations) = load_graph(graph_name, dim, dim_hint)?;
    println!("Parsed {} iterations", iterations.iterations().len());
    let embeddings = || convert_to_embeddings(&iterations, &graph);

    // Print summary
    println!("Total of  {} nodes", graph.nodes.len());

    println!("Building Data structure");
    let embedding = &embeddings().next().unwrap();
    let lsh = LayeredLsh::new(embedding);
    // let lsh = Lsh::new(embedding.clone());
    // let lsh = rembed::wrtree::WRTree::new(embedding.clone());
    // let lsh = SNN::new(embedding.clone());
    // let lsh = SNN::new(embedding.clone());
    // let lsh = embedding.clone();

    let options = EmbedderOptions::default();
    dbg!(embeddings().count());
    let embeddings: Vec<_> = embeddings().collect();
    let mut embedder = embedder::WEmbedder::new(lsh, options);
    // takes wembed 03:04 for the first 100 iterations
    embedder.embed();
    for slice in embeddings.windows(2) {
        embedder.calculate_step();
        let mut msa = 0.;
        for ((res, actual), _prev) in embedder
            .positions()
            .iter()
            .zip(slice[1].positions.iter())
            .zip(slice[0].positions.iter())
        {
            // dbg!(res, actual, prev);
            let dist = res.distance(actual) / actual.magnitude();
            msa += dist;
        }
        println!("{}", msa / embedding.positions.len() as f32);
    }

    // for embedding in embeddings().skip(35) {
    //     println!("Updating positions");
    //     lsh.update_positions(&embedding.positions);
    //     println!("Query all nodes");
    //     for node in 0..embedding.positions.len() {
    //         let weight = embedding.weight(node);
    //         if weight < 1. {
    //             let lsh_result = lsh.repelling_nodes(node);
    //             // continue;
    //             let naive_result = embedding.repelling_nodes(node);

    //             for naive_node in naive_result {
    //                 let other_weight = embedding.weight(naive_node);
    //                 if other_weight >= 1. {
    //                     continue;
    //                 }
    //                 if !lsh_result.contains(&naive_node) {
    //                     let p1 = *embedding.position(naive_node) * 2.;
    //                     let p2 = *embedding.position(node) * 2.;
    //                     let p1_rounded = p1.map(|x| x.floor());
    //                     let p2_rounded = p2.map(|x| x.floor());
    //                     dbg!(
    //                         naive_node,
    //                         node,
    //                         p1,
    //                         p1_rounded.to_int_array(),
    //                         p2,
    //                         p2_rounded.to_int_array(),
    //                         p1.distance_squared(&p2)
    //                     );
    //                     assert_eq!(p1_rounded.to_int_array(), p2_rounded.to_int_array());
    //                     panic!("foo",)
    //                 }
    //             }
    //         }
    //     }
    // }

    Ok(())
}
