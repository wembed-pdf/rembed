use rembed::{
    atree::ATree, dim_reduction::LayeredLsh, dynamic_queries::DynamicQuery,
    embedder::EmbedderOptions, query::Embedder, *,
};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    const D: usize = 3;
    // let dim = 8;
    let dim_hint = 8;

    // let graph_path = format!("data/{}/graph", graph_name);
    let graph_path = format!("data/{}/graph", graph_name);
    // let graph = "data/generated/graphs/1084_girg_n-1000_deg-25_dim-2_ple-2.5_alpha-inf_wseed-14_pseed-132_sseed-1402";
    // let graph = "data/generated/graphs/19_girg_n-1000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/55_girg_n-10000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/109_girg_n-100000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    let graph = "../praktikum-beating-the-worst-case-framework-rust-uwu/instances/graphs/609.gr";
    // let graph = "../praktikum-beating-the-worst-case-framework-rust-uwu/instances/graphs/72.gr";
    let graph = graph::Graph::parse_from_edge_list_file(&graph, D, dim_hint)?;

    let options = EmbedderOptions::default();
    // let mut embedder: embedder::WEmbedder<ATree<_>, D> =
    //     embedder::WEmbedder::random(42, &graph, options);
    let mut embedder: embedder::WEmbedder<DynamicQuery<_, ATree<_>>, D> =
        embedder::WEmbedder::random(42, &graph, options);

    // takes wembed 03:04 for the first 100 iterations on rel8
    // for i in 0..1000 {
    // embedder.calculate_step();
    // let mut last_cache = vec![vec![]; graph.nodes.len()];
    let mut last_positions = embedder.positions().to_vec();
    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if i % 10 == 0 && i > 0 {
            eprintln!(
                "Iteration {i}, Δp{}",
                e.last_pos_delta().unwrap_or_default()
            );
            // let pos = embedder.positions();
            // let embedding = Embedding {
            //     positions: pos.to_vec(),
            //     graph: &graph,
            // };
            // let (percision, recall) = embedder.spatial_index.graph_statistics();
            // let f1 = 2. / (recall.recip() + percision.recip());
            // println!("i: , percision: {percision}, recall: {recall}, f1: {f1}");
        }
        // let mut added = 0;
        // let mut removed = 0;
        // for (new, old) in e.query_cache().iter().zip(last_cache.iter()) {
        //     for node_id in new {
        //         if !old.contains(node_id) {
        //             added += 1;
        //         }
        //     }
        //     for node_id in old {
        //         if !new.contains(node_id) {
        //             removed += 1;
        //         }
        //     }
        // }

        if e.iteration() % 100 == 0 {
            // let mut dp = 0.;
            let mut distances = Vec::new();
            for (old, new) in last_positions.iter().zip(e.positions()) {
                // dp += old.distance(new) as f64;
                distances.push(old.distance(new) as f64);
            }
            // eprintln!("+ {added} - {removed}");
            // println!("{added} {removed} {dp}");
            distances.sort_unstable_by(|a, b| a.total_cmp(b).reverse());
            // println!("distances:  {:?}", &distances[..]);
            // for dist in distances
        }
        // last_cache = e.query_cache().to_vec();
        last_positions = e.positions().to_vec();
    });
    // }
    let pos = embedder.positions();
    let embedding = Embedding {
        positions: pos.to_vec(),
        graph: &graph,
    };
    // let (percision, recall) = embedding.graph_statistics();
    // let f1 = 2. / (recall.recip() + percision.recip());
    // eprintln!("i: , percision: {percision}, recall: {recall}, f1: {f1}");
    // for pos in embedding.positions {
    //     println!("{} {}", pos[0], pos[1]);
    // }

    Ok(())
}
