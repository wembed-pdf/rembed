use std::time::Instant;

use rembed::{embedder::EmbedderOptions, lossy_queries::LossyQuery, *};

fn main() -> io::Result<()> {
    // let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    const D: usize = 2;
    let dim_hint = 8;

    // let graph_path = format!("data/{}/graph", graph_name);
    let graph = "data/generated/graphs/1084_girg_n-1000_deg-25_dim-2_ple-2.5_alpha-inf_wseed-14_pseed-132_sseed-1402";
    // let graph = "data/generated/graphs/19_girg_n-1000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/55_girg_n-10000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/109_girg_n-100000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../praktikum-beating-the-worst-case-framework-rust-uwu/instances/graphs/609.gr";
    // let graph = "../praktikum-beating-the-worst-case-framework-rust-uwu/instances/graphs/72.gr";
    let graph = graph::Graph::parse_from_edge_list_file(&graph, D, dim_hint)?;

    let options = EmbedderOptions {
        max_iterations: 500,
        ..Default::default()
    };
    let embedder: embedder::WEmbedder<ATree<_>, D> =
        embedder::WEmbedder::random(42, &graph, options.clone());
    // let mut embedder: embedder::WEmbedder<DynamicQuery<_, ATree<_>>, D> =
    //     embedder::WEmbedder::random(42, &graph, options);
    let positions = embedder.positions().to_vec();
    let embedding = Embedding {
        positions,
        graph: &graph,
    };
    // let recall = 1.0;
    let recall = 0.7;
    // let strategy = rembed::lossy_queries::LossyStrategy::Random;
    let strategy = rembed::lossy_queries::LossyStrategy::InOrder;
    // let strategy = rembed::lossy_queries::LossyStrategy::Closest;
    // let strategy = rembed::lossy_queries::LossyStrategy::Furthest;
    // let strategy = rembed::lossy_queries::LossyStrategy::Heavy;
    // let strategy = rembed::lossy_queries::LossyStrategy::Droplist;
    // let strategy = rembed::lossy_queries::LossyStrategy::Light;
    // let lossy_queries = ATree::new(&embedding);
    let lossy_queries = LossyQuery::<_, ATree<_>>::new(&embedding, recall, strategy);
    let mut embedder = embedder::WEmbedder::new(lossy_queries, options);

    // takes wembed 03:04 for the first 100 iterations on rel8
    let mut last_time = Instant::now();
    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if i % 100 == 0 && i > 0 {
            eprintln!(
                "Iteration {i}, Δp{}, {:.1}s",
                e.last_pos_delta().unwrap_or_default(),
                last_time.elapsed().as_secs_f32(),
            );
            last_time = Instant::now();
        }
    });
    embedder.print_stats();

    Ok(())
}
