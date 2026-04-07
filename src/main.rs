use std::time::Instant;

use rembed::{Sprk, embedder::EmbedderOptions, query::Embedder as _, *};

fn main() -> io::Result<()> {
    let graph_name = "rel8";
    // let graph_name = "bio-grid-fruitfly";

    const D: usize = 2;
    let dim_hint = D;

    let _graph = &format!("data/{}/graph", graph_name);
    // let graph = "data/generated/graphs/1084_girg_n-1000_deg-25_dim-2_ple-2.5_alpha-inf_wseed-14_pseed-132_sseed-1402";
    // let graph = "data/generated/graphs/19_girg_n-1000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/55_girg_n-10000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    let graph = "data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "data/generated/graphs/109_girg_n-100000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../praktikum-beating-the-worst-case-framework-rust-uwu/instances/graphs/609.gr";
    // let graph = "../praktikum-beating-the-worst-case-framework-rust-uwu/instances/graphs/72.gr";
    // let graph = "../../cpp/sample/edge_lists_real/web-spam";
    // let graph = "../../cpp/sample/edge_lists_real/rel7";
    // let graph = "../../cpp/sample/edge_lists_real/rel6";
    // let graph = "soc-LiveJournal1.edges";
    let graph = graph::Graph::parse_from_edge_list_file(graph, D, dim_hint)?;

    let count = graph.nodes.len();
    eprintln!("n: {count}");

    let options = EmbedderOptions {
        // max_iterations: 500,
        ..Default::default()
    };
    let embedder: embedder::WEmbedder<Sprk<_>, D> =
        embedder::WEmbedder::random(42, &graph, options.clone());
    // let mut embedder: embedder::WEmbedder<DynamicQuery<_, Sprk<_>>, D> =
    //     embedder::WEmbedder::random(42, &graph, options);
    let positions = embedder.positions().to_vec();
    let embedding = Embedding {
        positions,
        graph: &graph,
    };

    // let lossy_queries = Sprk::new(&embedding);
    // let lossy_queries = Kiddo::new(embedding.clone());
    // let lossy_queries = embedding.clone();
    // let lossy_queries =
    //     RandomProjectionLsh::<_>::new_with_params(embedding.clone(), Some(1), Some(16));
    let lossy_queries = DynamicQuery::<_, Sprk<_>>::new(&embedding);
    let mut embedder = embedder::WEmbedder::new(lossy_queries, options);

    let start = Instant::now();
    // takes wembed 03:04 for the first 100 iterations on rel8
    let mut last_time = Instant::now();
    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if i % 10 == 0 && i > 0 {
            eprintln!(
                "Iteration {i}, Δp{}, {:.1}s",
                e.last_pos_delta().unwrap_or_default(),
                last_time.elapsed().as_secs_f32(),
            );
            last_time = Instant::now();
        }
    });
    eprintln!("Embedding took {:.2}s", start.elapsed().as_secs_f32());

    let embedder = WEmbedder::new(
        Sprk::new(&Embedding {
            positions: embedder.positions().to_vec(),
            graph: &graph,
        }),
        Default::default(),
    );
    embedder.print_stats();

    use std::fmt::Write as _;
    let mut out_pos = String::new();
    for pos in embedder.positions() {
        for x in pos.components {
            let _ = write!(&mut out_pos, "{}, ", x);
        }
        let _ = writeln!(&mut out_pos);
    }
    let _ = std::fs::write("pos", out_pos);

    Ok(())
}
