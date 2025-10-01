use rembed::{dim_reduction::LayeredLsh, embedder::EmbedderOptions, query::Embedder, *};
use simulation::radius_reduction::Statistics;

fn main() -> io::Result<()> {
    // let dim = 8;
    // let dim_hint = 8;

    // let graph = "../data/generated/graphs/19_girg_n-1000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    let graph = "../data/generated/graphs/55_girg_n-10000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../data/generated/graphs/109_girg_n-100000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../data/generated/graphs/1217_girg_n-1000000_deg-15_dim-4_ple-2.5_alpha-inf_wseed-12_pseed-130_sseed-1400";

    let graph = rembed::graph::Graph::parse_from_edge_list_file(graph, 4, 4).unwrap();
    let options = EmbedderOptions {
        max_iterations: 1000,
        ..Default::default()
    };
    let mut embedder: embedder::WEmbedder<atree::ATree<_>, 32> =
        embedder::WEmbedder::random(42, &graph, options);

    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if i % 100 == 0 && i > 0 {
            eprintln!("Iteration {i}");
            // eprintln!("Iteration {i}");
        }
    });
    let pos = embedder.positions();
    let embedding = embedding::Embedding {
        positions: pos.to_vec(),
        graph: &graph,
    };
    let atree = atree::ATree::new(&embedding);
    println!(
        "num iterations: {}, f1: {}",
        embedder.iteration(),
        atree.f1()
    );
    let analysis = simulation::radius_reduction::DimReduction::new(pos.to_vec());
    let mut stats_reduced = Statistics::default();
    let mut stats_normal = Statistics::default();
    for i in 0..pos.len().min(10_000) {
        let mut results = Vec::new();
        analysis.query(i, 1., &mut results, &mut stats_normal, false);
        analysis.query(i, 1., &mut results, &mut stats_reduced, true);
    }
    eprintln!(
        "normal: {} distance calculations\nreduction: {} distance calculations\n percent: {:.2}%",
        stats_normal.num_comparionsons,
        stats_reduced.num_comparionsons,
        stats_reduced.num_comparionsons as f64 * 100. / stats_normal.num_comparionsons as f64
    );
    return Ok(());

    let graph = 109;

    // let graph = graph::Graph::parse_from_edge_list_file(graph, dim, dim_hint)?;

    embedd_and_calc_stats::<2>(graph);
    embedd_and_calc_stats::<3>(graph);
    embedd_and_calc_stats::<4>(graph);
    embedd_and_calc_stats::<5>(graph);
    embedd_and_calc_stats::<6>(graph);
    embedd_and_calc_stats::<7>(graph);
    embedd_and_calc_stats::<8>(graph);
    embedd_and_calc_stats::<9>(graph);
    embedd_and_calc_stats::<10>(graph);
    embedd_and_calc_stats::<11>(graph);
    embedd_and_calc_stats::<12>(graph);
    embedd_and_calc_stats::<13>(graph);
    embedd_and_calc_stats::<14>(graph);
    embedd_and_calc_stats::<15>(graph);
    embedd_and_calc_stats::<16>(graph);
    // embedd_and_calc_stats::<17>(graph);
    // embedd_and_calc_stats::<18>(graph);
    // embedd_and_calc_stats::<19>(graph);
    // embedd_and_calc_stats::<20>(graph);
    // embedd_and_calc_stats::<21>(graph);
    // embedd_and_calc_stats::<22>(graph);
    // embedd_and_calc_stats::<23>(graph);
    // embedd_and_calc_stats::<24>(graph);

    Ok(())
}

fn embedd_and_calc_stats<const D: usize>(graph_id: usize) {
    let dim = D;
    let path = format!(
        "../data/generated/positions/graph-{graph_id}_dim-{dim}_dim-hint-{dim}_seed-42.log"
    );
    dbg!(&path);
    let embedding: parsing::Iterations<D> = rembed::parsing::parse_positions_file(path).unwrap();

    let pos = embedding.iterations().last().unwrap().positions.to_vec();

    eprintln!("building tree");
    let analysis = simulation::radius_reduction::DimReduction::new(pos.to_vec());
    eprintln!("querying");

    let mut stats_normal = Statistics::default();
    let mut stats_reduced = Statistics::default();
    for i in 0..pos.len().min(10_000) {
        let mut results = Vec::new();
        analysis.query(i, 1., &mut results, &mut stats_normal, false);
        analysis.query(i, 1., &mut results, &mut stats_reduced, true);
    }
    eprintln!(
        "normal: {} distance calculations\nreduction: {} distance calculations\n percent: {:.2}%",
        stats_normal.num_comparionsons,
        stats_reduced.num_comparionsons,
        stats_reduced.num_comparionsons as f64 * 100. / stats_normal.num_comparionsons as f64
    );
    // dbg!(&stats_reduced);
    println!(
        "{}, {}, {}",
        D, stats_normal.num_comparionsons, stats_reduced.num_comparionsons,
    );
}
