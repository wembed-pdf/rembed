use rembed::{embedder::EmbedderOptions, *};
use simulation::radius_reduction::{QueryParams, Statistics};

fn main() -> io::Result<()> {
    let dim = 8;
    let dim_hint = 8;

    // let graph = "../data/generated/graphs/19_girg_n-1000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../data/generated/graphs/55_girg_n-10000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    let graph = "../data/generated/graphs/109_girg_n-100000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../data/generated/graphs/1217_girg_n-1000000_deg-15_dim-4_ple-2.5_alpha-inf_wseed-12_pseed-130_sseed-1400";
    // let graph = "../data/generated/graphs/2263_girg_n-1000000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400";
    let graph = graph::Graph::parse_from_edge_list_file(graph, dim, dim_hint)?;

    println!(
        "D, normal, reduction_radius, reduction_snn, reduction_both, snn_best, both_best, snn_with_radius_reduction, snn_with_radius_reduction_best_dim, pruned_subtrees"
    );
    embedd_and_calc_stats::<2>(&graph);
    // embedd_and_calc_stats::<3>(&graph);
    embedd_and_calc_stats::<4>(&graph);
    // embedd_and_calc_stats::<5>(&graph);
    embedd_and_calc_stats::<6>(&graph);
    // embedd_and_calc_stats::<7>(&graph);
    embedd_and_calc_stats::<8>(&graph);
    // embedd_and_calc_stats::<9>(&graph);
    embedd_and_calc_stats::<10>(&graph);
    // embedd_and_calc_stats::<11>(&graph);
    embedd_and_calc_stats::<12>(&graph);
    // embedd_and_calc_stats::<13>(&graph);
    embedd_and_calc_stats::<14>(&graph);
    // embedd_and_calc_stats::<15>(&graph);
    embedd_and_calc_stats::<16>(&graph);
    // embedd_and_calc_stats::<17>(&graph);
    // embedd_and_calc_stats::<18>(&graph);
    // embedd_and_calc_stats::<19>(&graph);
    // embedd_and_calc_stats::<20>(&graph);
    // embedd_and_calc_stats::<21>(&graph);
    // embedd_and_calc_stats::<22>(&graph);
    // embedd_and_calc_stats::<23>(&graph);
    // embedd_and_calc_stats::<24>(&graph);

    Ok(())
}

fn embedd_and_calc_stats<const D: usize>(graph: &graph::Graph) {
    let options = EmbedderOptions {
        max_iterations: 500,
        ..Default::default()
    };
    // let mut embedder: embedder::WEmbedder<Sprk<_>, D> =
    //     embedder::WEmbedder::random(42, graph, options);
    let mut embedder = embedder::WEmbedder::<DynamicQuery<D, Sprk<D>>>::random(42, graph, options);

    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if i % 10 == 0 && i > 0 {
            // eprintln!("Iteration {i}");
        }
    });
    let pos: &[rembed::dvec::DVec<D>] = embedder.positions();

    let analysis = simulation::radius_reduction::DimReduction::new(pos.to_vec());

    let mut stats_normal = Statistics::default();
    let mut stats_reduced_radius = Statistics::default();
    let mut stats_snn = Statistics::default();
    let mut stats_both = Statistics::default();
    let mut stats_snn_best_dim = Statistics::default();
    let mut stats_both_best_dim = Statistics::default();
    let mut stats_snn_with_radius_reduction = Statistics::default();
    let mut stats_snn_with_radius_reduction_best_dim = Statistics::default();

    for i in 0..pos.len() {
        let mut results = Vec::new();
        let params = QueryParams::new(false, false, false, false, false);
        analysis.query(i, 1., &mut results, &mut stats_normal, &params);
        let params = QueryParams::new(true, false, false, false, false);
        analysis.query(i, 1., &mut results, &mut stats_reduced_radius, &params);
        let params = QueryParams::new(false, true, false, false, false);
        analysis.query(i, 1., &mut results, &mut stats_snn, &params);
        let params = QueryParams::new(true, true, false, false, false);
        analysis.query(i, 1., &mut results, &mut stats_both, &params);
        let params = QueryParams::new(false, true, true, false, false);
        analysis.query(i, 1., &mut results, &mut stats_snn_best_dim, &params);
        let params = QueryParams::new(true, true, true, false, false);
        analysis.query(i, 1., &mut results, &mut stats_both_best_dim, &params);
        let params = QueryParams::new(true, true, false, false, true);
        analysis.query(
            i,
            1.,
            &mut results,
            &mut stats_snn_with_radius_reduction,
            &params,
        );
        let params = QueryParams::new(true, true, true, false, true);
        analysis.query(
            i,
            1.,
            &mut results,
            &mut stats_snn_with_radius_reduction_best_dim,
            &params,
        );
    }
    println!(
        "{D}, {normal}, {reduction_radius}, {reduction_snn}, {reduction_both}, {snn_best}, {both_best}, {snn_with_radius_reduction}, {snn_with_radius_reduction_best_dim}, {pruned_trees}",
        D = D,
        normal = stats_normal.num_comparionsons,
        reduction_radius = stats_reduced_radius.num_comparionsons,
        reduction_snn = stats_snn.num_comparionsons,
        reduction_both = stats_both.num_comparionsons,
        snn_best = stats_snn_best_dim.num_comparionsons,
        both_best = stats_both_best_dim.num_comparionsons,
        snn_with_radius_reduction = stats_snn_with_radius_reduction.num_comparionsons,
        snn_with_radius_reduction_best_dim =
            stats_snn_with_radius_reduction_best_dim.num_comparionsons,
        pruned_trees = stats_normal.pruned_trees,
    );
}
