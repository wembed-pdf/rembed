use rembed::{
    Sprk, Embedding, MeasuredLSH,
    embedder::{EmbedderOptions, WEmbedder},
    graph, io,
    lossy_queries::{LossyQuery, LossyStrategy},
    query::Embedder as _,
    random_projection_lsh::RandomProjectionLsh,
};
use std::fmt::Write;
use std::time::Instant;

// ============================================================================
// CONFIGURATION
// ============================================================================

struct SweepConfig {
    // Graph and embedding settings
    graph_path: &'static str,
    embedding_dim: usize,
    dim_hint: usize,
    max_iterations: usize,

    // What to sweep
    sweep_lsh: bool,
    sweep_lossy: bool,

    // LSH sweep ranges
    lsh_num_tables_range: std::ops::RangeInclusive<usize>,
    lsh_num_projections_start: usize,
    lsh_num_projections_end: usize,
    lsh_num_projections_step: usize,

    // LossyQuery sweep settings
    lossy_p_values: Vec<f64>,
    lossy_strategies: Vec<LossyStrategy>,

    // Measurement options for LSH
    lsh_measure_recall: bool, // Requires MeasuredLSH wrapper (slower)
    lsh_measure_time: bool,   // Requires running without wrapper (for accuracy)
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            // Graph and embedding
            graph_path: "data/generated/graphs/73_girg_n-10000_deg-15_dim-4_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400",
            // graph_path;  "data/generated/graphs/109_girg_n-100000_deg-15_dim-2_ple-2.2_alpha-inf_wseed-12_pseed-130_sseed-1400",
            embedding_dim: 12,
            dim_hint: 12,
            max_iterations: 500,

            // What to sweep
            sweep_lsh: true,
            sweep_lossy: false,

            // LSH ranges
            lsh_num_tables_range: 1..=3,
            lsh_num_projections_start: 4,
            lsh_num_projections_end: 28,
            lsh_num_projections_step: 8,

            // LossyQuery settings
            lossy_p_values: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            lossy_strategies: vec![
                LossyStrategy::Random,
                LossyStrategy::InOrder,
                LossyStrategy::Closest,
                LossyStrategy::Furthest,
                LossyStrategy::Heavy,
                LossyStrategy::Light,
            ],

            // LSH measurement options
            // lsh_measure_recall: true,
            lsh_measure_recall: false,
            lsh_measure_time: true,
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> io::Result<()> {
    let config = SweepConfig::default();

    const D: usize = 8; // Must match config.embedding_dim

    // Load graph
    eprintln!("Loading graph from: {}", config.graph_path);
    let graph = graph::Graph::parse_from_edge_list_file(
        config.graph_path,
        config.embedding_dim,
        config.dim_hint,
    )?;

    let num_nodes = graph.nodes.len();
    eprintln!("Loaded graph with {} nodes", num_nodes);

    let options = EmbedderOptions {
        max_iterations: config.max_iterations,
        ..Default::default()
    };

    // Generate initial embedding with Sprk
    eprintln!("Generating initial embedding...");
    let embedder: WEmbedder<Sprk<_>, D> = WEmbedder::random(42, &graph, options.clone());
    let positions = embedder.positions().to_vec();
    let embedding = Embedding {
        positions,
        graph: &graph,
    };

    // CSV header with metadata columns
    let mut output =
        String::from("sweep_type,strategy,num_nodes,embedding_dim,p,f1,precision,recall,time_ms\n");

    // ========================================================================
    // LSH SWEEP
    // ========================================================================

    if config.sweep_lsh {
        eprintln!("\n=== Starting LSH parameter sweep ===");

        for num_tables in config.lsh_num_tables_range.clone() {
            for num_projections in (config.lsh_num_projections_start
                ..=config.lsh_num_projections_end)
                .step_by(config.lsh_num_projections_step)
            {
                let strategy_name = format!("LSH-L{}-K{}", num_tables, num_projections);
                eprintln!("Testing {}", strategy_name);

                // Measure recall if requested (requires MeasuredLSH wrapper)
                let p_value = if config.lsh_measure_recall {
                    let lsh = RandomProjectionLsh::<D>::new_with_params(
                        embedding.clone(),
                        Some(num_tables),
                        Some(num_projections),
                    );
                    let ground_truth = Sprk::<D>::new(&embedding);
                    let spatial_index = MeasuredLSH::new(lsh, ground_truth);
                    let mut embedder = WEmbedder::new(spatial_index, options.clone());
                    embedder.embed();
                    embedder.spatial_index.average_recall()
                } else {
                    0.0 // Not measured
                };

                // Measure time if requested (run without wrapper for accuracy)
                let time_ms = if config.lsh_measure_time {
                    let lsh = RandomProjectionLsh::<D>::new_with_params(
                        embedding.clone(),
                        Some(num_tables),
                        Some(num_projections),
                    );
                    let mut embedder = WEmbedder::new(lsh, options.clone());

                    let start = Instant::now();
                    embedder.embed();
                    start.elapsed().as_millis() as f64
                } else {
                    0.0 // Not measured
                };

                // Always compute final F1 with Sprk (non-approximate ground truth)
                let lsh = RandomProjectionLsh::<D>::new_with_params(
                    embedding.clone(),
                    Some(num_tables),
                    Some(num_projections),
                );
                let mut embedder = WEmbedder::new(lsh, options.clone());
                embedder.embed();

                // Get final embedding positions and compute F1 with Sprk
                let final_positions = embedder.positions().to_vec();
                let final_embedding = Embedding {
                    positions: final_positions,
                    graph: &graph,
                };
                let ground_truth = Sprk::<D>::new(&final_embedding);
                let (precision, recall_final) = ground_truth.graph_statistics();
                let f1 = 2. / (recall_final.recip() + precision.recip());

                // Output: sweep_type, strategy, num_nodes, embedding_dim, p, f1, precision, recall, time_ms
                writeln!(
                    &mut output,
                    "LSH,{},{},{},{},{},{},{},{}",
                    strategy_name,
                    num_nodes,
                    config.embedding_dim,
                    p_value,
                    f1,
                    precision,
                    recall_final,
                    time_ms
                )
                .unwrap();
            }
        }

        eprintln!("LSH sweep complete!");
    }

    // ========================================================================
    // LOSSY QUERY SWEEP
    // ========================================================================

    if config.sweep_lossy {
        eprintln!("\n=== Starting LossyQuery+Sprk strategy sweep ===");

        for &strategy in &config.lossy_strategies {
            for &p_target in &config.lossy_p_values {
                eprintln!("Testing strategy: {:?}, p={}", strategy, p_target);

                // Run embedding with LossyQuery
                let spatial_index = LossyQuery::<_, Sprk<D>>::new(&embedding, p_target, strategy);
                let mut embedder = WEmbedder::new(spatial_index, options.clone());

                let start = Instant::now();
                embedder.embed();
                let time_ms = start.elapsed().as_millis() as f64;

                // Get final embedding positions and compute F1 with fresh Sprk
                let final_positions = embedder.positions().to_vec();
                let final_embedding = Embedding {
                    positions: final_positions,
                    graph: &graph,
                };
                let ground_truth = Sprk::<D>::new(&final_embedding);
                let (precision, recall) = ground_truth.graph_statistics();
                let f1 = 2. / (recall.recip() + precision.recip());

                // Output: sweep_type, strategy, num_nodes, embedding_dim, p, f1, precision, recall, time_ms
                writeln!(
                    &mut output,
                    "LossyQuery,{:?},{},{},{},{},{},{},{}",
                    strategy,
                    num_nodes,
                    config.embedding_dim,
                    p_target,
                    f1,
                    precision,
                    recall,
                    time_ms
                )
                .unwrap();
            }
        }

        eprintln!("LossyQuery sweep complete!");
    }

    // ========================================================================
    // OUTPUT
    // ========================================================================

    println!("{}", output);

    let total_configs = (if config.sweep_lsh {
        let num_tables_count =
            config.lsh_num_tables_range.end() - config.lsh_num_tables_range.start() + 1;
        let num_projections_count = (config.lsh_num_projections_end
            - config.lsh_num_projections_start)
            / config.lsh_num_projections_step
            + 1;
        num_tables_count * num_projections_count
    } else {
        0
    }) + (if config.sweep_lossy {
        config.lossy_strategies.len() * config.lossy_p_values.len()
    } else {
        0
    });

    eprintln!("\n=== Sweep Summary ===");
    eprintln!("Total configurations tested: {}", total_configs);
    if config.sweep_lsh {
        eprintln!(
            "LSH: recall measured = {}, time measured = {}",
            config.lsh_measure_recall, config.lsh_measure_time
        );
    }

    Ok(())
}
