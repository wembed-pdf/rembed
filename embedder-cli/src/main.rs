use std::io;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use rembed::{
    DynamicQuery, Embedding, Sprk,
    dyn_embed::{DynDynamicQuery, DynVec, EmbedIndex},
    dvec::{DVec, Vector},
    embedder::{EmbedderOptions, WEmbedder},
    graph,
};

// Re-use rand from rembed's re-export isn't available, so we use the
// random init in the dynamic path via DynVec::from_fn directly.

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "embedder-cli", about = "Graph embedding tool using force-directed layout")]
struct Args {
    /// Path to input graph edge list file
    #[arg(short, long)]
    input: String,

    /// Embedding dimension
    #[arg(short, long)]
    dim: usize,

    /// Latent dimension hint (defaults to dim)
    #[arg(long)]
    dim_hint: Option<usize>,

    /// Output path for positions CSV
    #[arg(short, long)]
    output: Option<String>,

    /// Print iteration progress every 100 steps
    #[arg(long)]
    progress: bool,

    /// When to compute and print F1 score
    #[arg(long, value_enum, default_value = "end")]
    f1_mode: F1Mode,

    /// Spatial index to use
    #[arg(long, value_enum, default_value = "sprk-dynamic")]
    index: IndexKind,

    /// Learning rate for Adam optimizer
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Cooling factor per iteration
    #[arg(long)]
    cooling_factor: Option<f64>,

    /// Maximum number of iterations
    #[arg(long)]
    max_iterations: Option<usize>,

    /// Minimum relative position change for convergence
    #[arg(long)]
    min_position_change: Option<f64>,

    /// Scale factor for attraction forces
    #[arg(long)]
    attraction_scale: Option<f64>,

    /// Scale factor for repulsion forces
    #[arg(long)]
    repulsion_scale: Option<f64>,

    /// Print per-iteration timing breakdown
    #[arg(long)]
    print_timings: bool,

    /// Random seed for initial positions
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[derive(Clone, ValueEnum)]
enum F1Mode {
    /// Never compute F1
    Never,
    /// Compute F1 only at the end
    End,
    /// Compute F1 every 100 iterations
    Every,
}

#[derive(Clone, ValueEnum)]
enum IndexKind {
    /// Sprk tree (direct, no caching)
    Sprk,
    /// Sprk tree with dynamic query caching (default)
    SprkDynamic,
    // Uncomment to enable additional indices (requires impl_embed_index! in rembed):
    // Kiddo,
    // VpTree,
    // Quadtree,
    // Grid,
    // BruteForce,
}

// ---------------------------------------------------------------------------
// Dimension dispatch
// ---------------------------------------------------------------------------

/// Dispatches runtime dimension to compile-time const generic (1..=16),
/// falling back to the dynamic embedder for larger dimensions.
macro_rules! dispatch_dim {
    ($dim:expr, $args:expr) => {
        match $dim {
            1 => run::<1>($args),
            2 => run::<2>($args),
            3 => run::<3>($args),
            4 => run::<4>($args),
            5 => run::<5>($args),
            6 => run::<6>($args),
            7 => run::<7>($args),
            8 => run::<8>($args),
            9 => run::<9>($args),
            10 => run::<10>($args),
            11 => run::<11>($args),
            12 => run::<12>($args),
            13 => run::<13>($args),
            14 => run::<14>($args),
            15 => run::<15>($args),
            16 => run::<16>($args),
            d => {
                eprintln!("Dimension {d} > 16: using dynamic (heap-allocated) embedder");
                run_dynamic($args)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> io::Result<()> {
    let args = Args::parse();

    dispatch_dim!(args.dim, &args)
}

// ---------------------------------------------------------------------------
// Const-generic path (D = 1..16)
// ---------------------------------------------------------------------------

fn run<const D: usize>(args: &Args) -> io::Result<()> {
    let dim_hint = args.dim_hint.unwrap_or(D);
    let graph = graph::Graph::parse_from_edge_list_file(&args.input, D, dim_hint)?;
    let options = build_options(args);

    eprintln!("n: {}, dim: {D}, dim_hint: {dim_hint}", graph.nodes.len());

    match args.index {
        IndexKind::SprkDynamic => {
            run_embed::<DynamicQuery<_, Sprk<_>>, D>(&graph, options, args)
        }
        IndexKind::Sprk => run_embed::<Sprk<_>, D>(&graph, options, args),
    }
}

fn run_embed<'a, SI, const D: usize>(
    graph: &'a graph::Graph,
    options: EmbedderOptions,
    args: &Args,
) -> io::Result<()>
where
    SI: rembed::query::Embedder<'a, D> + EmbedIndex<Vec = DVec<D>>,
{
    let mut embedder: WEmbedder<SI> = WEmbedder::random(args.seed, graph, options.clone());

    let start = Instant::now();
    let mut last_time = Instant::now();
    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if args.progress && i > 0 && i % 100 == 0 {
            eprintln!(
                "Iteration {i}, Δp={}, {:.1}s",
                e.last_pos_delta().unwrap_or_default(),
                last_time.elapsed().as_secs_f32(),
            );
            last_time = Instant::now();
        }
        if matches!(args.f1_mode, F1Mode::Every) && i > 0 && i % 100 == 0 {
            e.print_stats();
        }
    });
    eprintln!("Embedding took {:.2}s", start.elapsed().as_secs_f32());

    // F1 at end
    if !matches!(args.f1_mode, F1Mode::Never) {
        let final_embedding: Embedding<'_, D> = Embedding {
            positions: embedder.positions().to_vec(),
            graph,
        };
        let eval = WEmbedder::new(Sprk::new(&final_embedding), Default::default());
        eval.print_stats();
    }

    // Write output
    if let Some(ref path) = args.output {
        write_positions(embedder.positions(), path)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Dynamic path (D > 16)
// ---------------------------------------------------------------------------

fn run_dynamic(args: &Args) -> io::Result<()> {
    let dim = args.dim;
    let dim_hint = args.dim_hint.unwrap_or(dim);
    let graph = graph::Graph::parse_from_edge_list_file(&args.input, dim, dim_hint)?;
    let options = build_options(args);

    eprintln!(
        "n: {}, dim: {dim}, dim_hint: {dim_hint} (dynamic)",
        graph.nodes.len()
    );

    // Build random initial positions
    use rand::{Rng, SeedableRng, rngs::SmallRng};
    let n = graph.nodes.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(args.seed);
    let cube_side = (n as f64).powf(1.0 / dim as f64);
    let positions: Vec<DynVec> = (0..n)
        .map(|_| DynVec::from_fn(dim, |_| rng.random_range(0.0..cube_side) as f32))
        .collect();

    let spatial_index = DynDynamicQuery::new(dim, &positions, &graph);
    let mut embedder = WEmbedder::new(spatial_index, options);

    let start = Instant::now();
    let mut last_time = Instant::now();
    embedder.embed_with_callback(|e| {
        let i = e.iteration();
        if args.progress && i > 0 && i % 100 == 0 {
            eprintln!(
                "Iteration {i}, Δp={}, {:.1}s",
                e.last_pos_delta().unwrap_or_default(),
                last_time.elapsed().as_secs_f32(),
            );
            last_time = Instant::now();
        }
        if matches!(args.f1_mode, F1Mode::Every) && i > 0 && i % 100 == 0 {
            e.print_stats();
        }
    });
    eprintln!("Embedding took {:.2}s", start.elapsed().as_secs_f32());

    // F1 at end — use the same DynDynamicQuery for evaluation
    if !matches!(args.f1_mode, F1Mode::Never) {
        let eval_index = DynDynamicQuery::new(dim, embedder.positions(), &graph);
        let eval = WEmbedder::new(eval_index, Default::default());
        eval.print_stats();
    }

    // Write output
    if let Some(ref path) = args.output {
        write_dyn_positions(embedder.positions(), path)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_options(args: &Args) -> EmbedderOptions {
    let mut opts = EmbedderOptions::default();
    if let Some(v) = args.learning_rate {
        opts.learning_rate = v;
    }
    if let Some(v) = args.cooling_factor {
        opts.cooling_factor = v;
    }
    if let Some(v) = args.max_iterations {
        opts.max_iterations = v;
    }
    if let Some(v) = args.min_position_change {
        opts.min_position_change = v;
    }
    if let Some(v) = args.attraction_scale {
        opts.attraction_scale = v;
    }
    if let Some(v) = args.repulsion_scale {
        opts.repulsion_scale = v;
    }
    opts.print_timings = args.print_timings;
    opts
}

fn write_positions<const D: usize>(positions: &[DVec<D>], path: &str) -> io::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for pos in positions {
        for (i, &x) in pos.components.iter().enumerate() {
            if i > 0 {
                let _ = write!(&mut out, ", ");
            }
            let _ = write!(&mut out, "{x}");
        }
        let _ = writeln!(&mut out);
    }
    std::fs::write(path, out)
}

fn write_dyn_positions(positions: &[DynVec], path: &str) -> io::Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for pos in positions {
        for (i, &x) in pos.components.iter().enumerate() {
            if i > 0 {
                let _ = write!(&mut out, ", ");
            }
            let _ = write!(&mut out, "{x}");
        }
        let _ = writeln!(&mut out);
    }
    std::fs::write(path, out)
}
