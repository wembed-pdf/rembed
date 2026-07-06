use benchmark::benchmark::LoadData;
use benchmark::benchmark::runner::BenchmarkType;
use benchmark::correctness_test::CorrectnessTestManager;
use clap::{Parser, Subcommand};
use dotenv::dotenv;
use sqlx::PgPool;
use std::env;
use std::str::FromStr;

use benchmark::generate_positions::PositionGenerator;
use benchmark::job_manager::JobManager;
use benchmark::{GraphGenerator, push_files};

#[derive(Parser)]
#[command(name = "benchmark")]
#[command(about = "Graph generation and position embedding benchmark tool")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pull files from remote directory
    Pull {
        /// Just pull this graph_id
        #[arg(long)]
        graph_id: Option<i64>,
        /// Just pull this result_id
        #[arg(long)]
        result_id: Option<i64>,
    },
    /// Push files to remote directory
    Push,
    /// Run Benchmarks matching the specified parameters
    Bench {
        /// Only run the last iteration of the benchmark (default: false)
        #[arg(long, default_value_t = false)]
        only_last_iteration: bool,
        /// Range graph sizes (node count) to benchmark (e.g. "100-1000")
        #[arg(long, short)]
        n: Option<String>,
        /// Specific graph sizes to select (e.g. "100,200,300")
        #[arg(long, value_delimiter = ',')]
        n_selection: Option<Vec<String>>,
        /// Range of dimensionality of the embeddings (e.g. "8-9")
        #[arg(long)]
        dim: Option<String>,
        /// Specific dimensions to select (e.g. "8,16,32")
        #[arg(long, value_delimiter = ',')]
        dim_selection: Option<Vec<String>>,
        /// Range of sampling dimensions of the original graph (e.g. "2-4")
        #[arg(long)]
        graph_dim: Option<String>,
        /// Range of average node degrees (e.g. "10-100")
        #[arg(long)]
        deg: Option<String>,
        /// Range of PLE values (e.g. "0.1-0.9")
        #[arg(long)]
        ple: Option<String>,
        /// Specific PLE values to select (e.g. "2.2,2.5,2.8")
        #[arg(long, value_delimiter = ',')]
        ple_selection: Option<Vec<String>>,
        /// Range of alpha values (e.g. "0.1-0.9")
        #[arg(long)]
        alpha: Option<String>,
        /// Seed that was used in the embedding
        #[arg(long)]
        seed: Option<String>,
        /// Seed that was used to generate the graph
        #[arg(long)]
        wseed: Option<String>,
        /// Number of threads to use for the benchmark (default: 1) (0 for all available)
        #[arg(long, default_value_t = 1)]
        n_threads: usize,
        /// Store the results of this benchmark run to the database
        #[arg(long)]
        store: bool,
        /// Circumvent the repository dirtyness check for storing results. Use with caution
        #[arg(long)]
        allow_dirty: bool,
        /// List of benchmarks to run
        #[arg(long)]
        benchmarks: Option<Vec<String>>,
        /// List of datastructures to bench
        #[arg(long)]
        structures: Option<Vec<String>>,
        /// Skip running of unit tests
        #[arg(long, default_value_t = false)]
        skip_test: bool,
        /// Alias for skipping tests (for convenience)
        #[arg(long, default_value_t = false)]
        skip_tests: bool,
        /// Enable dynamic downloading of graphs and positions during benchmarking (instead of requiring a prior pull)
        #[arg(long, default_value_t = false)]
        dynamic_download: bool,
        /// Set benchmark to fast mode with shorter warmup and measurement times (for quick local testing)
        #[arg(long, default_value_t = false)]
        fast: bool,
        /// Export datasets instead of running the benchmarks
        #[arg(long, default_value_t = false)]
        export_only: bool,
    },
    /// Generate graphs using GIRGs
    GenerateGraphs,

    /// Generate position embeddings (daemon mode)
    GeneratePositions,

    /// Compute F-Scores for position embeddings
    FScores {
        /// Only compute fscore for the last iteration of each result (default: false)
        #[arg(long, default_value_t = false)]
        only_last_iteration: bool,
        /// Compute F-Scores for a specific result ID
        #[arg(long)]
        result_id: Option<i64>,
    },

    /// Show job queue status
    Status {
        /// Show detailed status information
        #[arg(short, action)]
        v: bool,
    },

    /// Create position generation jobs for a graph
    CreateJobs {
        /// Graph ID to create jobs for
        graph_id: i64,
    },

    /// Create missing position jobs for all graphs
    CreateMissingJobs,

    /// Clean up stale jobs
    Cleanup {
        /// Timeout in hours for stale jobs (default: 2)
        #[arg(long, default_value = "2")]
        timeout_hours: i32,
        /// Flag to clean up failed jobs
        #[arg(long, action)]
        failed: bool,
    },

    /// Clean up orphaned files not referenced in database
    CleanupFiles {
        /// Perform dry run without actually deleting files
        #[arg(long)]
        dry_run: bool,
    },

    /// Generate correctness test file for a specific result
    GenerateTest {
        /// Result ID to generate test for
        result_id: i64,
    },

    /// Compute Missing Intrinsic Dimensions
    Intrinsic {
        /// Only compute intrinsic dimensions for the last iteration of each result (default: false)
        #[arg(long)]
        only_last_iteration: bool,
    },

    /// Run correctness tests (quick by default, extensive with options)
    Test {
        /// Test all iterations instead of just the last one
        #[arg(long)]
        all_iterations: bool,
        /// Test all graphs instead of just small ones
        #[arg(long)]
        all_graphs: bool,
        /// Filter by specific result ID
        #[arg(long)]
        result_id: Option<i64>,
        /// Filter by specific graph ID
        #[arg(long)]
        graph_id: Option<i64>,
        /// Filter by embedding dimension
        #[arg(long)]
        dim: Option<i32>,
        /// Also run unit tests from main crate
        #[arg(long)]
        run_unit_tests: bool,
        /// List of datastructures to bench (default: all)
        #[arg(long)]
        structures: Option<Vec<String>>,
        /// Enable dynamic downloading of graphs and positions during benchmarking (instead of requiring a prior pull)
        #[arg(long, default_value_t = false)]
        dynamic_download: bool,
        /// Also check for over-queried nodes (nodes returned by the structure but not in the ground truth)
        #[arg(long, default_value_t = false)]
        check_over_query: bool,
    },

    /// Benchmark data structures with synthetic distributions
    BenchDistributions {
        /// Dimensions to test (range format: "2-16" or single value "8")
        #[arg(long)]
        dimensions: Option<String>,

        /// Node counts to test (range format: "1000-10000" or single value)
        #[arg(long)]
        node_counts: Option<String>,

        /// Radiuses to test (range format: "0.5-2.0" or single value)
        #[arg(long)]
        radiuses: Option<String>,

        /// Point distributions: "normal", "uniform", or both (comma-separated)
        #[arg(long)]
        distributions: Option<String>,

        /// Benchmarkset names (optional)
        #[arg(long)]
        benchmarksets: Option<String>,

        /// Queryset names (optional, only relevant if benchmarksets are specified)
        #[arg(long)]
        querysets: Option<String>,

        /// Per-query radius file names, parallel to querysets (one radius per line).
        /// When specified, these replace --radiuses for the corresponding benchmarkset.
        #[arg(long)]
        query_radii: Option<String>,

        /// Path to benchmarksets (optional)
        #[arg(long)]
        benchmarksets_path: Option<String>,

        /// Filter to specific data structures (optional)
        #[arg(long)]
        structures: Vec<String>,

        /// Number of query points to sample per configuration
        #[arg(long, default_value = "1000")]
        num_queries: usize,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Output file path (optional, defaults to stdout with pretty print)
        #[arg(long, short)]
        output: Option<String>,

        /// Set benchmark to fast mode with shorter warmup and measurement times (for quick local testing)
        #[arg(long, default_value_t = false)]
        fast: bool,

        /// Expected number of queried nodes. This is mutually exclusive with radius and is used to set the radius such that on average this many nodes are within the radius.
        #[arg(long)]
        expected_queries: Option<usize>,

        /// Choose only nodes that are not closer than the radius to the boundary as query points to avoid edge effects
        #[arg(long, default_value_t = false)]
        only_center_nodes: bool,

        /// Execute all dimension/node count combinations in parallel using rayon (only recommended if dimensions * node_counts << num_cpus)
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Execute All-To-All benchmarks for the input benchmarksets
        #[arg(long, default_value_t = false)]
        all_to_all: bool,

        /// Maximum number of query points to use. If the queryset exceeds this, every k-th point is selected to reach this count.
        #[arg(long)]
        max_query_points: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let args = Args::parse();

    match args.command {
        Commands::Pull {
            graph_id,
            result_id,
        } => {
            benchmark::pull_files(false, None, graph_id, result_id).await?;
        }

        Commands::Push => {
            benchmark::push_files().await?;
        }

        Commands::Bench {
            only_last_iteration,
            n,
            n_selection,
            dim,
            dim_selection,
            graph_dim,
            deg,
            ple,
            ple_selection,
            alpha,
            seed,
            wseed,
            n_threads,
            store,
            benchmarks,
            structures,
            allow_dirty,
            skip_test,
            skip_tests,
            dynamic_download,
            fast,
            export_only,
        } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            if !(skip_test || skip_tests) {
                let test_manager = CorrectnessTestManager::new(pool.clone());
                test_manager
                    .run_tests(
                        false,
                        false,
                        None,
                        None,
                        None,
                        true,
                        Vec::new(),
                        dynamic_download,
                        false,
                    )
                    .await?;
            }

            let n_range = match n {
                Some(range) => parse_usize_range(&range).map_err(|e| e.to_string())?,
                None => (0, 0),
            };

            let dim_range = match dim {
                Some(range) => parse_usize_range(&range).map_err(|e| e.to_string())?,
                None => (0, 0),
            };

            let graph_dim_range = match graph_dim {
                Some(range) => parse_usize_range(&range).map_err(|e| e.to_string())?,
                None => (0, 0),
            };

            let deg_range = match deg {
                Some(range) => parse_usize_range(&range).map_err(|e| e.to_string())?,
                None => (0, 0),
            };

            let ple_range = match ple {
                Some(range) => parse_f64_range(&range).map_err(|e| e.to_string())?,
                None => (0.0, 0.0),
            };

            let alpha_range = match alpha {
                Some(range) => parse_f64_range(&range).map_err(|e| e.to_string())?,
                None => (0.0, 0.0),
            };

            let seed_range = match seed {
                Some(range) => parse_usize_range(&range).map_err(|e| e.to_string())?,
                None => (0, 0),
            };

            let wseed_range = match wseed {
                Some(range) => parse_usize_range(&range).map_err(|e| e.to_string())?,
                None => (0, 0),
            };

            let mut load_data = LoadData::new(pool);
            load_data.store = store;
            load_data.allow_dirty = allow_dirty;

            let benchmarks: Option<Vec<_>> = benchmarks.map(|x| {
                x.iter()
                    .map(|benchmark| BenchmarkType::from_str(benchmark).unwrap())
                    .collect()
            });

            load_data
                .run_benchmarks(
                    only_last_iteration,
                    n_range,
                    n_selection,
                    dim_range,
                    dim_selection,
                    graph_dim_range,
                    deg_range,
                    ple_range,
                    ple_selection,
                    alpha_range,
                    seed_range,
                    wseed_range,
                    n_threads,
                    benchmarks,
                    structures,
                    dynamic_download,
                    fast,
                    export_only,
                )
                .await?;
        }

        Commands::GenerateGraphs => {
            let generator = GraphGenerator::new(
                env::var("GIRGS_PATH").unwrap_or("../../girgs/build/genhrg".to_string()),
                env::var("DATA_DIRECTORY").unwrap_or("../data/".to_string()),
            );
            generator.generate().await?;
        }

        Commands::Intrinsic {
            only_last_iteration,
        } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            benchmark::intrinsic_dim::compute_missing_intrinsic_dimensions(
                pool,
                only_last_iteration,
            )
            .await?;
        }

        Commands::GeneratePositions => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;
            let job_manager = JobManager::new(pool);

            let generator = PositionGenerator::new(
                env::var("WEMBED_PATH")
                    .unwrap_or("../../wembed/release/bin/cli_wembed".to_string()),
                env::var("DATA_DIRECTORY").unwrap_or("../data/".to_string()),
                job_manager,
            );

            generator.run_daemon().await?;
        }

        Commands::FScores {
            only_last_iteration,
            result_id,
        } => {
            if let Some(result_id) = result_id {
                let statistic_generator = benchmark::statistics::StatisticGenerator::new();
                statistic_generator.compute_f_scores(result_id).await?;
                println!("F-Scores computed for result ID: {}", result_id);
            } else {
                let database_url = env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
                let pool = PgPool::connect(&database_url).await?;

                benchmark::fscore::compute_missing_fscores(pool, only_last_iteration).await?;
                println!("F-Scores computed for all missing entries");
            }
        }

        Commands::Status { v } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;
            let job_manager = JobManager::new(pool);

            let generator = PositionGenerator::new(
                String::new(), // Not needed for status
                String::new(), // Not needed for status
                job_manager,
            );

            generator.show_summary_status().await?;

            if v {
                generator.show_detailed_status().await?;
            }
        }

        Commands::CreateJobs { graph_id } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;
            let job_manager = JobManager::new(pool);

            let created = job_manager.create_jobs_for_graph(graph_id).await?;
            println!("Created {} jobs for graph {}", created, graph_id);
        }

        Commands::CreateMissingJobs => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;
            let job_manager = JobManager::new(pool);

            println!("Creating missing position jobs for all graphs...");
            let created = job_manager.create_missing_jobs().await?;
            println!("Created {} new jobs across all graphs", created);
        }

        Commands::Cleanup {
            timeout_hours,
            failed,
        } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            if failed {
                let cleaned = sqlx::query_scalar!(
                    "UPDATE position_jobs 
                     SET 
                        status = 'pending', 
                        claimed_at = NULL, 
                        claimed_by_hostname = NULL, 
                        error_message = COALESCE(error_message, '') || ' [Reset due to timeout]'
                     WHERE status = 'failed' AND claimed_at < NOW() - INTERVAL '1 hour' * $1 RETURNING 1",
                    timeout_hours as i32
                )
                .fetch_all(&pool)
                .await?;
                println!("Cleaned up {} failed jobs", cleaned.len());
            } else {
                let cleaned = sqlx::query_scalar!("SELECT cleanup_stale_jobs($1)", timeout_hours)
                    .fetch_one(&pool)
                    .await?;
                println!("Cleaned up {} stale jobs", cleaned.unwrap_or(0));
            }
        }

        Commands::CleanupFiles { dry_run } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            benchmark::cleanup::cleanup_orphaned_files(&pool, dry_run).await?;
        }

        Commands::GenerateTest { result_id } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            let test_manager = CorrectnessTestManager::new(pool);
            test_manager.generate_test(result_id).await?;
            push_files().await?;
        }

        Commands::Test {
            all_iterations,
            all_graphs,
            result_id,
            graph_id,
            dim,
            run_unit_tests,
            structures,
            dynamic_download,
            check_over_query,
        } => {
            // pull_files().await?;
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            let test_manager = CorrectnessTestManager::new(pool);
            test_manager
                .run_tests(
                    all_iterations,
                    all_graphs,
                    result_id,
                    graph_id,
                    dim,
                    run_unit_tests,
                    structures.unwrap_or_default(),
                    dynamic_download,
                    check_over_query,
                )
                .await?;
        }

        Commands::BenchDistributions {
            dimensions,
            node_counts,
            radiuses,
            distributions,
            benchmarksets,
            benchmarksets_path,
            querysets,
            query_radii,
            structures,
            num_queries,
            seed,
            output,
            fast,
            expected_queries,
            only_center_nodes,
            parallel,
            all_to_all,
            max_query_points,
        } => {
            use benchmark::benchmark::distribution_bench::{
                DistributionBenchConfig, DistributionBenchRunner,
            };

            let mut config = DistributionBenchConfig {
                dims: Vec::new(),
                counts: Vec::new(),
                radii: Vec::new(),
                distributions: None,
                structures: structures
                    .into_iter()
                    .flat_map(|s| {
                        s.split(',')
                            .map(|s| s.trim().to_string())
                            .collect::<Vec<String>>()
                    })
                    .collect(),
                num_queries,
                seed,
                benchmarksets: None,
                querysets: None,
                query_radii_sets: None,
                path_to_benchmarksets: benchmarksets_path.clone(),
                fast,
                expected_queries,
                only_center_nodes,
                parallel,
                all_to_all,
                max_query_points,
            };

            if let (Some(dimensions), Some(node_counts)) = (dimensions, node_counts) {
                config.dims = dimensions
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<usize>, _>>()?;
                config.counts = node_counts
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<usize>, _>>()?;
            }
            if let Some(expected) = expected_queries {
                assert!(
                    radiuses.is_none(),
                    "expected_queries is mutually exclusive with radiuses"
                );
                assert!(
                    query_radii.is_none(),
                    "expected_queries is mutually exclusive with query_radii"
                );
                // If expected_queries is provided, calculate the radius for each node count to achieve that many expected queries
                config.expected_queries = Some(expected);
            } else if query_radii.is_some() {
                assert!(
                    radiuses.is_none(),
                    "query_radii is mutually exclusive with radiuses"
                );
            } else {
                assert!(
                    radiuses.is_some(),
                    "Either expected_queries, radiuses, or query_radii must be provided"
                );
                config.radii = radiuses
                    .unwrap()
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<Vec<f64>, _>>()?;
            }

            // Parse distributions
            if let Some(distributions) = distributions {
                config.distributions = Some(parse_distributions(distributions.as_str())?);
            }

            // Parse benchmarksets
            if let Some(ref benchmarksets) = benchmarksets {
                assert!(
                    benchmarksets_path.is_some(),
                    "Benchmarksets path must be provided when benchmarksets are specified"
                );
                config.benchmarksets = Some(parse_benchmarksets(benchmarksets)?);
            }

            if let Some(ref querysets) = querysets {
                assert!(
                    benchmarksets.is_some(),
                    "Queryset is only relevant if benchmarksets are specified"
                );
                config.querysets = Some(parse_benchmarksets(querysets)?);
            }

            if let Some(ref query_radii) = query_radii {
                assert!(
                    benchmarksets.is_some(),
                    "query_radii is only relevant if benchmarksets are specified"
                );
                config.query_radii_sets = Some(parse_benchmarksets(query_radii)?);
            }

            let runner = DistributionBenchRunner::new(config);
            let results = runner.run()?;

            // Write output
            runner.write_output(&results, output.as_deref())?;
        }
    }

    Ok(())
}

fn parse_usize_range(s: &str) -> Result<(usize, usize), String> {
    let s = s.replace("_", ""); // Remove underscores for easier parsing

    if !s.contains('-') {
        // If no dash, treat as a single value range
        let value = s
            .parse::<usize>()
            .map_err(|_| "Invalid value".to_string())?;
        return Ok((value, value));
    }

    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        return Err("Range must be in format start-end".into());
    }

    let start = parts[0]
        .parse::<usize>()
        .map_err(|_| "Invalid start of range")?;
    let end = parts[1]
        .parse::<usize>()
        .map_err(|_| "Invalid end of range")?;

    Ok((start, end))
}

fn parse_f64_range(s: &str) -> Result<(f64, f64), String> {
    let s = s.replace("_", ""); // Remove underscores for easier parsing

    if !s.contains('-') {
        // If no dash, treat as a single value range
        let value = s.parse::<f64>().map_err(|_| "Invalid value".to_string())?;
        return Ok((value, value));
    }

    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        return Err("Range must be in format start-end".into());
    }

    let start = parts[0]
        .parse::<f64>()
        .map_err(|_| "Invalid start of range")?;
    let end = parts[1]
        .parse::<f64>()
        .map_err(|_| "Invalid end of range")?;

    Ok((start, end))
}

fn parse_distributions(
    s: &str,
) -> Result<Vec<benchmark::synthetic_data::PointDistribution>, String> {
    s.split(',')
        .map(|s| s.trim())
        .map(|s| match s {
            "normal" => Ok(benchmark::synthetic_data::PointDistribution::standard_normal()),
            "uniform" => Ok(benchmark::synthetic_data::PointDistribution::unit_uniform()),
            _ => Err(format!(
                "Invalid distribution '{}'. Must be 'normal' or 'uniform'",
                s
            )),
        })
        .collect()
}

fn parse_benchmarksets(s: &str) -> Result<Vec<String>, String> {
    Ok(s.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}
