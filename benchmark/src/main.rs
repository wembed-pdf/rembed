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
    Pull,
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
        /// Range of dimensionality of the embeddings (e.g. "8-9")
        #[arg(long)]
        dim: Option<String>,
        /// Range of average node degrees (e.g. "10-100")
        #[arg(long)]
        deg: Option<String>,
        /// Range of PLE values (e.g. "0.1-0.9")
        #[arg(long)]
        ple: Option<String>,
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
        #[arg(long)]
        skip_test: bool,
    },
    /// Generate graphs using GIRGs
    GenerateGraphs,

    /// Generate position embeddings (daemon mode)
    GeneratePositions,

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
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let args = Args::parse();

    match args.command {
        Commands::Pull => {
            benchmark::pull_files().await?;
        }

        Commands::Push => {
            benchmark::push_files().await?;
        }

        Commands::Bench {
            only_last_iteration,
            n,
            dim,
            deg,
            ple,
            alpha,
            seed,
            wseed,
            n_threads,
            store,
            benchmarks,
            structures,
            allow_dirty,
            skip_test,
        } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            if !skip_test {
                let test_manager = CorrectnessTestManager::new(pool.clone());
                test_manager
                    .run_tests(false, false, Some(1), None, None, true, Vec::new())
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
                    dim_range,
                    deg_range,
                    ple_range,
                    alpha_range,
                    seed_range,
                    wseed_range,
                    n_threads,
                    benchmarks,
                    structures,
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
                )
                .await?;
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
