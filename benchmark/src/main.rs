use clap::{Parser, Subcommand};
use dotenv::dotenv;
use sqlx::PgPool;
use std::env;

use benchmark::GraphGenerator;
use benchmark::generate_positions::PositionGenerator;
use benchmark::job_manager::JobManager;

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
        all_iterations: bool,
        n: std::ops::Range<i32>,
        dim: i8,
    },
    /// Generate graphs using GIRGs
    GenerateGraphs,

    /// Generate position embeddings (daemon mode)
    GeneratePositions,

    /// Show job queue status
    Status,

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
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let args = Args::parse();

    match args.command {
        Commands::GenerateGraphs => {
            let generator = GraphGenerator::new(
                env::var("GIRGS_PATH").unwrap_or("../../girgs/build/genhrg".to_string()),
                env::var("GRAPHS_OUTPUT_PATH").unwrap_or("../data/generated/graphs".to_string()),
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
                env::var("POSITIONS_OUTPUT_PATH")
                    .unwrap_or("../data/generated/positions".to_string()),
                job_manager,
            );

            generator.run_daemon().await?;
        }

        Commands::Status => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;
            let job_manager = JobManager::new(pool);

            let generator = PositionGenerator::new(
                String::new(), // Not needed for status
                String::new(), // Not needed for status
                job_manager,
            );

            generator.show_status().await?;
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

        Commands::Cleanup { timeout_hours } => {
            let database_url = env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
            let pool = PgPool::connect(&database_url).await?;

            let cleaned = sqlx::query_scalar!("SELECT cleanup_stale_jobs($1)", timeout_hours)
                .fetch_one(&pool)
                .await?;

            println!("Cleaned up {} stale jobs", cleaned.unwrap_or(0));
        }
    }

    Ok(())
}
