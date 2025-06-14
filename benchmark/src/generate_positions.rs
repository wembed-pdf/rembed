use crate::job_manager::{JobManager, PositionJob};
use rembed::dim_reduction::LayeredLsh;
use rembed::embedder::{EmbedderOptions, WEmbedder};
use rembed::kiddo::Kiddo;
use sha2::{Digest, Sha256};
use std::time::Duration;
use tokio::time::sleep;

pub struct PositionGenerator {
    pub wembed_path: String,
    pub output_path: String,
    pub job_manager: JobManager,
}

impl PositionGenerator {
    pub fn new(wembed_path: String, output_path: String, job_manager: JobManager) -> Self {
        Self {
            wembed_path,
            output_path,
            job_manager,
        }
    }

    pub async fn run_daemon(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting position generation daemon...");
        std::fs::create_dir_all(&self.output_path)?;
        crate::pull_files().await?;

        loop {
            match self.job_manager.claim_next_job().await {
                Ok(Some(job)) => {
                    println!(
                        "Processing job {} - Graph {} Dim {}",
                        job.job_id, job.graph_id, job.embedding_dim
                    );
                    if let Err(e) = self.process_job(job.clone()).await {
                        eprintln!("Job {} failed: {}", job.job_id, e);
                        let _ = self.job_manager.fail_job(job.job_id, &e.to_string()).await;
                    }
                }
                Ok(None) => {
                    sleep(Duration::from_secs(5)).await;
                    crate::push_files().await?;
                }
                Err(e) => {
                    eprintln!("Error claiming job: {}", e);
                    sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    async fn process_job(&self, job: PositionJob) -> Result<(), Box<dyn std::error::Error>> {
        let output_filename = format!(
            "graph-{}_dim-{}_dim-hint-{}_seed-{}.log",
            job.graph_id, job.embedding_dim, job.dim_hint, job.seed
        );
        let output_path_without_prefix = format!("generated/positions/{}", output_filename);
        let output_path = format!("{}/{}", self.output_path, output_path_without_prefix);
        let graph_path = format!("{}/{}", self.output_path, job.graph_file_path);
        let graph = rembed::graph::Graph::parse_from_edge_list_file(
            &graph_path,
            job.embedding_dim as usize,
            job.dim_hint as usize,
        )?;

        let options = EmbedderOptions::default();
        run_embedding_dynamic(
            job.seed as u64,
            &graph,
            options,
            job.embedding_dim as usize,
            &output_path,
        )?;

        if !std::path::Path::new(&output_path).exists() {
            return Err("WEmbed completed but output file was not created".into());
        }

        let checksum = calculate_file_checksum(&output_path)?;

        // Parse actual iterations from log file if needed
        let actual_iterations = parse_actual_iterations(&output_path).unwrap_or(None);

        self.job_manager
            .complete_job(
                job.job_id,
                &output_path_without_prefix,
                &checksum,
                actual_iterations,
            )
            .await?;
        println!("Completed job {} - {}", job.job_id, output_filename);
        Ok(())
    }

    pub async fn show_status(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (pending, running, completed, failed) = self.job_manager.get_job_stats().await?;
        println!(
            "Jobs: {} pending, {} running, {} completed, {} failed",
            pending, running, completed, failed
        );
        Ok(())
    }
}

fn calculate_file_checksum(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let contents = std::fs::read(file_path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_actual_iterations(file_path: &str) -> Result<Option<i32>, Box<dyn std::error::Error>> {
    let mut max_iteration = 0;
    for line in std::fs::read_to_string(file_path)?.lines() {
        if line.starts_with("ITERATION") {
            max_iteration = line.split_whitespace().last().unwrap().parse()?;
        }
    }
    Ok(Some(max_iteration))
}
fn run_embedding_dynamic(
    seed: u64,
    graph: &rembed::graph::Graph,
    options: EmbedderOptions,
    dim: usize,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match dim {
        2 => run_embedding::<2>(seed, graph, options, output_path),
        4 => run_embedding::<4>(seed, graph, options, output_path),
        8 => run_embedding::<8>(seed, graph, options, output_path),
        16 => run_embedding::<16>(seed, graph, options, output_path),
        32 => run_embedding::<32>(seed, graph, options, output_path),
        _ => unreachable!("not compiled for dim {dim}"),
    }
}
fn run_embedding<const D: usize>(
    seed: u64,
    graph: &rembed::graph::Graph,
    options: EmbedderOptions,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut embedder: WEmbedder<Kiddo<D>, D> = WEmbedder::random(seed, graph, options);
    embedder.embed();
    let sparse_iterations: Vec<_> = embedder.history().iter().step_by(10).cloned().collect();

    rembed::parsing::write_test_file(output_path, sparse_iterations.as_slice())
}
