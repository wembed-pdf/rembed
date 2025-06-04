use crate::job_manager::{JobManager, PositionJob};
use sha2::{Digest, Sha256};
use std::process::Command;
use std::time::Duration;
use tokio::time::sleep;

pub struct PositionGenerator {
    pub wembed_path: String,
    pub output_path: String,
    pub job_manager: JobManager,
}

impl PositionGenerator {
    pub fn new(wembed_path: String, output_path: String, job_manager: JobManager) -> Self {
        Self { wembed_path, output_path, job_manager }
    }

    pub async fn run_daemon(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting position generation daemon...");
        std::fs::create_dir_all(&self.output_path)?;

        loop {
            match self.job_manager.claim_next_job().await {
                Ok(Some(job)) => {
                    println!("Processing job {} - Graph {} Dim {}", job.job_id, job.graph_id, job.embedding_dim);
                    if let Err(e) = self.process_job(job.clone()).await {
                        eprintln!("Job {} failed: {}", job.job_id, e);
                        let _ = self.job_manager.fail_job(job.job_id, &e.to_string()).await;
                    }
                }
                Ok(None) => sleep(Duration::from_secs(5)).await,
                Err(e) => {
                    eprintln!("Error claiming job: {}", e);
                    sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    async fn process_job(&self, job: PositionJob) -> Result<(), Box<dyn std::error::Error>> {
        let output_filename = format!(
            "{}_graph-{}_dim-{}_seed-{}.log",
            job.job_id, job.graph_id, job.embedding_dim, job.seed
        );
        let output_path = String::from("spatial_log_positions.log");
        // let output_path = format!("{}/{}", self.output_path, output_filename);

        let status = Command::new(&self.wembed_path)
            .arg("-i").arg(&job.graph_file_path)
            .arg("--dim-hint").arg(job.dim_hint.to_string())
            .arg("--dim").arg(job.embedding_dim.to_string())
            .arg("--iterations").arg(job.max_iterations.to_string())
            .arg("--seed").arg(job.seed.to_string()) // Add seed for reproducibility
            .status()?;

        if !status.success() {
            return Err(format!("WEmbed failed with exit code: {:?}", status.code()).into());
        }

        if !std::path::Path::new(&output_path).exists() {
            return Err("WEmbed completed but output file was not created".into());
        }

        let checksum = calculate_file_checksum(&output_path)?;
        
        // Parse actual iterations from log file if needed
        let actual_iterations = parse_actual_iterations(&output_path).unwrap_or(None);

        self.job_manager.complete_job(job.job_id, &output_path, &checksum, actual_iterations).await?;
        println!("Completed job {} - {}", job.job_id, output_filename);
        Ok(())
    }

    pub async fn show_status(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (pending, running, completed, failed) = self.job_manager.get_job_stats().await?;
        println!("Jobs: {} pending, {} running, {} completed, {} failed", pending, running, completed, failed);
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
