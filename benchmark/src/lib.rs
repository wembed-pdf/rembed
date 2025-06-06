mod generate_graphs;
pub mod generate_positions;
pub mod job_manager;

pub use generate_graphs::GraphGenerator;
pub use generate_positions::PositionGenerator;
use indicatif::{ProgressBar, ProgressStyle};

fn create_progress_bar(total_graphs: usize) -> ProgressBar {
    let pb = ProgressBar::new(total_graphs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb
}
pub async fn push_files() -> Result<(), Box<dyn std::error::Error>> {
    let sync_destination =
        std::env::var("RSYNC_DESTINATION").expect("Please set the RSYNC_DESTINATION env var");
    let sync_source =
        std::env::var("DATA_DIRECTORY").expect("Please set the DATA_DIRECTORY env var");

    println!("Syncing files to: {}", sync_destination);

    let status = tokio::process::Command::new("rsync")
        .arg("-rlvtz")
        .arg("--progress")
        .arg(sync_source)
        .arg(&sync_destination)
        .status()
        .await?;

    if !status.success() {
        return Err("Rsync failed".into());
    }

    println!("File sync completed successfully");
    Ok(())
}
pub async fn pull_files() -> Result<(), Box<dyn std::error::Error>> {
    let sync_destination =
        std::env::var("RSYNC_DESTINATION").expect("Please set the RSYNC_DESTINATION env var");
    let sync_source =
        std::env::var("DATA_DIRECTORY").expect("Please set the DATA_DIRECTORY env var");

    println!("Syncing files to: {}", sync_destination);

    let status = tokio::process::Command::new("rsync")
        .arg("-rlvtz")
        .arg("--progress")
        .arg(&sync_destination)
        .arg(&sync_source)
        .status()
        .await?;

    if !status.success() {
        return Err("Rsync failed".into());
    }

    println!("File sync completed successfully");
    Ok(())
}
