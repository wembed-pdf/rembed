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
