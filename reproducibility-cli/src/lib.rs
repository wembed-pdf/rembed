pub mod benchmark;
// pub mod cleanup;
// pub mod code_state;
// pub mod correctness_test;
// pub mod fscore;
// mod generate_graphs;
// pub mod generate_positions;
// pub mod intrinsic_dim;
// pub mod job_manager;
// pub mod statistics;
pub mod synthetic_data;

// pub use generate_graphs::GraphGenerator;
// pub use generate_positions::PositionGenerator;
// use indicatif::{ProgressBar, ProgressStyle};

// fn create_progress_bar(total_graphs: usize) -> ProgressBar {
//     let pb = ProgressBar::new(total_graphs as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template("{spinner:.green} [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
//             .unwrap()
//             .progress_chars("#>-"),
//     );
//     pb
// }

// pub async fn push_files() -> Result<(), Box<dyn std::error::Error>> {
//     let sync_destination =
//         std::env::var("RSYNC_DESTINATION").expect("Please set the RSYNC_DESTINATION env var");
//     let sync_source =
//         std::env::var("DATA_DIRECTORY").expect("Please set the DATA_DIRECTORY env var");

//     println!("Syncing files to: {}", sync_destination);

//     let status = tokio::process::Command::new("rsync")
//         .arg("-rlvtz")
//         .arg("--progress")
//         .arg(sync_source)
//         .arg(&sync_destination)
//         .status()
//         .await?;

//     if !status.success() {
//         return Err("Rsync failed".into());
//     }

//     println!("File sync completed successfully");
//     Ok(())
// }
// pub async fn pull_files(
//     only_graphs: bool,
//     path: Option<&str>,
//     graph_id: Option<i64>,
//     result_id: Option<i64>,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     let mut sync_destination =
//         std::env::var("RSYNC_DESTINATION").expect("Please set the RSYNC_DESTINATION env var");
//     let mut sync_source =
//         std::env::var("DATA_DIRECTORY").expect("Please set the DATA_DIRECTORY env var");

//     if only_graphs {
//         sync_destination.push_str("/generated/graphs/");
//         sync_source.push_str("/generated/graphs/");
//     } else if path.is_some() {
//         sync_destination.push_str(path.unwrap());
//         sync_source.push_str(path.unwrap());
//     }

//     println!("Syncing files from: {}", sync_destination);

//     let status = if let Some(graph_id) = graph_id {
//         tokio::process::Command::new("rsync")
//             .arg("-rlvt")
//             .arg("--progress")
//             .arg("--include")
//             .arg("*/") // allow recursion into directories
//             .arg("--include")
//             .arg(format!("*{}*", graph_id))
//             .arg("--exclude")
//             .arg("*")
//             .arg(&sync_destination)
//             .arg(&sync_source)
//             .status()
//             .await?
//     } else if let Some(result_id) = result_id {
//         tokio::process::Command::new("rsync")
//             .arg("-rlvt")
//             .arg("--progress")
//             .arg("--include")
//             .arg("*/") // allow recursion into directories
//             .arg("--include")
//             .arg(format!("*{}*", result_id))
//             .arg("--exclude")
//             .arg("*")
//             .arg(&sync_destination)
//             .arg(&sync_source)
//             .status()
//             .await?
//     } else {
//         tokio::process::Command::new("rsync")
//             .arg("-rlvt")
//             .arg("--progress")
//             .arg(&sync_destination)
//             .arg(&sync_source)
//             .status()
//             .await?
//     };

//     if !status.success() {
//         return Err("Rsync failed".into());
//     }

//     println!("File sync completed successfully");
//     Ok(())
// }
