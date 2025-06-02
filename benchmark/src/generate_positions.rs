use std::process::Command;

use indicatif::{ProgressBar, ProgressStyle};

pub struct GraphGenerator {
    pub wembed_path: String,
    pub graphs_path: String,
    pub output_path: String,
}

impl GraphGenerator {
    pub fn new(wembed_path: String, graphs_path: String, output_path: String) -> Self {
        Self {
            wembed_path,
            graphs_path,
            output_path,
        }
    }

    pub fn generate(&self) {
        // Get all graph instances from the graphs path
        let graph_files = std::fs::read_dir(&self.graphs_path)
            .expect("Failed to read graph directory")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("graph"))
            .collect::<Vec<_>>();

        println!("Generating positions using WEmbed at: {}", self.wembed_path);
        println!("Graph files found: {}", graph_files.len());
        println!("Output will be saved to: {}", self.output_path);
        let pb = ProgressBar::new(graph_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {wide_msg} [{elapsed_precise}]")
                .unwrap(),
        );
        pb.set_message("Starting position generation");
        let own_path = std::env::current_dir().expect("Failed to get current directory");
        for graph_file in graph_files {
            let graph_name = graph_file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let output_file = format!("{}/{}.pos", self.output_path, graph_name);
            let i = own_path.join(&graph_file);
            let o = own_path.join(&output_file);

            Command::new(&self.wembed_path)
                .arg("-i")
                .arg(i)
                .arg("-o")
                .arg(o)
                .arg("--dim-hint")
                .arg("8")
                .arg("--dim")
                .arg("8")
                .arg("--iterations")
                .arg("100")
                .status()
                .expect("Failed to execute WEmbed command");

            pb.inc(1);
        }
    }
}
