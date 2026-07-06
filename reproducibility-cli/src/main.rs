use clap::{Parser, Subcommand};
use dotenv::dotenv;
use std::{io::Write, path::PathBuf};

use reproducibility_cli::benchmark::distribution_bench::{
    DistributionBenchConfig, DistributionBenchRunner, BenchmarkRecord,
};

#[derive(Parser)]
#[command(name = "benchmark")]
#[command(about = "Graph generation and position embedding benchmark tool")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Reproduce a figure or table from the paper by running the corresponding benchmark
    Reproduce {
        /// Number of the figure or table to reproduce (e.g. "1", "2", ...)
        #[arg(long, short)]
        figure: i64,
        /// Specify to reproduce a table instead of a figure
        #[arg(long, short, default_value_t = false)]
        table: bool,
        /// Path to the data directory containing the input files for the benchmark
        #[arg(long, short)]
        data_dir: PathBuf,
        /// Path to the metadata CSV file (required for embedding benchmarks)
        #[arg(long, short)]
        metadata_file: Option<PathBuf>,
        /// Path to the directory containing the plot scripts (e.g. "plots")
        #[arg(long, short)]
        plotscript_dir: PathBuf,
        /// Output file path for the benchmark results
        #[arg(long, short)]
        output: String,
        /// Run the benchmark in fast mode (may skip some computations for speed)
        #[arg(long, short, default_value_t = false)]
        fast: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let args = Args::parse();

    match args.command {

        Commands::Reproduce { figure, table , data_dir, metadata_file, plotscript_dir, output, fast } => {
            let (benchmark_configs, plotscript_path) = map_figure_to_benchmark_config(figure, table, data_dir, metadata_file, plotscript_dir, fast)?;
            let mut results = Vec::new();
            for benchmark_config in benchmark_configs {
                println!("Running benchmark for config: {:?}", benchmark_config);
                let runner = DistributionBenchRunner::new(benchmark_config);
                let result = runner.run()?;
                results.extend(result);
            }
            save_results_to_csv(&results, &output)?;
            // Execute plot script
            std::process::Command::new("Rscript")
                .arg(&plotscript_path)
                .output()
                .expect("Failed to execute plot script");
        }
    }

    Ok(())
}

// This is not the place to create your own benchmark configurations. Instead, use the benchmark cli.
fn map_figure_to_benchmark_config(figure: i64, table: bool, data_dir: PathBuf, metadata_file: Option<PathBuf>, plotscript_dir: PathBuf, fast: bool) -> Result<(Vec<DistributionBenchConfig>, PathBuf), String> {
    match (figure, table) {
        (1, false) => Ok((generate_embedding_benchmark_configs(data_dir, metadata_file.expect("embedding benchmarks require metadata"), vec!["atree".to_string()], fast), plotscript_dir.join("fixed_n_dim.R"))),
        // Add more mappings for other figures and tables as needed
        _ => Err(format!("{} {} was not in the paper", {if table { "Table" } else { "Figure" }}, figure)),
    }
}

struct SmallMetadata {
    n: usize,
    wseed: u64,
}

fn generate_embedding_benchmark_configs(data_dir: PathBuf, metadata_file: PathBuf, structures: Vec<String>, fast: bool) -> Vec<DistributionBenchConfig> {
    // read metadata csv file
    let mut metadata_map = std::collections::HashMap::new();
    // graph_id,result_id,embedding_dim,dim_hint,max_iterations,actual_iterations,seed,file_path,checksum,created_at,created_at,n,deg,ple,dim,alpha,wseed,pseed,sseed,processed_n,processed_avg_degree,file_path,checksum
    let metadata_file = std::fs::read_to_string(&metadata_file)
        .expect("Failed to read metadata file");
    for line in metadata_file.lines().skip(1) { // skip header
        let parts: Vec<&str> = line.split(',').collect();
        assert!(parts.len() == 23, "Metadata file has unexpected number of columns");
        let n = parts[11].parse::<usize>().expect("Failed to parse n");
        let wseed = parts[16].parse::<u64>().expect("Failed to parse wseed");
        metadata_map.insert(parts[1].to_string(), SmallMetadata { n, wseed });
    }

    // Generate benchmark configurations for all input files in the data directory
    let mut configs = Vec::new();
    for entry in std::fs::read_dir(&data_dir).expect("Failed to read data directory") {
        let entry = entry.expect("Failed to read directory entry");
        // skip files that do not end with train.csv
        if !entry.file_name().to_string_lossy().ends_with("train.csv") {
            continue;
        }
        let general_data_file_name = entry.file_name().to_string_lossy().to_string();
        let general_data_file_path = entry.path().parent().unwrap().to_path_buf().join(general_data_file_name.replace("train.csv", ""));
        let result_id = general_data_file_name.split("@").next().expect("Expecting data file name of form embedding_result_{result_id}@{iteration_id}_dim-{embedding_dimension}_train.csv").split("_").nth(2).expect("Expecting data file name of form embedding_result_{result_id}@{iteration_id}_dim-{embedding_dimension}_train.csv");
        let metadata = metadata_map.get(&result_id.to_string())
            .expect(&format!("Metadata not found for file: {}", general_data_file_path.to_string_lossy()));
        configs.push(DistributionBenchConfig {
            train_path: entry.path(),
            radii_path: general_data_file_path.with_file_name(format!("{}query_radii.csv", general_data_file_path.file_stem().unwrap().to_string_lossy())),
            query_path: general_data_file_path.with_file_name(format!("{}query_points.csv", general_data_file_path.file_stem().unwrap().to_string_lossy())),
            structures: structures.clone(),
            fast: fast,
            node_count_override: Some(metadata.n),
            graph_generation_seed: Some(metadata.wseed),
        });
    }
    configs
}

fn save_results_to_csv(results: &Vec<BenchmarkRecord>, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output_file = std::fs::File::create(output_path)?;
    writeln!(&output_file, "dimension,node_count,radius,graph_gen_seed,avg_returned_points,name,data_structure,wall_time_mean_ns,wall_time_stddev_ns,instructions_mean,instructions_stddev,cycles_mean,cycles_stddev,sample_count")?;
    for record in results {
        writeln!(&output_file, "{},{},{},{},{},{},{},{},{},{},{},{},{},{}", record.dimension, record.node_count, record.radius, record.graph_generation_seed.unwrap_or(0), record.avg_returned_points, record.name, record.data_structure, record.wall_time_mean_ns, record.wall_time_stddev_ns, record.instructions_mean, record.instructions_stddev, record.cycles_mean, record.cycles_stddev, record.sample_count)?;
    }
    Ok(())
}
