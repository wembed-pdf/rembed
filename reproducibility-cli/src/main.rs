use clap::{Parser, Subcommand};
use dotenv::dotenv;
use std::{fs::File, io::Write, path::PathBuf};

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
    /// Reproduce a figure or table from the paper by running the corresponding benchmark either figure or table needs to be specified
    Reproduce {
        /// Number of the figure to reproduce (e.g. "1", "2", ...)
        #[arg(long, short)]
        figure: Option<i64>,
        /// Number of the table to reproduce (e.g. "1", "2", ...)
        #[arg(long, short)]
        table: Option<i64>,
        /// Structures to benchmark (e.g. "atree", "neighbourhood", ...) default are the structures used in the paper
        #[arg(long, num_args = 1..)]
        structures: Option<Vec<String>>,
        /// Path to the data directory containing the input files and metadata for the benchmark
        #[arg(long, short, default_value = "data")]
        data_dir: PathBuf,
        /// Path to the directory containing the plot scripts (e.g. "plots")
        #[arg(long, short, default_value = "plots/scripts")]
        plotscript_dir: PathBuf,
        /// Output file path for the benchmark results
        #[arg(long, short)]
        output: Option<String>,
        /// Run the benchmark in fast mode (may skip some computations for speed)
        #[arg(long, default_value_t = false)]
        fast: bool,
        /// Filter to only run benchmarks for specified dimensions (e.g. "2 3 ...") default is to run all dimensions
        #[arg(long, num_args = 1..)]
        dimensions: Option<Vec<usize>>,
        /// Filter to only run benchmarks for specified node counts (e.g. "10000,31623,100000,316228,1000000,3162280,10000000") default is to run all node counts
        #[arg(long, num_args = 1..)]
        node_counts: Option<Vec<usize>>,
        /// Filter to only run benchmarks for specified embedding seed (e.g 12 13 14 ) default is to run all embedding seeds
        #[arg(long, num_args = 1..)]
        embedding_seeds: Option<Vec<u64>>,
        /// Filter to only run benchmarks for specified realworld category (e.g. "nn", "clustering", "poi") default is to run all categories
        #[arg(long, num_args = 1..)]
        realworld_categories: Option<Vec<String>>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let args = Args::parse();

    match args.command {

        Commands::Reproduce { figure, table , structures, data_dir, plotscript_dir, output, fast, dimensions, node_counts, embedding_seeds, realworld_categories } => {
            if !figure.is_some() && !table.is_some() {
                return Err("Either figure or table must be specified. For example, use `cargo run -r --figure 3` or `cargo run -r --table 2`".into());
            }

            if figure.is_some() && table.is_some() {
                return Err("Cannot specify both figure and table at the same time.".into());
            }

            let is_figure = figure.is_some();
            let index = if is_figure { figure.unwrap() } else { table.unwrap() };
            let (benchmark_configs, plotscript_command, default_output) = map_figure_to_benchmark_config(index, is_figure, data_dir, plotscript_dir, fast, realworld_categories.unwrap_or(vec!["nn".to_string(), "clustering".to_string(), "poi".to_string()]))?;
        
            let output = output.unwrap_or(default_output.clone());
            if let Some(mut output_file) = validate_output_dialog(&output)? {
                println!("Running benchmark for {} with {} configurations...", if is_figure { "figure" } else { "table" }, benchmark_configs.len());
                let bar = indicatif::ProgressBar::new(benchmark_configs.len() as u64);
                for mut benchmark_config in benchmark_configs {
                    if benchmark_config.graph_generation_seed.is_some() && embedding_seeds.is_some() && !embedding_seeds.clone().unwrap().contains(&benchmark_config.graph_generation_seed.unwrap()) {
                        bar.inc(1);
                        continue;
                    }
                    benchmark_config.structures = map_structure_alias(&structures.clone().unwrap_or(benchmark_config.structures));
                    benchmark_config.dimension_filter = dimensions.clone();
                    benchmark_config.node_count_filter = node_counts.clone();
                    let runner = DistributionBenchRunner::new(benchmark_config);
                    let result = runner.run()?;
                    save_results_to_csv(&result, &mut output_file)?;
                    bar.inc(1);
                }
                bar.finish_with_message(format!("Benchmark completed. Results saved to {}.", output));
            }

            std::process::Command::new(plotscript_command.split(" ").nth(0).unwrap())
                .arg(plotscript_command.split(" ").skip(1).collect::<Vec<&str>>().join(" "))
                .arg(&output)
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .status()
                .expect(&format!("Failed to execute plot script: {}", plotscript_command));
        }
    }

    Ok(())
}

// This is not the place to create your own benchmark configurations. Instead, use the benchmark cli.
fn map_figure_to_benchmark_config(index: i64, is_figure: bool, data_dir: PathBuf, plotscript_dir: PathBuf, fast: bool, realworld_categories: Vec<String>) -> Result<(Vec<DistributionBenchConfig>, String, String), String> {
    if is_figure {
        match index {
            3 => Ok((generate_embedding_benchmark_configs(data_dir, vec!["atree".to_string(), "neighbourhood".to_string()], fast), format!("Rscript {}/fixed_n_dim.R", plotscript_dir.to_string_lossy()), "benchmark_results/embedding_benchmark_results.csv".to_string())),
            4 => Ok((generate_distribution_benchmark_configs(data_dir, vec!["atree".to_string()], fast), format!("Rscript {}/distributions.R", plotscript_dir.to_string_lossy()), "benchmark_results/distribution_benchmark_results.csv".to_string())),
            9 => Ok((generate_embedding_benchmark_configs(data_dir, vec!["atree".to_string(), "neighbourhood".to_string()], fast), format!("Rscript {}/fixed_n_dim.R", plotscript_dir.to_string_lossy()), "benchmark_results/embedding_benchmark_results.csv".to_string())),
            11 => Ok((generate_distribution_benchmark_configs(data_dir, vec!["naive_atree".to_string()], fast), format!("Rscript {}/distributions.R", plotscript_dir.to_string_lossy()), "benchmark_results/distribution_benchmark_results.csv".to_string())),
            _ => Err(format!("Figure {} was not in the paper, not a benchmark or is not supported yet", index)),
        }
    } else {
        match index {
            2 => Ok((generate_embedding_benchmark_configs(data_dir, vec!["atree".to_string()], fast), format!("python {}/table.py", plotscript_dir.to_string_lossy()), "benchmark_results/embedding_benchmark_results.csv".to_string())),
            3 => Ok((generate_realworld_benchmark_configs(data_dir, vec!["atree".to_string(), "kiddo".to_string()], fast, realworld_categories), format!("python {}/extern.py", plotscript_dir.to_string_lossy()), "benchmark_results/realworld_benchmark_results.csv".to_string())),
            4 => Ok((generate_realworld_benchmark_configs(data_dir, vec!["atree".to_string(), "kiddo".to_string()], fast, realworld_categories), format!("python {}/extern.py", plotscript_dir.to_string_lossy()), "benchmark_results/realworld_benchmark_results.csv".to_string())),
            5 => Ok((generate_realworld_benchmark_configs(data_dir, vec!["atree".to_string(), "kiddo".to_string()], fast, realworld_categories), format!("python {}/extern.py", plotscript_dir.to_string_lossy()), "benchmark_results/realworld_benchmark_results.csv".to_string())),
            6 => Ok((generate_realworld_benchmark_configs(data_dir, vec!["atree".to_string(), "kiddo".to_string()], fast, realworld_categories), format!("python {}/extern.py", plotscript_dir.to_string_lossy()), "benchmark_results/realworld_benchmark_results.csv".to_string())),
            7 => Ok((generate_embedding_benchmark_configs(data_dir, vec!["atree".to_string()], fast), format!("python {}/snn_comparison.py", plotscript_dir.to_string_lossy()), "benchmark_results/embedding_benchmark_results.csv".to_string())),
            _ => Err(format!("Table {} was not in the paper, not a benchmark or is not supported yet", index)),
        }
    }
}

struct SmallEmbeddingMetadata {
    n: usize,
    wseed: u64,
}

fn generate_embedding_benchmark_configs(data_dir: PathBuf, structures: Vec<String>, fast: bool) -> Vec<DistributionBenchConfig> {
    let data_dir = data_dir.join("embedding");
    download_dialog(&data_dir, "data/download_scripts/download_embedding.sh").expect("Failed to download embedding data");
    // read metadata csv file
    let mut metadata_map = std::collections::HashMap::new();
    // graph_id,result_id,embedding_dim,dim_hint,max_iterations,actual_iterations,seed,file_path,checksum,created_at,created_at,n,deg,ple,dim,alpha,wseed,pseed,sseed,processed_n,processed_avg_degree,file_path,checksum
    let metadata_file = std::fs::read_to_string(&data_dir.join("embedding_metadata.csv"))
        .expect("Failed to read metadata file");
    for line in metadata_file.lines().skip(1) { // skip header
        let parts: Vec<&str> = line.split(',').collect();
        assert!(parts.len() == 17, "Metadata file has unexpected number of columns");
        let n = parts[5].parse::<usize>().expect("Failed to parse n");
        let wseed = parts[12].parse::<u64>().expect("Failed to parse wseed");
        metadata_map.insert(parts[1].to_string(), SmallEmbeddingMetadata { n, wseed });
    }

    // Generate benchmark configurations for all input files in the data directory
    let mut configs = Vec::new();
    for entry in std::fs::read_dir(&data_dir.join("embedding_data")).expect("Failed to read data directory") {
        let entry = entry.expect("Failed to read directory entry");
        // skip files that do not end with train.csv
        if !entry.file_name().to_string_lossy().ends_with("train.csv") {
            continue;
        }
        println!("Processing file: {}", entry.path().display());
        let general_data_file_name = entry.file_name().to_string_lossy().to_string();
        let general_data_file_path = entry.path().parent().unwrap().to_path_buf().join(general_data_file_name.replace("train.csv", ""));
        let result_id = general_data_file_name.split("@").next().expect("Expecting data file name of form embedding_result_{result_id}@{iteration_id}_dim-{embedding_dimension}_train.csv").split("_").nth(2).expect("Expecting data file name of form embedding_result_{result_id}@{iteration_id}_dim-{embedding_dimension}_train.csv");
        let metadata = metadata_map.get(&result_id.to_string())
            .expect(&format!("Metadata not found for file: {}", general_data_file_path.to_string_lossy()));
        configs.push(DistributionBenchConfig {
            train_path: entry.path(),
            radii_path: general_data_file_path.with_file_name(format!("{}query_radii.csv", general_data_file_path.file_stem().unwrap().to_string_lossy())),
            radius: None, // use the radii from the query_radii.csv file
            query_path: Some(general_data_file_path.with_file_name(format!("{}query_points.csv", general_data_file_path.file_stem().unwrap().to_string_lossy()))),
            structures: structures.clone(),
            fast: fast,
            name: general_data_file_path.file_stem().unwrap().to_string_lossy().to_string(),
            node_count_override: Some(metadata.n),
            graph_generation_seed: Some(metadata.wseed),
            category: "embedding".into(),
            dimension_filter: None,
            node_count_filter: None,
        });
    }
    configs
}

fn generate_distribution_benchmark_configs(data_dir: PathBuf, structures: Vec<String>, fast: bool) -> Vec<DistributionBenchConfig> {
    let data_dir = data_dir.join("distributions");
    download_dialog(&data_dir, "data/download_scripts/download_distributions.sh").expect("Failed to download distributions data");
    let mut configs = Vec::new();
    for entry in std::fs::read_dir(&data_dir.join("distributions_data")).expect("Failed to read data directory") {
        let entry = entry.expect("Failed to read directory entry");
        if !entry.file_name().to_string_lossy().ends_with("train.csv") {
            continue;
        }
        let general_data_file_name = entry.file_name().to_string_lossy().to_string().replace("_train.csv", "");
        let general_data_file_path = entry.path().parent().unwrap().to_path_buf();

        configs.push(DistributionBenchConfig {
            train_path: general_data_file_path.join(general_data_file_name.clone() + "_train.csv"),
            radii_path: general_data_file_path.join(general_data_file_name.clone() + "_query_radii.csv"),
            radius: None, 
            query_path: Some(general_data_file_path.join(general_data_file_name.clone() + "_query_points.csv")),
            structures: structures.clone(),
            fast: fast,
            name: general_data_file_path.file_stem().unwrap().to_string_lossy().to_string(),
            node_count_override: None,
            graph_generation_seed: None,
            category: "distribution".into(),
            dimension_filter: None,
            node_count_filter: None,
        });
    }
    configs
}

fn generate_realworld_benchmark_configs(data_dir: PathBuf, structures: Vec<String>, fast: bool, categories: Vec<String>) -> Vec<DistributionBenchConfig> {
    let data_dir = data_dir.join("realworld");
    let mut configs = Vec::new();
    
    if categories.contains(&"nn".to_string()) {
        // generate nearest neighbor benchmark configurations 
        let nearest_neighbor_data_dir = data_dir.join("nearest_neighbor_data");
        download_dialog(&nearest_neighbor_data_dir, "data/download_scripts/download_nn.sh").expect("Failed to download nearest neighbor data");
        for entry in std::fs::read_dir(&nearest_neighbor_data_dir).expect(format!("Missing nearest neighbor data directory: {}", nearest_neighbor_data_dir.to_string_lossy()).as_str()) {
            let entry = entry.expect("Failed to read directory entry");
            if !entry.file_name().to_string_lossy().ends_with("train.csv") {
                continue;
            }
            let general_data_file_name = entry.file_name().to_string_lossy().to_string();
            let general_data_file_path = entry.path().parent().unwrap().to_path_buf().join(general_data_file_name.replace("_train.csv", ""));
            let query_path = general_data_file_path.with_file_name(format!("{}_query.csv", general_data_file_path.file_stem().unwrap().to_string_lossy()));
            for radius in get_radii_for_realworld_dataset(general_data_file_path.file_stem().unwrap().to_string_lossy().as_ref()) {
                configs.push(DistributionBenchConfig {
                    train_path: entry.path(),
                    radii_path: PathBuf::new(), // use one radius for all queries
                    radius: Some(radius),
                    query_path: Some(query_path.clone()),
                    structures: structures.clone(),
                    fast: fast,
                    name: general_data_file_path.file_stem().unwrap().to_string_lossy().to_string(),
                    node_count_override: None,
                    graph_generation_seed: None,
                    category: "nn".into(),
                    dimension_filter: None,
                    node_count_filter: None,
                });
            }
        }
    }

    if categories.contains(&"clustering".to_string()) {
        // generate clustering benchmark configurations
        let clustering_data_dir = data_dir.join("clustering_data");
        download_dialog(&clustering_data_dir, "data/download_scripts/download_clustering.sh").expect("Failed to download clustering data");
        for entry in std::fs::read_dir(&clustering_data_dir).expect(format!("Missing clustering data directory: {}", clustering_data_dir.to_string_lossy()).as_str()) {
            let entry = entry.expect("Failed to read directory entry");
            assert!(entry.file_name().to_string_lossy().ends_with(".csv"), "Expecting clustering data files to end with .csv");
            let general_data_file_name = entry.file_name().to_string_lossy().to_string();
            let general_data_file_path = entry.path().parent().unwrap().to_path_buf().join(general_data_file_name.replace(".csv", ""));
            for radius in get_radii_for_realworld_dataset(general_data_file_path.file_stem().unwrap().to_string_lossy().as_ref()) {
                configs.push(DistributionBenchConfig {
                    train_path: entry.path(),
                    radii_path: PathBuf::new(), // use one radius for all queries
                    radius: Some(radius),
                    query_path: None, // clustering queries are all-to-all, so no query file is needed
                    structures: structures.clone(),
                    fast: fast,
                    name: general_data_file_path.file_stem().unwrap().to_string_lossy().to_string(),
                    node_count_override: None,
                    graph_generation_seed: None,
                    category: "clustering".into(),
                    dimension_filter: None,
                    node_count_filter: None,
                });
            }
        }
    }

    if categories.contains(&"poi".to_string()) {
        // generate poi benchmark configurations
        let poi_data_dir = data_dir.join("poi_data");
        download_dialog(&poi_data_dir, "data/download_scripts/download_poi.sh").expect("Failed to download poi data");
        let poi_radii = vec![500.0, 1000.0, 2000.0, 5000.0];
        let poi_pairs = vec![
            ("parking_hospital", "parking.csv", "hospital.csv"),
            ("restaurant_trainstation", "restaurant.csv", "trainstation.csv"),
            ("pharmacy_hospital", "pharmacy.csv", "hospital.csv"),
            ("busstop_trainstation", "busstop.csv", "trainstation.csv"),
            ("atm_supermarket", "atm.csv", "supermarket.csv"),
            ("hospital_university", "hospital.csv", "university.csv"),
            ("bakery_university", "bakery.csv", "university.csv"),
        ];
        for (name, poi_file, query_file) in poi_pairs {
            let poi_path = poi_data_dir.join(poi_file);
            let query_path = poi_data_dir.join(query_file);
            for radius in &poi_radii {
                configs.push(DistributionBenchConfig {
                    train_path: poi_path.clone(),
                    radii_path: PathBuf::new(), // use one radius for all queries
                    radius: Some(*radius),
                    query_path: Some(query_path.clone()),
                    structures: structures.clone(),
                    fast: fast,
                    name: name.to_string(),
                    node_count_override: None,
                    graph_generation_seed: None,
                    category: "poi".into(),
                    dimension_filter: None,
                    node_count_filter: None,
                });
            }
        }
    }
    configs
}

fn get_radii_for_realworld_dataset(dataset_name: &str) -> Vec<f64> {
    match dataset_name {
        "deep" => vec![0.69, 0.75, 0.82, 0.88, 0.94],
        "fmn" => vec![800.0, 900.0, 1000.0, 1100.0, 1200.0],
        "sift" => vec![210.0, 230.0, 250.0, 270.0, 290.0],
        "sift_large" => vec![210.0, 230.0, 250.0, 270.0, 290.0],
        "gist" => vec![0.8, 0.85, 0.9, 0.95, 1.0],
        "glo" => vec![0.94, 0.97, 1.01, 1.04, 1.07],
        "banknote" => vec![0.1, 0.2, 0.3, 0.4, 0.5],
        "dermatology" => vec![5.0, 5.1, 5.2, 5.3, 5.4],
        "ecoli" => vec![0.5, 0.6, 0.7, 0.8, 0.9],
        "phoneme" => vec![8.5, 8.6, 8.7, 8.8, 8.9],
        "wine" => vec![2.2, 2.3, 2.4, 2.5, 2.6],
        _ => panic!("Radii not defined for dataset: {}", dataset_name),
    }
}

fn save_results_to_csv(results: &Vec<BenchmarkRecord>, output_file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for record in results {
        writeln!(output_file, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", record.dimension, record.node_count, record.radius, record.graph_generation_seed.unwrap_or(0), record.avg_returned_points, record.name, record.category, record.data_structure, record.wall_time_mean_ns, record.wall_time_stddev_ns, record.instructions_mean, record.instructions_stddev, record.cycles_mean, record.cycles_stddev, record.sample_count)?;
    }
    Ok(())
}

fn validate_output_dialog(output: &String) -> Result<Option<File>, Box<dyn std::error::Error>> {
    let output_path = std::path::Path::new(&output);
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create output directory");
        }
        // If the output file already exists, give the user a warning and ask if they want to overwrite it [Y], abort [N] or skip the benchmark and use the existing file for plotting [S]
        if output_path.exists() {
            println!("Output file already exists: {}", output_path.display());
            println!("You can specify a different output file path using the --output option.");
            println!("Do you want to overwrite the existing file? [Y]es, [N]o, [S]kip benchmark and use existing results for plotting, [A]ppend to existing file (duplicate rows can lead to incorrect plots)");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).expect("Failed to read input");
            match input.trim().to_uppercase().as_str() {
                "Y" => {}
                "N" => return Err("Aborted by user".into()),
                "S" => {
                    println!("Skipping benchmark and using existing file for plotting.");
                    return Ok(None);
                }
                "A" => {
                    println!("Appending to existing file.");
                    return Ok(Some(std::fs::OpenOptions::new().append(true).open(output_path).expect("Failed to open existing file for appending")));
                }
                _ => {
                    println!("Invalid input, expected Y, N, S or A");
                    return validate_output_dialog(output);
                }
            }
        }
    }
    let mut file = std::fs::File::create(output_path).expect("Failed to create output file");
    writeln!(file, "dimension,node_count,radius,graph_gen_seed,avg_returned_points,name,category,data_structure,wall_time_mean_ns,wall_time_stddev_ns,instructions_mean,instructions_stddev,cycles_mean,cycles_stddev,sample_count")?;
    Ok(Some(file))
}

fn download_dialog(input_path: &PathBuf, download_script: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !input_path.exists() {
        println!("Benchmark data does not exist: {}", input_path.display());
        println!("You can download the required data using the following script: {}", download_script);
        println!("Do you want to run the download script now? [Y]es, [N]o");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("Failed to read input");
        match input.trim().to_uppercase().as_str() {
            "Y" => {
                std::process::Command::new("bash")
                    .arg(download_script)
                    .stdout(std::process::Stdio::inherit())
                    .stderr(std::process::Stdio::inherit())
                    .status()
                    .expect(&format!("Failed to execute download script: {}", download_script));
            }
            "N" => return Err("Aborted by user".into()),
            _ => {
                println!("\n Invalid input, expected Y or N");
                return download_dialog(input_path, download_script);
            }
        }
    }
    Ok(())
}

fn map_structure_alias(structures: &Vec<String>) -> Vec<String> {
    structures.iter().map(|s| match s.as_str() {
        "sprk" => "atree".to_string(),
        "spark" => "atree".to_string(),
        _ => s.clone(),
    }).collect()
}