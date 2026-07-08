use crate::benchmark::runner;
use crate::synthetic_data::{self, PointDistribution};
use criterion::Criterion;
use memmap2::Mmap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rembed::{Embedding, NodeId};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct DistributionBenchConfig {
    pub train_path: PathBuf,
    pub query_path: Option<PathBuf>,
    pub radii_path: PathBuf,
    pub radius: Option<f64>,
    pub structures: Vec<String>,
    pub fast: bool,
    pub name: String,
    pub category: String,
    // embedding related parameters
    pub node_count_override: Option<usize>,
    pub graph_generation_seed: Option<u64>,
    // Filters
    pub dimension_filter: Option<Vec<usize>>,
    pub node_count_filter: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkRecord {
    pub dimension: usize,
    pub node_count: usize,
    pub radius: f64,
    pub graph_generation_seed: Option<u64>,
    pub avg_returned_points: f64,
    pub name: String,
    pub category: String,
    pub data_structure: String,
    pub wall_time_mean_ns: u64,
    pub wall_time_stddev_ns: u64,
    pub instructions_mean: f64,
    pub instructions_stddev: f64,
    pub cycles_mean: f64,
    pub cycles_stddev: f64,
    pub sample_count: usize,
}

pub struct DistributionBenchRunner {
    config: DistributionBenchConfig,
}

impl DistributionBenchRunner {
    pub fn new(config: DistributionBenchConfig) -> Self {
        Self { config }
    }

    pub fn run(&self) -> Result<Vec<BenchmarkRecord>, Box<dyn std::error::Error>> {
        let mut all_results = Vec::new();
        let trainset = parse_point_file(&self.config.train_path.to_string_lossy())?;
        let distribution = PointDistribution::Benchmarkset {
            points: trainset.points.clone(),
            name: self.config.name.clone(),
        };
        let node_count = trainset.node_count;
        if let Some(filter) = &self.config.node_count_filter {
            let real_node_count = if let Some(n_override) = self.config.node_count_override {
                n_override
            } else {
                node_count
            };
            if !filter.contains(&real_node_count) {
                println!("Skipping node count {} due to filter", node_count);
                return Ok(all_results);
            }
        }
        if let Some(filter) = &self.config.dimension_filter {
            if !filter.contains(&trainset.dimension) {
                println!("Skipping dimension {} due to filter", trainset.dimension);
                return Ok(all_results);
            }
        }
        let dimension = trainset.dimension;
        // let queryset = self.config.query_path.as_ref().map(|p| parse_point_file(p.to_string_lossy()).unwrap().points).unwrap_or_else(Vec::new);
        let queryset = {
            if let Some(query_path) = &self.config.query_path {
                Some(parse_point_file(&query_path.to_string_lossy())?.points)
            } else {
                Some(trainset.points.clone())
            }
        };
        
        let query_radii = if self.config.radii_path.exists() {
            Some(parse_radius_file(&self.config.radii_path.to_string_lossy())?)
        } else {
            None
        };
        
        let global_radius = if let Some(radius) = self.config.radius {
            radius
        } else if let Some(radii) = &query_radii {
            // Use the average of the provided query radii as the global radius for benchmarking
            let avg_radius = radii.iter().sum::<f64>() / radii.len() as f64;
            avg_radius
        } else {
            panic!("No radius specified and no query radii provided");
        };

        let results = self.run_benchmarks_for_dimension(
            dimension,
            node_count,
            global_radius,
            &distribution,
            queryset,
            query_radii,
            self.config.fast,
        )?;
        all_results.extend(results);

        Ok(all_results)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_benchmark_for_config<const D: usize>(
        &self,
        node_count: usize,
        radius: f64,
        distribution: &PointDistribution,
        queryset: Option<Vec<Vec<f32>>>,
        query_radii: Option<Vec<f64>>,
        fast: bool,
    ) -> Result<Vec<BenchmarkRecord>, Box<dyn std::error::Error>> {
        // Use the provided distribution points
        assert!(
            matches!(distribution, PointDistribution::Benchmarkset { .. }),
            "Only Benchmarkset distribution is supported for reproducibility benchmarks"
        );
        let points =
            synthetic_data::generate_points::<D>(node_count, distribution, 0);

        // Create minimal graph
        let graph = synthetic_data::create_minimal_graph(node_count, radius.sqrt());

        // Build embedding
        let embedding = Embedding {
            positions: points,
            graph: &graph,
        };

        // Get data structures
        let mut data_structures: Vec<_> = if !self.config.structures.is_empty() {
            rembed::data_structures(&embedding)
                .filter(|s| self.config.structures.contains(&s.name()))
                .collect()
        } else {
            rembed::data_structures(&embedding).collect()
        };
        for structure in &mut data_structures {
            structure.set_radius_hint(radius);
        }

        println!("Structures to benchmark: {:?}", data_structures.iter().map(|s| s.name()).collect::<Vec<_>>());
        println!("Available structures: {:?}", rembed::data_structures(&embedding).map(|s| s.name()).collect::<Vec<_>>());

        // Update positions for all structures
        for structure in &mut data_structures {
            structure.update_positions(&embedding.positions, None);
        }

        let query_indices: Vec<NodeId> = (0..node_count).collect();

        // Benchmark each structure
        let mut results = Vec::new();
        // let mut c = Criterion::default().without_plots();
        let mut c = Criterion::default();
        let mut group = c.benchmark_group(format!(
            "dist_bench_{}_d{}_n{}",
            distribution.name(),
            D,
            node_count
        ));

        let query_pos_list = if let Some(queryset) = queryset {
            let all_points: Vec<rembed::dvec::DVec<D>> = queryset
                .iter()
                .map(|q| {
                    assert_eq!(
                        q.len(),
                        D,
                        "Query point dimension does not match embedding dimension"
                    );
                    rembed::dvec::DVec::<D> {
                        components: q.clone().try_into().expect("Failed to convert query point"),
                    }
                })
                .collect();
            Some(all_points)
        } else {
            None
        };


        for structure in &data_structures {
            let measurement = runner::profile_datastructure_query(
                &embedding,
                &mut group,
                &query_indices,
                query_pos_list.clone(),
                Some(radius),
                query_radii.clone(),
                runner::BenchmarkType::Radius(radius as f32, format!("radius_{}", radius)),
                structure.as_ref(),
                fast,
            );

            results.push(BenchmarkRecord {
                dimension: D,
                node_count: {if let Some(n_override) = self.config.node_count_override {
                    n_override
                } else {
                    node_count
                }},
                radius,
                graph_generation_seed: self.config.graph_generation_seed,
                avg_returned_points: measurement.avg_returned_points,
                name: distribution.name().to_string(),
                category: self.config.category.clone(),
                data_structure: measurement.data_structure_name,
                wall_time_mean_ns: measurement.measurement.wall_time_mean.as_nanos() as u64,
                wall_time_stddev_ns: measurement.measurement.wall_time_stddev.as_nanos() as u64,
                instructions_mean: measurement.measurement.instructions_mean,
                instructions_stddev: measurement.measurement.instructions_stddev,
                cycles_mean: measurement.measurement.cycles_mean,
                cycles_stddev: measurement.measurement.cycles_stddev,
                sample_count: measurement.sample_count,
            });
        }

        group.finish();
        Ok(results)
    }

    pub fn write_output(
        &self,
        results: &[BenchmarkRecord],
        output_path: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match output_path {
            Some(path) => {
                // Write CSV to file
                let file = std::fs::File::create(path)?;
                let mut writer = std::io::BufWriter::new(file);
                Self::write_csv(&mut writer, results)?;
                writer.flush()?;
                eprintln!("Results written to {}", path);
            }
            None => {
                // Pretty print to stdout
                Self::write_pretty(&mut std::io::stdout(), results)?;
            }
        }
        Ok(())
    }

    fn write_csv<W: Write>(
        writer: &mut W,
        results: &[BenchmarkRecord],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Write header
        writeln!(
            writer,
            "dimension,node_count,radius,avg_returned_points,distribution,category,structure,wall_time_ns,wall_time_std_ns,instructions,instructions_std,cycles,cycles_std,samples"
        )?;

        // Write data rows
        for record in results {
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{}",
                record.dimension,
                record.node_count,
                record.radius,
                record.avg_returned_points,
                record.name,
                record.data_structure,
                record.wall_time_mean_ns,
                record.wall_time_stddev_ns,
                record.instructions_mean,
                record.instructions_stddev,
                record.cycles_mean,
                record.cycles_stddev,
                record.sample_count
            )?;
        }

        Ok(())
    }

    fn write_pretty<W: Write>(
        writer: &mut W,
        results: &[BenchmarkRecord],
    ) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(writer, "\n=== Distribution Benchmark Results ===")?;
        writeln!(writer)?;

        // Group results by configuration (use radius as f64 for display but string for grouping)
        let mut grouped: std::collections::HashMap<
            (usize, usize, String, String),
            Vec<&BenchmarkRecord>,
        > = std::collections::HashMap::new();

        for record in results {
            let key = (
                record.dimension,
                record.node_count,
                format!("{}", record.radius), // Convert f64 to string for HashMap key
                record.name.clone(),
            );
            grouped.entry(key).or_default().push(record);
        }

        // Sort by configuration
        let mut configs: Vec<_> = grouped.keys().collect();
        configs.sort();

        for key in configs {
            let records = &grouped[key];
            writeln!(
                writer,
                "Configuration: Dimension={}, Nodes={}, Radius={}, Distribution={}",
                key.0, key.1, key.2, key.3
            )?;
            writeln!(writer, "{}", "-".repeat(90))?;

            for record in records {
                // Convert nanoseconds to milliseconds
                let wall_time_mean_ms = record.wall_time_mean_ns as f64 / 1_000_000.0;
                let wall_time_stddev_ms = record.wall_time_stddev_ns as f64 / 1_000_000.0;

                writeln!(
                    writer,
                    "{:20} {:>8.3} ms ± {:>6.3} ms  |  {:>8} ± {:>6} inst  |  {:>8} ± {:>6} cycles",
                    record.data_structure,
                    wall_time_mean_ms,
                    wall_time_stddev_ms,
                    runner::format_number(record.instructions_mean),
                    runner::format_number(record.instructions_stddev),
                    runner::format_number(record.cycles_mean),
                    runner::format_number(record.cycles_stddev)
                )?;
            }

            writeln!(writer)?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_benchmarks_for_dimension(
        &self,
        dim: usize,
        node_count: usize,
        radius: f64,
        distribution: &PointDistribution,
        queryset: Option<Vec<Vec<f32>>>,
        query_radii: Option<Vec<f64>>,
        fast: bool,
    ) -> Result<Vec<BenchmarkRecord>, Box<dyn std::error::Error>> {
        match dim {
            2 => Ok(self.run_benchmark_for_config::<2>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            3 => Ok(self.run_benchmark_for_config::<3>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            4 => Ok(self.run_benchmark_for_config::<4>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            5 => Ok(self.run_benchmark_for_config::<5>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            6 => Ok(self.run_benchmark_for_config::<6>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            7 => Ok(self.run_benchmark_for_config::<7>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            8 => Ok(self.run_benchmark_for_config::<8>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            9 => Ok(self.run_benchmark_for_config::<9>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            10 => Ok(self.run_benchmark_for_config::<10>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            11 => Ok(self.run_benchmark_for_config::<11>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            12 => Ok(self.run_benchmark_for_config::<12>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            13 => Ok(self.run_benchmark_for_config::<13>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            14 => Ok(self.run_benchmark_for_config::<14>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            15 => Ok(self.run_benchmark_for_config::<15>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            16 => Ok(self.run_benchmark_for_config::<16>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            32 => Ok(self.run_benchmark_for_config::<32>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            34 => Ok(self.run_benchmark_for_config::<34>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            96 => Ok(self.run_benchmark_for_config::<96>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            100 => Ok(self.run_benchmark_for_config::<100>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            128 => Ok(self.run_benchmark_for_config::<128>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            256 => Ok(self.run_benchmark_for_config::<256>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            784 => Ok(self.run_benchmark_for_config::<784>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            960 => Ok(self.run_benchmark_for_config::<960>(
                node_count,
                radius,
                distribution,
                queryset,
                query_radii,
                fast,
            )?),
            _ => {
                eprintln!("Unsupported dimension: {}", dim);
                panic!("Unsupported dimension");
            }
        }
    }
}

/// Parsed contents of a CSV point file (benchmarkset or queryset).
pub struct ParsedPointFile {
    pub dimension: usize,
    pub node_count: usize,
    pub points: Vec<Vec<f32>>,
}

/// Read and parse a CSV point file using mmap + rayon for performance.
///
/// The file is memory-mapped to avoid copying into a String, and lines
/// are parsed in parallel via rayon.
pub fn parse_point_file(path: &str) -> Result<ParsedPointFile, Box<dyn std::error::Error>> {
    assert!(
        // assert that the file exists
        std::path::Path::new(path).exists(),
        "Point file does not exist: {}",
        path
    );
    let file = fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let content = std::str::from_utf8(&mmap)?;

    // Determine dimension from the first line
    let first_line = content.lines().next().ok_or("Empty point file")?;
    let dimension = first_line.split(',').count();

    // Collect line byte-slices first, then parse in parallel
    let lines: Vec<&str> = content.lines().collect();
    let node_count = lines.len();

    let points: Vec<Vec<f32>> = lines
        .par_iter()
        .map(|line| {
            line.split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f32>()
                        .expect("Failed to parse point coordinate")
                })
                .collect()
        })
        .collect();

    Ok(ParsedPointFile {
        dimension,
        node_count,
        points,
    })
}

pub fn parse_radius_file(path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    fs::read_to_string(path)
        .expect(format!("Failed to read radius file: {}", path).as_str())
        .lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .map_err(|e| format!("Failed to parse radius '{}': {}", line.trim(), e).into())
        })
        .collect()
}

pub fn hypersphere_volume_factor_recursive(d: usize) -> f64 {
    match d {
        0 => 1.0,
        1 => 2.0,
        _ => (2.0 * std::f64::consts::PI / d as f64) * hypersphere_volume_factor_recursive(d - 2),
    }
}
