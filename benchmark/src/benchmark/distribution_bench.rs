use crate::benchmark::runner;
use crate::synthetic_data::{self, PointDistribution};
use criterion::Criterion;
use rembed::{Embedding, NodeId};
use std::fs;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct DistributionBenchConfig {
    pub dim_range: Option<(usize, usize)>,
    pub count_range: Option<(usize, usize)>,
    pub radius_range: (f64, f64),
    pub distributions: Option<Vec<PointDistribution>>,
    pub benchmarksets: Option<Vec<String>>,
    pub path_to_benchmarksets: Option<String>,
    pub structures: Vec<String>,
    pub num_queries: usize,
    pub seed: u64,
}

impl DistributionBenchConfig {
    pub fn expand_dimensions(&self) -> Vec<usize> {
        vec![2, 3, 4, 8, 16, 32]
            .into_iter()
            .filter(|&d| d >= self.dim_range.unwrap().0 && d <= self.dim_range.unwrap().1)
            .collect()
    }

    pub fn expand_node_counts(&self) -> Vec<usize> {
        // Generate powers of 10 within range
        vec![100, 1000, 10000, 100000, 1000000]
            .into_iter()
            .filter(|&n| n >= self.count_range.unwrap().0 && n <= self.count_range.unwrap().1)
            .collect()
    }

    pub fn expand_radiuses(&self) -> Vec<f64> {
        if self.radius_range.0 == self.radius_range.1 {
            return vec![self.radius_range.0];
        }
        // Generate log-spaced radiuses
        vec![0.1, 0.25, 0.5, 1.0]
            .into_iter()
            .filter(|&r| r >= self.radius_range.0 && r <= self.radius_range.1)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkRecord {
    pub dimension: usize,
    pub node_count: usize,
    pub radius: f64,
    pub name: String,
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
        let radiuses = self.config.expand_radiuses();
        if self.config.distributions.is_some() {
            assert!(
                self.config.dim_range.is_some() && self.config.count_range.is_some(),
                "When specifying distributions, dim_range and count_range must also be specified"
            );
            let dimensions = self.config.expand_dimensions();
            let node_counts = self.config.expand_node_counts();

            eprintln!("Running distribution benchmarks:");
            eprintln!("  Dimensions: {:?}", dimensions);
            eprintln!("  Node counts: {:?}", node_counts);
            eprintln!("  Radiuses: {:?}", radiuses);
            eprintln!(
                "  Distributions: {:?}",
                self.config
                    .distributions
                    .clone()
                    .unwrap()
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>()
            );
            eprintln!();
            let total = dimensions.len()
                * node_counts.len()
                * radiuses.len()
                * self.config.distributions.as_ref().unwrap().len();
            let mut current = 0;

            for &dim in &dimensions {
                for &node_count in &node_counts {
                    for &radius in &radiuses {
                        for distribution in self.config.distributions.as_ref().unwrap() {
                            current += 1;
                            eprintln!(
                                "[{}/{}] Running dim={}, n={}, r={}, dist={}",
                                current,
                                total,
                                dim,
                                node_count,
                                radius,
                                distribution.name()
                            );

                            let results = self.run_benchmarks_for_dimension(
                                dim,
                                node_count,
                                radius,
                                distribution,
                            )?;

                            all_results.extend(results);
                        }
                    }
                }
            }
        }

        if let Some(benchmarksets) = &self.config.benchmarksets {
            for benchmarkset in benchmarksets {
                for &radius in &radiuses {
                    assert!(
                        self.config.path_to_benchmarksets.is_some(),
                        "path_to_benchmarksets must be specified when using benchmarksets"
                    );
                    let path = format!(
                        "{}/{}",
                        self.config.path_to_benchmarksets.as_ref().unwrap(),
                        benchmarkset
                    );
                    eprintln!("Running benchmarkset: {} at {}", benchmarkset, path);
                    let distribution = PointDistribution::Benchmarkset {
                        path,
                        name: benchmarkset.clone(),
                    };
                    let path = format!(
                        "{}/{}",
                        self.config.path_to_benchmarksets.as_ref().unwrap(),
                        benchmarkset
                    );
                    let node_count = fs::read_to_string(path.clone())?.lines().count();
                    let dimension = fs::read_to_string(path)?
                        .lines()
                        .next()
                        .unwrap()
                        .split(',')
                        .count();
                    let results = self.run_benchmarks_for_dimension(
                        dimension,
                        node_count,
                        radius,
                        &distribution,
                    )?;
                    all_results.extend(results);
                }
            }
        }

        Ok(all_results)
    }

    fn run_benchmark_for_config<const D: usize>(
        &self,
        node_count: usize,
        radius: f64,
        distribution: &PointDistribution,
    ) -> Result<Vec<BenchmarkRecord>, Box<dyn std::error::Error>> {
        // Generate synthetic points
        let points =
            synthetic_data::generate_points::<D>(node_count, distribution, self.config.seed);

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

        // Update positions for all structures
        for structure in &mut data_structures {
            structure.update_positions(&embedding.positions, None);
        }

        // Generate query indices
        let query_indices: Vec<NodeId> = if node_count <= self.config.num_queries {
            (0..node_count).collect()
        } else {
            let step = node_count / self.config.num_queries;
            (0..node_count).step_by(step).collect()
        };

        // Benchmark each structure
        let mut results = Vec::new();
        // let mut c = Criterion::default().without_plots();
        let mut c = Criterion::default();
        let mut group = c.benchmark_group(format!("dist_bench_d{}_n{}", D, node_count));

        for structure in &data_structures {
            let measurement = runner::profile_datastructure_query(
                &embedding,
                &mut group,
                &query_indices,
                runner::BenchmarkType::MixedNodes,
                structure.as_ref(),
            );

            results.push(BenchmarkRecord {
                dimension: D,
                node_count,
                radius,
                name: distribution.name().to_string(),
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
            "dimension,node_count,radius,distribution,structure,wall_time_ns,wall_time_std_ns,instructions,instructions_std,cycles,cycles_std,samples"
        )?;

        // Write data rows
        for record in results {
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{}",
                record.dimension,
                record.node_count,
                record.radius,
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

    fn run_benchmarks_for_dimension(
        &self,
        dim: usize,
        node_count: usize,
        radius: f64,
        distribution: &PointDistribution,
    ) -> Result<Vec<BenchmarkRecord>, Box<dyn std::error::Error>> {
        match dim {
            2 => Ok(self.run_benchmark_for_config::<2>(node_count, radius, distribution)?),
            3 => Ok(self.run_benchmark_for_config::<3>(node_count, radius, distribution)?),
            4 => Ok(self.run_benchmark_for_config::<4>(node_count, radius, distribution)?),
            8 => Ok(self.run_benchmark_for_config::<8>(node_count, radius, distribution)?),
            16 => Ok(self.run_benchmark_for_config::<16>(node_count, radius, distribution)?),
            32 => Ok(self.run_benchmark_for_config::<32>(node_count, radius, distribution)?),
            _ => {
                eprintln!("Unsupported dimension: {}", dim);
                panic!("Unsupported dimension");
            }
        }
    }
}
