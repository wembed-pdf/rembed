use crate::benchmark::runner;
use crate::synthetic_data::{self, PointDistribution};
use criterion::Criterion;
use memmap2::Mmap;
use rand_distr::Distribution;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rembed::{Embedding, NodeId};
use std::fs;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct DistributionBenchConfig {
    pub dims: Vec<usize>,
    pub counts: Vec<usize>,
    pub radii: Vec<f64>,
    pub distributions: Option<Vec<PointDistribution>>,
    pub benchmarksets: Option<Vec<String>>,
    pub querysets: Option<Vec<String>>,
    pub path_to_benchmarksets: Option<String>,
    pub structures: Vec<String>,
    pub num_queries: usize,
    pub seed: u64,
    pub fast: bool,
    pub expected_queries: Option<usize>,
    pub only_center_nodes: bool,
    pub parallel: bool,
    pub all_to_all: bool,
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
        let fast = self.config.fast;
        if self.config.distributions.is_some() && self.config.expected_queries.is_none() {
            eprintln!("Running distribution benchmarks:");
            eprintln!("  Dimensions: {:?}", self.config.dims);
            eprintln!("  Node counts: {:?}", self.config.counts);
            eprintln!("  Radiuses: {:?}", self.config.radii);
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
            let total = self.config.dims.len()
                * self.config.counts.len()
                * self.config.radii.len()
                * self.config.distributions.as_ref().unwrap().len();
            let mut current = 0;

            if self.config.parallel {
                return Ok(self
                    .config
                    .dims
                    .par_iter()
                    .flat_map(|&dim| {
                        self.config.counts.par_iter().map(move |&node_count| {
                            let mut temp_results = Vec::new();
                            self.run_benchmarks_for_dimension_and_nodecount(
                                &mut temp_results,
                                fast,
                                total,
                                &mut 0,
                                dim,
                                node_count,
                            )
                            .unwrap();
                            temp_results
                        })
                    })
                    .flatten()
                    .collect());
            }

            for &dim in &self.config.dims {
                for &node_count in &self.config.counts {
                    self.run_benchmarks_for_dimension_and_nodecount(
                        &mut all_results,
                        fast,
                        total,
                        &mut current,
                        dim,
                        node_count,
                    )?;
                }
            }
        }

        if self.config.distributions.is_some() && self.config.expected_queries.is_some() {
            eprintln!("Generating centered benchmarks for expected queries:");
            eprintln!("  Dimensions: {:?}", self.config.dims);
            eprintln!("  Node counts: {:?}", self.config.counts);
            eprintln!("  Expected queries: {:?}", self.config.expected_queries);
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
            let total = self.config.dims.len()
                * self.config.counts.len()
                * self.config.distributions.as_ref().unwrap().len();
            let pb = indicatif::ProgressBar::new(total as u64);

            for distribution in self.config.distributions.as_ref().unwrap() {
                eprintln!("Running distribution: {}", distribution.name());
                for &dim in &self.config.dims {
                    for &node_count in &self.config.counts {
                        let radius = (self.config.expected_queries.unwrap() as f64
                            / node_count as f64
                            / hypersphere_volume_factor_recursive(dim))
                        .powf(1.0 / dim as f64);

                        // Compute query set with distance to boundary at least radius to avoid edge effects skewing results
                        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                            self.config.seed,
                        );
                        let distribution_gen =
                            rand_distr::Uniform::new(radius as f32, 1.0 - radius as f32).unwrap();
                        let queryset = (0..self.config.num_queries)
                            .map(|_| {
                                let components = (0..dim)
                                    .map(|_| distribution_gen.sample(&mut rng))
                                    .collect::<Vec<f32>>();
                                components
                            })
                            .collect();

                        let results = self.run_benchmarks_for_dimension(
                            dim,
                            node_count,
                            radius,
                            distribution,
                            Some(queryset),
                            fast,
                        )?;
                        all_results.extend(results);
                        pb.inc(1);
                    }
                }
            }
            pb.finish_with_message("Benchmarks completed");
        }

        if let Some(benchmarksets) = &self.config.benchmarksets {
            for (i, benchmarkset) in benchmarksets.iter().enumerate() {
                if self.config.expected_queries.is_some() {
                    todo!("Implement expected_queries for benchmarksets");
                }
                for &radius in &self.config.radii {
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
                    let parsed = parse_point_file(&path)?;
                    let distribution = PointDistribution::Benchmarkset {
                        points: parsed.points.clone(),
                        name: benchmarkset.clone(),
                    };
                    let node_count = parsed.node_count;
                    let dimension = parsed.dimension;
                    assert!(
                        self.config.querysets.is_none() || !self.config.all_to_all,
                        "querysets and all_to_all options are mutually exclusive"
                    );
                    let queryset = if self.config.querysets.is_some() {
                        assert!(
                            benchmarksets.len() == self.config.querysets.as_ref().unwrap().len(),
                            "Number of querysets must match number of benchmarksets"
                        );
                        let queryset = &self.config.querysets.as_ref().unwrap()[i];
                        eprintln!(
                            "Using queryset: {} for benchmarkset: {}",
                            queryset, benchmarkset
                        );
                        let query_path = format!(
                            "{}/{}",
                            self.config.path_to_benchmarksets.as_ref().unwrap(),
                            queryset
                        );
                        Some(parse_point_file(&query_path)?.points)
                    } else if self.config.all_to_all {
                        eprintln!(
                            "Running All-To-All benchmark for benchmarkset: {}",
                            benchmarkset
                        );
                        Some(parsed.points)
                    } else {
                        None
                    };
                    let results = self.run_benchmarks_for_dimension(
                        dimension,
                        node_count,
                        radius,
                        &distribution,
                        queryset,
                        fast,
                    )?;
                    all_results.extend(results);
                }
            }
        }

        Ok(all_results)
    }

    fn run_benchmarks_for_dimension_and_nodecount(
        &self,
        all_results: &mut Vec<BenchmarkRecord>,
        fast: bool,
        total: usize,
        current: &mut i32,
        dim: usize,
        node_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(expected) = self.config.expected_queries {
            // Adjust radius to achieve expected number of queries
            let radius =
                (expected as f64 / node_count as f64 / hypersphere_volume_factor_recursive(dim))
                    .powf(1.0 / dim as f64);
            for distribution in self.config.distributions.as_ref().unwrap() {
                *current += 1;
                eprintln!(
                    "[{}/{}] Running dim={}, n={}, r~{:.3}, dist={}",
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
                    None,
                    fast,
                )?;

                all_results.extend(results);
            }
        } else {
            for &radius in &self.config.radii {
                for distribution in self.config.distributions.as_ref().unwrap() {
                    *current += 1;
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
                        None,
                        fast,
                    )?;

                    all_results.extend(results);
                }
            }
        };
        Ok(())
    }

    fn run_benchmark_for_config<const D: usize>(
        &self,
        node_count: usize,
        radius: f64,
        distribution: &PointDistribution,
        queryset: Option<Vec<Vec<f32>>>,
        fast: bool,
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

        let eligible_nodes: Vec<NodeId> = if self.config.only_center_nodes && queryset.is_none() {
            // Generate query indices, only use nodes that are not closer than radius to the boundary to avoid edge effects skewing results
            let mut reduced_elegible_nodes: Vec<NodeId> = (0..node_count)
                .filter(|&i| {
                    let pos = &embedding.positions[i];
                    pos.components
                        .iter()
                        .all(|&coord| coord >= radius as f32 && coord <= 1.0 - radius as f32)
                })
                .collect();
            eprintln!(
                "Generated {} eligible query nodes out of {} total nodes for radius {}",
                reduced_elegible_nodes.len(),
                node_count,
                radius
            );
            if reduced_elegible_nodes.is_empty() {
                // order the nodes by distance to the boundary
                let mut nodes_with_dist: Vec<(NodeId, f32)> = (0..node_count)
                    .map(|i| {
                        let pos = &embedding.positions[i];
                        let dist_to_boundary = pos
                            .components
                            .iter()
                            .map(|&coord| coord.min(1.0 - coord))
                            .fold(f32::INFINITY, |a, b| a.min(b));
                        (i, dist_to_boundary)
                    })
                    .collect();
                nodes_with_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                reduced_elegible_nodes = nodes_with_dist.into_iter().map(|(i, _)| i).collect();
            }
            reduced_elegible_nodes
        } else {
            (0..node_count).collect()
        };
        assert!(
            !eligible_nodes.is_empty(),
            "No eligible query nodes found. Are there any nodes? only_center_nodes should not be able to cause this.",
        );
        let query_indices: Vec<NodeId> = if eligible_nodes.len() <= self.config.num_queries {
            eligible_nodes
        } else {
            let step = eligible_nodes.len() / self.config.num_queries;
            eligible_nodes.iter().step_by(step).copied().collect()
        };

        // Benchmark each structure
        let mut results = Vec::new();
        // let mut c = Criterion::default().without_plots();
        let mut c = Criterion::default();
        let mut group = c.benchmark_group(format!("dist_bench_d{}_n{}", D, node_count));

        let query_pos_list = if queryset.is_some() {
            Some(
                queryset
                    .unwrap()
                    .iter()
                    .map(|q| {
                        assert_eq!(
                            q.len(),
                            D,
                            "Query point dimension does not match embedding dimension"
                        );
                        rembed::dvec::DVec::<D> {
                            components: q
                                .clone()
                                .try_into()
                                .expect("Failed to convert query point"),
                        }
                    })
                    .collect(),
            )
        } else {
            None
        };

        for structure in &data_structures {
            let measurement = runner::profile_datastructure_query(
                &embedding,
                &mut group,
                &query_indices,
                query_pos_list.clone(),
                Some(radius as f64),
                runner::BenchmarkType::Radius(radius as f32, format!("radius_{}", radius)),
                structure.as_ref(),
                fast,
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
        queryset: Option<Vec<Vec<f32>>>,
        fast: bool,
    ) -> Result<Vec<BenchmarkRecord>, Box<dyn std::error::Error>> {
        match dim {
            2 => Ok(self.run_benchmark_for_config::<2>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            3 => Ok(self.run_benchmark_for_config::<3>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            4 => Ok(self.run_benchmark_for_config::<4>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            5 => Ok(self.run_benchmark_for_config::<5>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            6 => Ok(self.run_benchmark_for_config::<6>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            7 => Ok(self.run_benchmark_for_config::<7>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            8 => Ok(self.run_benchmark_for_config::<8>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            9 => Ok(self.run_benchmark_for_config::<9>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            10 => Ok(self.run_benchmark_for_config::<10>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            11 => Ok(self.run_benchmark_for_config::<11>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            12 => Ok(self.run_benchmark_for_config::<12>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            13 => Ok(self.run_benchmark_for_config::<13>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            14 => Ok(self.run_benchmark_for_config::<14>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            15 => Ok(self.run_benchmark_for_config::<15>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            16 => Ok(self.run_benchmark_for_config::<16>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            32 => Ok(self.run_benchmark_for_config::<32>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            34 => Ok(self.run_benchmark_for_config::<34>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            96 => Ok(self.run_benchmark_for_config::<96>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            100 => Ok(self.run_benchmark_for_config::<100>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            128 => Ok(self.run_benchmark_for_config::<128>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            256 => Ok(self.run_benchmark_for_config::<256>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            784 => Ok(self.run_benchmark_for_config::<784>(
                node_count,
                radius,
                distribution,
                queryset,
                fast,
            )?),
            960 => Ok(self.run_benchmark_for_config::<960>(
                node_count,
                radius,
                distribution,
                queryset,
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
                .map(|s| s.trim().parse::<f32>().expect("Failed to parse point coordinate"))
                .collect()
        })
        .collect();

    Ok(ParsedPointFile {
        dimension,
        node_count,
        points,
    })
}

pub fn hypersphere_volume_factor_recursive(d: usize) -> f64 {
    match d {
        0 => 1.0,
        1 => 2.0,
        _ => (2.0 * std::f64::consts::PI / d as f64) * hypersphere_volume_factor_recursive(d - 2),
    }
}
