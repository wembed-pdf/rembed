use std::str::FromStr;
use std::time::Duration;

use super::perf_measurement::{PerfMeasurements, PerfStatistics};
use criterion::{BenchmarkGroup, measurement::WallTime};
use rembed::{Embedding, NodeId, query::IndexClone};

#[derive(Debug, Clone)]
pub enum BenchmarkType {
    PositionUpdate,
    MixedNodes,
    LightNodes,
    HeavyNodes,
    AllNodes,
    Radius(f32, String),
}

impl BenchmarkType {
    pub fn as_str(&self) -> &str {
        match self {
            BenchmarkType::PositionUpdate => "position_update",
            BenchmarkType::MixedNodes => "mixed_nodes",
            BenchmarkType::LightNodes => "light_nodes",
            BenchmarkType::HeavyNodes => "heavy_nodes",
            BenchmarkType::AllNodes => "all_nodes",
            BenchmarkType::Radius(_, reference) => reference.as_str(),
        }
    }

    pub(crate) fn all() -> &'static [BenchmarkType] {
        &[
            BenchmarkType::MixedNodes,
            BenchmarkType::LightNodes,
            BenchmarkType::HeavyNodes,
            // BenchmarkType::PositionUpdate,
            // BenchmarkType::AllNodes,
        ]
    }
}

impl FromStr for BenchmarkType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "construction" => BenchmarkType::PositionUpdate,
            "mixed_nodes" => BenchmarkType::MixedNodes,
            "light_nodes" => BenchmarkType::LightNodes,
            "heavy_nodes" => BenchmarkType::HeavyNodes,
            "all_nodes" => BenchmarkType::AllNodes,
            s if s.starts_with("radius_") => {
                let parts: Vec<&str> = s.splitn(2, '_').collect();
                if parts.len() != 2 {
                    return Err("Invalid radius benchmark format".to_string());
                }
                let radius = parts[1]
                    .parse::<f32>()
                    .map_err(|_| "Invalid radius value".to_string())?;
                BenchmarkType::Radius(radius, s.to_string())
            }
            _ => return Err("Invalid benchmark type".to_string()),
        })
    }
}

pub struct BenchmarkResult {
    pub benchmark_type: BenchmarkType,
    pub data_structure_name: String,
    pub result_id: i64,
    pub iteration_number: usize,
    pub sample_count: usize,
    pub measurement: PerfStatistics,
}
pub struct MeasurementResult {
    pub data_structure_name: String,
    pub sample_count: usize,
    pub measurement: PerfStatistics,
}

pub fn profile_datastructures<'a, const D: usize>(
    embedding: &Embedding<'a, D>,
    c: &mut BenchmarkGroup<WallTime>,
    data_structures: &[Box<dyn IndexClone<D> + 'a>],
    query_list: &[NodeId],
    benchmark_type: BenchmarkType,
    fast: bool,
) -> Vec<MeasurementResult> {
    let mut results = Vec::with_capacity(data_structures.len());
    for structure in data_structures {
        results.push(profile_datastructure_query(
            embedding,
            c,
            query_list,
            None,
            None,
            benchmark_type.clone(),
            structure.as_ref(),
            fast,
        ));
    }
    results
}

pub fn profile_datastructure_query<'a, const D: usize>(
    embedding: &Embedding<'a, D>,
    c: &mut BenchmarkGroup<WallTime>,
    query_list: &[usize],
    query_pos_list: Option<Vec<rembed::dvec::DVec<D>>>,
    radius: Option<f64>,
    benchmark_type: BenchmarkType,
    structure: &(dyn IndexClone<D> + 'a),
    fast: bool,
) -> MeasurementResult {
    let mut samples = PerfMeasurements::new(1000);
    let mut warmup = Duration::from_secs(3);
    let mut measure = Duration::from_secs(20);
    if fast {
        warmup = Duration::from_secs(1);
        measure = Duration::from_secs(5);
    }
    let sample_count = 10;
    c.warm_up_time(warmup);
    c.measurement_time(measure);
    c.sampling_mode(criterion::SamplingMode::Auto);
    c.sample_size(sample_count);
    let benchmark_id = format!("{}/{}", benchmark_type.as_str(), structure.name());

    let queries = if query_pos_list.is_some() {
        query_pos_list.as_ref().unwrap().len()
    } else {
        query_list.len()
    };
    if let Some(query_pos_list) = query_pos_list {
        println!(
            "Running benchmark '{}' with {} queries",
            benchmark_id,
            query_pos_list.len()
        );
        c.bench_with_input(benchmark_id, &structure.name(), |b, _| {
            b.iter_custom(|iters| {
                let data_structures: Vec<_> = (0..iters).map(|_| structure.clone_box()).collect();
                let structure = structure.clone_box();
                let mut results = Vec::with_capacity(structure.num_nodes());
                samples.start();
                // for _ in 0..iters {
                for mut structure in data_structures {
                    match benchmark_type {
                        BenchmarkType::PositionUpdate => {
                            structure.update_positions(&embedding.positions, None);
                        }
                        _ => {
                            for &pos in &query_pos_list {
                                results.clear();
                                structure.query_radius(
                                    pos,
                                    radius.expect("Radius must be provided for queryset benchmarks")
                                        as f64,
                                    &mut results,
                                );
                                std::hint::black_box(&results);
                            }
                        }
                    }
                }
                let sample_bench = samples.stop(iters) / queries as u32;
                sample_bench
            });
        });
    } else {
        c.bench_with_input(benchmark_id, &structure.name(), |b, _| {
            b.iter_custom(|iters| {
                // let data_structures: Vec<_> = (0..iters).map(|_| structure.clone_box()).collect();
                let mut structure = structure.clone_box();
                let mut results = Vec::with_capacity(structure.num_nodes());
                samples.start();
                for _ in 0..iters {
                    // for mut structure in data_structures {
                    match benchmark_type {
                        BenchmarkType::PositionUpdate => {
                            structure.update_positions(&embedding.positions, None);
                        }
                        _ => {
                            for &i in query_list {
                                results.clear();
                                structure.nearest_neighbors(i, 1., &mut results);
                                std::hint::black_box(&results);
                            }
                        }
                    }
                }
                let sample_bench = samples.stop(iters) / queries as u32;
                sample_bench
            });
        });
    }

    let statistics = samples.get_statistics(queries, warmup);

    eprintln!(
        "Perf Counter:\n\tInstructions: {} σ: {}",
        format_number(statistics.instructions_mean),
        format_number(statistics.instructions_stddev)
    );
    eprintln!(
        "\tCycles: {} σ: {}\n",
        format_number(statistics.cycles_mean),
        format_number(statistics.cycles_stddev)
    );
    if let (Some(mean), Some(stddev)) = (statistics.ref_cycles_mean, statistics.ref_cycles_stddev) {
        eprintln!(
            "\tRef Cycles: {} σ: {}\n",
            format_number(mean),
            format_number(stddev)
        );
    }
    MeasurementResult {
        data_structure_name: structure.name(),
        sample_count: samples.num_samples(),
        measurement: statistics,
    }
}

pub fn format_number(num: f64) -> String {
    if num < 1000. {
        format!("{:.2}", num)
    } else if num < 1_000_000. {
        format!("{:.1}K", num / 1000.)
    } else if num < 1_000_000_000. {
        format!("{:.1}M", num / 1_000_000.)
    } else {
        format!("{:.1}G", num / 1_000_000_000.)
    }
}
