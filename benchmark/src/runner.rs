use std::str::FromStr;
use std::time::Duration;

use crate::perf_measurement::{PerfMeasurements, PerfStatistics};
use criterion::{BenchmarkGroup, measurement::WallTime};
use rembed::{Embedding, NodeId, query::IndexClone};

#[derive(Debug, Clone, Copy)]
pub enum BenchmarkType {
    PositionUpdate,
    MixedNodes,
    LightNodes,
    HeavyNodes,
    AllNodes,
}

impl BenchmarkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BenchmarkType::PositionUpdate => "position_update",
            BenchmarkType::MixedNodes => "mixed_nodes",
            BenchmarkType::LightNodes => "light_nodes",
            BenchmarkType::HeavyNodes => "heavy_nodes",
            BenchmarkType::AllNodes => "all_nodes",
        }
    }

    pub(crate) fn all() -> &'static [BenchmarkType] {
        &[
            BenchmarkType::MixedNodes,
            BenchmarkType::PositionUpdate,
            BenchmarkType::LightNodes,
            BenchmarkType::HeavyNodes,
            BenchmarkType::AllNodes,
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

pub fn profile_datastructure_query<'a, const D: usize>(
    embedding: &Embedding<'a, D>,
    c: &mut BenchmarkGroup<WallTime>,
    data_structures: &[Box<dyn IndexClone<D> + 'a>],
    query_list: &[NodeId],
    benchmark_type: BenchmarkType,
) -> Vec<MeasurementResult> {
    let mut results = Vec::with_capacity(data_structures.len());
    for structure in data_structures {
        let mut samples = PerfMeasurements::new(1000);
        let warmup = Duration::from_secs(1);
        let measure = Duration::from_secs(8);
        let queries = query_list.len();
        let sample_count = 10;
        c.warm_up_time(warmup);
        c.measurement_time(measure);
        c.sampling_mode(criterion::SamplingMode::Auto);
        c.sample_size(sample_count);
        let benchmark_id = format!("{}/{}", benchmark_type.as_str(), structure.name());

        c.bench_with_input(benchmark_id, &structure.name(), |b, _| {
            b.iter_custom(|iters| {
                let data_structures: Vec<_> = (0..iters).map(|_| structure.clone_box()).collect();
                samples.start();
                for mut structure in data_structures {
                    match benchmark_type {
                        BenchmarkType::PositionUpdate => {
                            structure.update_positions(&embedding.positions);
                        }
                        _ => {
                            for &i in query_list {
                                let result = structure.nearest_neighbors(i, 1.);
                                std::hint::black_box(result);
                            }
                        }
                    }
                }
                samples.stop(iters) / queries as u32
            });
        });

        let statistics = samples.get_statistics(queries, warmup);

        println!(
            "Perf Counter:\n\tInstructions: {} σ: {}",
            format_number(statistics.instructions_mean),
            format_number(statistics.instructions_stddev)
        );
        println!(
            "\tCycles: {} σ: {}\n",
            format_number(statistics.cycles_mean),
            format_number(statistics.cycles_stddev)
        );
        results.push(MeasurementResult {
            data_structure_name: structure.name(),
            sample_count: samples.num_samples(),
            measurement: statistics,
        })
    }
    results
}

fn format_number(num: f64) -> String {
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
