use std::ops::Deref;

use criterion::Criterion;
use rembed::{Embedding, NodeId, graph::Graph, parsing::Iterations};
use sqlx::{Pool, Postgres};

use crate::{
    code_state::RepoCodeStateManager,
    runner::{BenchmarkResult, BenchmarkType, MeasurementResult},
};

pub struct Testcase<'a, const D: usize> {
    pub iterations: Vec<Embedding<'a, D>>,
}

pub struct LoadData {
    pub pool: Pool<Postgres>,
    pub hostname: String,
    pub repo_code_manager: RepoCodeStateManager,
}

impl LoadData {
    pub fn new(pool: Pool<Postgres>) -> Self {
        let hostname = gethostname::gethostname().to_string_lossy().to_string();
        let repo_code_manager = RepoCodeStateManager::new(pool.clone());

        LoadData {
            pool,
            hostname,
            repo_code_manager,
        }
    }

    pub async fn run_test_cases(
        &self,
        only_last_iteration: bool,
        n_range: (usize, usize),
        dim_range: (usize, usize),
        store: bool,
        benchmarks: Option<Vec<BenchmarkType>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tx = self.pool.begin().await?;

        // fetch all graphs
        let position_results = sqlx::query!(
            "SELECT position_results.file_path as pos_path, graphs.file_path as graph_path, embedding_dim, dim_hint, result_id
            FROM position_results
            JOIN graphs USING (graph_id)
            WHERE embedding_dim >= $1 AND embedding_dim <= $2
                AND processed_n >= $3 AND processed_n <= $4",
            dim_range.0 as i32,
            dim_range.1 as i32,
            n_range.0 as i64,
            n_range.1 as i64
        )
        .fetch_all(&mut *tx)
        .await?;

        let data_directory = std::env::var("DATA_DIRECTORY").unwrap_or(String::from("../data/"));

        // check if the results exist
        for result in &position_results {
            let pos_path = &format!("{data_directory}/{}", result.pos_path);
            let graph_path = &format!("{data_directory}/{}", result.graph_path);

            let path = std::path::Path::new(&pos_path);
            if !path.exists() {
                return Err(format!(
                    "File not found: {} \n Please trigger Pull via Command",
                    pos_path
                )
                .into());
            }

            let graph_path = std::path::Path::new(&graph_path);
            if !graph_path.exists() {
                return Err(format!(
                    "Graph file not found: {} \n Please trigger Pull via Command",
                    graph_path.display()
                )
                .into());
            }
        }

        let mut c = Criterion::default().with_output_color(true);

        // load embeddings from files
        for result in &position_results {
            let pos_path = &format!("{data_directory}/{}", result.pos_path);
            let graph_path = &format!("{data_directory}/{}", result.graph_path);

            // Get the graph
            let graph = rembed::graph::Graph::parse_from_edge_list_file(
                graph_path,
                result.embedding_dim as usize,
                result.dim_hint as usize,
            )
            .map_err(|e| format!("Failed to load graph from {}: {}", graph_path, e))?;

            let results = load_and_run_dynamic(
                result.embedding_dim as u8,
                BenchmarkArgs {
                    graph: &graph,
                    result_id: result.result_id,
                    embedding_path: pos_path,
                    only_last_iteration,
                    benchmarks: &benchmarks,
                },
                &mut c,
            );
            if store {
                self.store_benchmark_results(results).await?;
            }
        }

        Ok(())
    }

    async fn store_benchmark_results(
        &self,
        results: Vec<BenchmarkResult>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for result in results {
            // Get or create code state for this data structure
            let code_state = self
                .repo_code_manager
                .get_or_create_code_state(
                    &result.data_structure_name,
                    "placeholder_checksum", // TODO: Get actual checksum from Query trait
                )
                .await?;

            // Store measurement result
            sqlx::query!(
                r#"
                INSERT INTO measurements (
                    code_state_id, result_id, iteration_number, sample_count,
                    hostname, architecture, benchmark_type,
                    wall_time_mean, wall_time_stddev, 
                    instruction_count_mean, instruction_count_stddev, cycles_mean, cycles_stddev
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                "#,
                code_state.code_state_id,
                result.result_id,
                result.iteration_number as i32,
                result.sample_count as i32,
                self.hostname,
                std::env::consts::ARCH,
                result.benchmark_type.as_str(),
                result.measurement.wall_time_mean.as_nanos() as i64,
                result.measurement.wall_time_stddev.as_nanos() as i64,
                result.measurement.instructions_mean,
                result.measurement.instructions_stddev,
                result.measurement.cycles_mean,
                result.measurement.cycles_stddev,
            )
            .execute(&self.pool)
            .await?;

            println!(
                "Stored benchmark result: {} for result_id: {} iteration: {}",
                result.benchmark_type.as_str(),
                result.result_id,
                result.iteration_number
            );
        }

        Ok(())
    }
}
struct BenchmarkArgs<'a> {
    graph: &'a Graph,
    result_id: i64,
    embedding_path: &'a str,
    only_last_iteration: bool,
    benchmarks: &'a Option<Vec<BenchmarkType>>,
}

fn load_and_run_dynamic(dim: u8, args: BenchmarkArgs, c: &mut Criterion) -> Vec<BenchmarkResult> {
    match dim {
        2 => load_and_run::<2>(args, c),
        4 => load_and_run::<4>(args, c),
        8 => load_and_run::<8>(args, c),
        16 => load_and_run::<16>(args, c),
        32 => load_and_run::<32>(args, c),
        _ => panic!("dim {dim} not covered",),
    }
}

fn load_and_run<const D: usize>(args: BenchmarkArgs, c: &mut Criterion) -> Vec<BenchmarkResult> {
    let BenchmarkArgs {
        graph,
        result_id,
        embedding_path,
        only_last_iteration,
        benchmarks,
    } = args;
    let iterations: Iterations<D> = rembed::parsing::parse_positions_file(embedding_path).unwrap();

    // Load the embeddings from the file
    let embeddings = || {
        iterations.iterations().iter().map(|x| Embedding::<D> {
            positions: x.positions.deref().clone(),
            graph,
        })
    };

    assert!(only_last_iteration);
    let mut group = c.benchmark_group(format!("result_{result_id}_dim-{D}"));

    let embedding = &embeddings().next_back().unwrap();

    let structures: Vec<_> = rembed::data_structures(embedding).collect();
    let mut results = Vec::new();

    let process_results = |results: Vec<MeasurementResult>, ty: &BenchmarkType| {
        results
            .into_iter()
            .map(|m| BenchmarkResult {
                benchmark_type: *ty,
                data_structure_name: m.data_structure_name,
                result_id,
                iteration_number: iterations.iterations().last().unwrap().number,
                sample_count: m.sample_count,
                measurement: m.measurement,
            })
            .collect::<Vec<_>>()
    };

    let mut run_benchmark_with_query_list = |query_list: Vec<_>, benchmark_type: &BenchmarkType| {
        results.extend(process_results(
            crate::runner::profile_datastructure_query(
                embedding,
                &mut group,
                &structures,
                &query_list,
                *benchmark_type,
            ),
            benchmark_type,
        ));
    };

    let benchmarks = benchmarks
        .as_ref()
        .map(|x| x.as_slice())
        .unwrap_or(BenchmarkType::all());
    for benchmark in benchmarks {
        let query_list = query_list_for_type(*benchmark, embedding);
        run_benchmark_with_query_list(query_list, benchmark);
    }
    results
}

fn query_list_for_type<'a, const D: usize>(
    ty: BenchmarkType,
    embedding: &Embedding<'a, D>,
) -> Vec<NodeId> {
    match ty {
        BenchmarkType::SparseQuery => query_sparse(embedding, 1000),
        BenchmarkType::LightNodes => query_light(embedding, 1000),
        BenchmarkType::HeavyNodes => query_heavy(embedding, 1000),
        BenchmarkType::PositionUpdate => (0..embedding.positions.len()).collect(),
    }
}

fn query_sparse<'a, const D: usize>(embedding: &Embedding<'a, D>, n: usize) -> Vec<NodeId> {
    let total = embedding.positions.len();
    (0..total).step_by(total / n).collect()
}
fn query_light<'a, const D: usize>(embedding: &Embedding<'a, D>, n: usize) -> Vec<NodeId> {
    let light_nodes: Vec<_> = embedding
        .graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.weight < 1.)
        .map(|(i, _)| i)
        .collect();
    let total = light_nodes.len();
    light_nodes.into_iter().step_by(total / n).collect()
}
fn query_heavy<'a, const D: usize>(embedding: &Embedding<'a, D>, n: usize) -> Vec<NodeId> {
    let light_nodes: Vec<_> = embedding
        .graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.weight >= 1.)
        .map(|(i, _)| i)
        .collect();
    let total = light_nodes.len();
    light_nodes.into_iter().step_by(total / n).collect()
}
