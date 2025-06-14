use std::ops::Deref;

use criterion::Criterion;
use rembed::{Embedding, NodeId, graph::Graph, parsing::Iterations};
use sqlx::{Pool, Postgres, Row};

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
    pub store: bool,
}

impl LoadData {
    pub fn new(pool: Pool<Postgres>) -> Self {
        let hostname = gethostname::gethostname().to_string_lossy().to_string();
        let repo_code_manager = RepoCodeStateManager::new(pool.clone());

        LoadData {
            pool,
            hostname,
            repo_code_manager,
            store: false,
        }
    }

    pub async fn run_test_cases(
        &self,
        only_last_iteration: bool,
        n_range: (usize, usize),
        dim_range: (usize, usize),
        deg_range: (usize, usize),
        ple_range: (f64, f64),
        alpha_range: (f64, f64),
        benchmarks: Option<Vec<BenchmarkType>>,
        structures: Option<Vec<String>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tx = self.pool.begin().await?;

        let query =
            "SELECT position_results.file_path as pos_path, graphs.file_path as graph_path, 
                   embedding_dim, dim_hint, result_id
            FROM position_results
            JOIN graphs USING (graph_id)";

        let position_results = {
            if n_range.1 > 0
                || dim_range.1 > 0
                || deg_range.1 > 0
                || ple_range.1 > 0.0
                || alpha_range.1 > 0.0
            {
                let mut conditions = vec![];
                if dim_range.1 > 0 {
                    conditions.push(format!(
                        "embedding_dim >= {} AND embedding_dim <= {}",
                        dim_range.0, dim_range.1
                    ));
                }
                if n_range.1 > 0 {
                    conditions.push(format!(
                        "processed_n >= {} AND processed_n <= {}",
                        n_range.0, n_range.1
                    ));
                }
                if deg_range.1 > 0 {
                    conditions.push(format!(
                        "degree >= {} AND degree <= {}",
                        deg_range.0, deg_range.1
                    ));
                }
                if ple_range.1 > 0.0 {
                    conditions.push(format!("ple >= {} AND ple <= {}", ple_range.0, ple_range.1));
                }
                if alpha_range.1 > 0.0 {
                    conditions.push(format!(
                        "alpha >= '{}' AND alpha <= '{}'", //supports infinity
                        alpha_range.0, alpha_range.1
                    ));
                }

                let condition_str = conditions.join(" AND ");
                let full_query = format!("{} WHERE {}", query, condition_str);
                sqlx::query(&full_query).fetch_all(&mut *tx).await?
            } else {
                sqlx::query(query).fetch_all(&mut *tx).await?
            }
        };

        let data_directory = std::env::var("DATA_DIRECTORY").unwrap_or(String::from("../data/"));

        // check if the results exist
        for result in &position_results {
            let pos_path: String = result.get::<String, _>("pos_path");
            let pos_path = format!("{data_directory}/{}", pos_path);
            let graph_path: String = result.get::<String, _>("graph_path");
            let graph_path = format!("{data_directory}/{}", graph_path);

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
        for result in position_results {
            let pos_path: String = result.get::<String, _>("pos_path");
            let pos_path = format!("{data_directory}/{}", pos_path);
            let graph_path: String = result.get::<String, _>("graph_path");
            let graph_path = format!("{data_directory}/{}", graph_path);
            let embedding_dim: i32 = result.get("embedding_dim");
            let dim_hint: i32 = result.get("dim_hint");

            // Get the graph
            let graph = rembed::graph::Graph::parse_from_edge_list_file(
                &graph_path,
                embedding_dim as usize,
                dim_hint as usize,
            )
            .map_err(|e| format!("Failed to load graph from {}: {}", graph_path, e))?;

            load_and_run_dynamic(
                embedding_dim as u8,
                BenchmarkArgs {
                    graph: &graph,
                    result_id: result.get("result_id"),
                    embedding_path: &pos_path,
                    only_last_iteration,
                    benchmarks: &benchmarks,
                    structures: &structures,
                    load_data: self,
                },
                &mut c,
            )
            .await;
        }

        Ok(())
    }

    async fn store_benchmark_result(
        &self,
        result: BenchmarkResult,
        checksum: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if RepoCodeStateManager::git_dirty()? {
            return Err(
                "Repository is dirty Please commit changes before sumbitting a run"
                    .to_string()
                    .into(),
            );
        }
        // Get or create code state for this data structure
        let code_state = self
            .repo_code_manager
            .get_or_create_code_state(&result.data_structure_name, checksum)
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

        Ok(())
    }
}
struct BenchmarkArgs<'a> {
    graph: &'a Graph,
    result_id: i64,
    embedding_path: &'a str,
    only_last_iteration: bool,
    benchmarks: &'a Option<Vec<BenchmarkType>>,
    structures: &'a Option<Vec<String>>,
    load_data: &'a LoadData,
}

async fn load_and_run_dynamic(dim: u8, args: BenchmarkArgs<'_>, c: &mut Criterion) {
    match dim {
        2 => load_and_run::<2>(args, c).await,
        4 => load_and_run::<4>(args, c).await,
        8 => load_and_run::<8>(args, c).await,
        16 => load_and_run::<16>(args, c).await,
        32 => load_and_run::<32>(args, c).await,
        _ => panic!("dim {dim} not covered",),
    }
}

async fn load_and_run<const D: usize>(args: BenchmarkArgs<'_>, c: &mut Criterion) {
    let BenchmarkArgs {
        graph,
        result_id,
        embedding_path,
        only_last_iteration,
        benchmarks,
        structures,
        load_data,
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

    let data_structures = if let Some(structures) = structures {
        rembed::data_structures(embedding)
            .filter(|s| structures.contains(&s.name()))
            .collect()
    } else {
        rembed::data_structures(embedding).collect::<Vec<_>>()
    };

    let process_results = |m: MeasurementResult, ty: &BenchmarkType| BenchmarkResult {
        benchmark_type: *ty,
        data_structure_name: m.data_structure_name,
        result_id,
        iteration_number: iterations.iterations().last().unwrap().number,
        sample_count: m.sample_count,
        measurement: m.measurement,
    };

    let mut run_benchmark_with_query_list =
        async |query_list: Vec<_>, benchmark_type: &BenchmarkType| {
            for structure in &data_structures {
                if load_data.store {
                    if let Ok(Some(_)) = load_data
                        .repo_code_manager
                        .get_code_state(&structure.name(), &structure.checksum())
                        .await
                    {
                        println!("skipping previously recorded run");
                        continue;
                    }
                }
                let result = process_results(
                    crate::runner::profile_datastructure_query(
                        embedding,
                        &mut group,
                        &query_list,
                        *benchmark_type,
                        structure,
                    ),
                    benchmark_type,
                );
                if load_data.store {
                    let result = load_data
                        .store_benchmark_result(result, &structure.checksum())
                        .await;
                    if let Err(e) = result {
                        println!("encontered error while storing results {e}");
                    }
                }
            }
        };

    let benchmarks = benchmarks
        .as_ref()
        .map(|x| x.as_slice())
        .unwrap_or(BenchmarkType::all());
    for benchmark in benchmarks {
        let query_list = query_list_for_type(*benchmark, embedding);
        run_benchmark_with_query_list(query_list, benchmark).await;
    }
}

fn query_list_for_type<'a, const D: usize>(
    ty: BenchmarkType,
    embedding: &Embedding<'a, D>,
) -> Vec<NodeId> {
    match ty {
        BenchmarkType::MixedNodes => query_sparse(embedding, 1000),
        BenchmarkType::LightNodes => query_light(embedding, 1000),
        BenchmarkType::AllNodes => query_sparse(embedding, embedding.positions.len()),
        BenchmarkType::HeavyNodes => query_heavy(embedding, 1000),
        BenchmarkType::PositionUpdate => (0..embedding.positions.len()).collect(),
    }
}

fn query_sparse<'a, const D: usize>(embedding: &Embedding<'a, D>, n: usize) -> Vec<NodeId> {
    let total = embedding.positions.len();
    (0..total).step_by(total / n.min(total)).collect()
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
    light_nodes
        .into_iter()
        .step_by(total / n.min(total))
        .collect()
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
    light_nodes
        .into_iter()
        .step_by(total / n.min(total))
        .collect()
}
