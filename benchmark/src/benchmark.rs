use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::Arc,
};

use criterion::Criterion;
use rembed::{Embedding, NodeId, graph::Graph, parsing::Iterations};
use sqlx::{Pool, Postgres, Row};

pub mod perf_measurement;

pub mod runner;

use crate::code_state::RepoCodeStateManager;
use runner::{BenchmarkResult, BenchmarkType, MeasurementResult};

pub struct Testcase<'a, const D: usize> {
    pub iterations: Vec<Embedding<'a, D>>,
}

#[derive(Clone)]
pub struct LoadData {
    pub pool: Pool<Postgres>,
    pub hostname: String,
    pub repo_code_manager: RepoCodeStateManager,
    pub store: bool,
    pub allow_dirty: bool,
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
            allow_dirty: false,
        }
    }

    pub async fn run_benchmarks(
        &self,
        only_last_iteration: bool,
        n_range: (usize, usize),
        dim_range: (usize, usize),
        deg_range: (usize, usize),
        ple_range: (f64, f64),
        alpha_range: (f64, f64),
        seed_range: (usize, usize),
        wseed_range: (usize, usize),
        n_threads: usize,
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
                    conditions.push(format!("n >= {} AND n <= {}", n_range.0, n_range.1));
                }
                if deg_range.1 > 0 {
                    conditions.push(format!("deg >= {} AND deg <= {}", deg_range.0, deg_range.1));
                }
                if seed_range.1 > 0 {
                    conditions.push(format!(
                        "seed >= {} AND seed <= {}",
                        seed_range.0, seed_range.1
                    ));
                }
                if wseed_range.1 > 0 {
                    conditions.push(format!(
                        "wseed >= {} AND wseed <= {}",
                        wseed_range.0, wseed_range.1
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
        println!(
            "Found {} graphs matching parameters",
            position_results.len()
        );
        if position_results.is_empty() {
            return Ok(());
        }
        let queue = crossbeam::queue::ArrayQueue::new(position_results.len());
        for result in position_results {
            queue.push(result).unwrap();
        }
        let concurrency: usize = std::thread::available_parallelism().unwrap().into();
        let concurrency = if n_threads > 0 {
            n_threads.min(concurrency)
        } else {
            concurrency
        };
        let prog = crate::create_progress_bar(queue.len());

        let queue = Arc::new(queue);
        let mut handles = Vec::new();

        for _ in 0..concurrency {
            let queue = queue.clone();
            let benchmarks = benchmarks.clone();
            let structures = structures.clone();
            let data_directory = data_directory.clone();
            let load_data = self.clone();
            let prog = prog.clone();

            let handle = tokio::task::spawn_blocking(move || {
                let handle = tokio::runtime::Handle::current();
                let local = tokio::task::LocalSet::new();

                handle.block_on(local.run_until(async move {
                    while let Some(result) = queue.pop() {
                        if let Err(e) = load_data
                            .bench_embedding(
                                only_last_iteration,
                                &benchmarks,
                                &structures,
                                &data_directory,
                                result,
                            )
                            .await
                        {
                            println!("error while benchmarking {e}");
                        }
                        prog.inc(1);
                    }
                }));
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        futures::future::join_all(handles).await;

        Ok(())
    }

    async fn bench_embedding(
        &self,
        only_last_iteration: bool,
        benchmarks: &Option<Vec<BenchmarkType>>,
        structures: &Option<Vec<String>>,
        data_directory: &str,
        result: sqlx::postgres::PgRow,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut c = Criterion::default().with_output_color(true).without_plots();
        let pos_path: String = result.get::<String, _>("pos_path");
        let pos_path = format!("{data_directory}/{}", pos_path);
        let graph_path: String = result.get::<String, _>("graph_path");
        let graph_path = format!("{data_directory}/{}", graph_path);
        let embedding_dim: i32 = result.get("embedding_dim");
        let dim_hint: i32 = result.get("dim_hint");
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
                benchmarks,
                structures,
                load_data: self,
            },
            &mut c,
        )
        .await;
        Ok(())
    }

    async fn store_benchmark_result(
        &self,
        result: BenchmarkResult,
        checksum: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.allow_dirty && RepoCodeStateManager::git_dirty()? {
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
    async fn skip_list(
        &self,
        result_id: i64,
        code_state_id: i64,
    ) -> Result<HashSet<Measurement>, sqlx::Error> {
        // Store measurement result
        sqlx::query_as!(
            Measurement,
            r#"
                SELECT benchmark_type, iteration_number as iteration FROM measurements
                WHERE code_state_id = $1 AND result_id = $2 AND hostname = $3
                "#,
            code_state_id,
            result_id,
            self.hostname,
        )
        .fetch_all(&self.pool)
        .await
        .map(|x| x.into_iter().collect())
    }
}

#[derive(Hash, PartialEq, Eq)]
struct Measurement {
    benchmark_type: String,
    iteration: i32,
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

macro_rules! dispatch_dim {
    ($dim:ident, $args:ident, $c:ident, $($c_dim:literal,)*) => {
        match  $dim {
            $($c_dim => load_and_run::<$c_dim>($args, $c).await,)*
            _ => panic!("dim {} not covered",$dim),
        }
    };
}

async fn load_and_run_dynamic(dim: u8, args: BenchmarkArgs<'_>, c: &mut Criterion) {
    dispatch_dim!(
        dim, args, c, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32,
    )
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
        iterations.iterations().iter().map(|x| {
            (
                x.number,
                Embedding::<D> {
                    positions: x.positions.deref().clone(),
                    graph,
                },
            )
        })
    };

    let embeddings: Vec<_> = embeddings().collect();

    let embeddings = if only_last_iteration {
        &embeddings[(embeddings.len().max(2) - 2)..]
    } else {
        embeddings.as_slice()
    };
    if embeddings.is_empty() {
        println!("Empty embedding, skipping");
        return;
    }
    let mut data_structures = if let Some(structures) = structures {
        rembed::data_structures(&embeddings[0].1)
            .filter(|s| structures.contains(&s.name()))
            .collect()
    } else {
        rembed::data_structures(&embeddings[0].1).collect::<Vec<_>>()
    };
    let mut code_states = HashMap::new();
    for structure in &mut data_structures {
        if let Ok(Some(code_state)) = load_data
            .repo_code_manager
            .get_code_state(&structure.name(), &structure.checksum())
            .await
        {
            let Ok(skiplist) = load_data
                .skip_list(result_id, code_state.code_state_id)
                .await
            else {
                continue;
            };
            code_states.insert(structure.name(), (code_state, skiplist));
        }
    }

    for &(iteration, ref embedding) in embeddings {
        for structure in &mut data_structures {
            structure.update_positions(&embedding.positions);
        }
        let mut group = c.benchmark_group(format!("result_{result_id}@{iteration}_dim-{D}"));

        let process_results = |m: MeasurementResult, ty: &BenchmarkType| BenchmarkResult {
            benchmark_type: *ty,
            data_structure_name: m.data_structure_name,
            result_id,
            iteration_number: iteration,
            sample_count: m.sample_count,
            measurement: m.measurement,
        };

        let mut run_benchmark_with_query_list =
            async |query_list: Vec<_>, benchmark_type: &BenchmarkType| {
                for structure in &data_structures {
                    if load_data.store {
                        if let Some((_, skiplist)) = code_states.get(&structure.name()) {
                            if skiplist.contains(&Measurement {
                                benchmark_type: benchmark_type.as_str().to_owned(),
                                iteration: iteration as i32,
                            }) {
                                continue;
                            }
                        }
                    }
                    let result = process_results(
                        runner::profile_datastructure_query(
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
