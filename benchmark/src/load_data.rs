use std::ops::Deref;

use criterion::Criterion;
use rembed::{
    Embedding,
    graph::Graph,
    parsing::{Iteration, Iterations},
};
use sqlx::{Pool, Postgres};

pub struct Testcase<'a, const D: usize> {
    pub iterations: Vec<Embedding<'a, D>>,
}

pub struct LoadData {
    pub pool: Pool<Postgres>,
    pub hostname: String,
}

impl LoadData {
    pub fn new(pool: Pool<Postgres>) -> Self {
        // let test_cases: Vec<_> = Vec::new();
        let hostname = gethostname::gethostname().to_string_lossy().to_string();
        LoadData { pool, hostname }
    }

    pub async fn run_test_cases(
        &self,
        only_last_iteration: bool,
        n_range: (usize, usize),
        dim_range: (usize, usize),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tx = self.pool.begin().await?;

        // fetch all graphs
        let position_results = sqlx::query!(
            "SELECT position_results.file_path as pos_path, graphs.file_path as graph_path, embedding_dim, dim_hint, result_id FROM position_results JOIN graphs USING (graph_id) WHERE embedding_dim >= $1 AND embedding_dim <= $2 AND processed_n >= $3 AND processed_n <= $4",
            dim_range.0 as i32,
            dim_range.1 as i32,
            n_range.0 as i64,
            n_range.1 as i64
        )
        .fetch_all(&mut *tx)
        .await?;

        let data_directory = std::env::var("DATA_DIRECTORY").unwrap_or(String::from("../data/"));
        // check if the results exist
        for result in &position_results[..0] {
            let file_path = &result.pos_path;

            let path = std::path::Path::new(&file_path);
            if !path.exists() {
                return Err(format!(
                    "File not found: {} \n Please trigger Pull via Command",
                    file_path
                )
                .into());
            }

            let graph_path = &result.graph_path;
            let graph_path = std::path::Path::new(&graph_path);
            if !graph_path.exists() {
                return Err(format!(
                    "Graph file not found: {} \n Please trigger Pull via Command",
                    graph_path.display()
                )
                .into());
            }
        }

        let mut c = Criterion::default().sample_size(10).with_output_color(true);

        // load embeddings from files
        for result in &position_results {
            // Get the graph
            let graph = rembed::graph::Graph::parse_from_edge_list_file(
                &result.graph_path,
                result.embedding_dim as usize,
                result.dim_hint as usize,
            )
            .map_err(|e| format!("Failed to load graph from {}: {}", result.graph_path, e))?;

            load_and_run_dynamic(
                result.embedding_dim as u8,
                &graph,
                result.result_id,
                &result.pos_path,
                only_last_iteration,
                &mut c,
            );
        }

        Ok(())
    }
}

fn load_and_run_dynamic(
    dim: u8,
    graph: &Graph,
    result_id: i64,
    embedding_path: &str,
    only_last_iteration: bool,
    c: &mut Criterion,
) {
    match dim {
        2 => load_and_run::<2>(graph, result_id, embedding_path, only_last_iteration, c),
        4 => load_and_run::<4>(graph, result_id, embedding_path, only_last_iteration, c),
        8 => load_and_run::<8>(graph, result_id, embedding_path, only_last_iteration, c),
        16 => load_and_run::<16>(graph, result_id, embedding_path, only_last_iteration, c),
        32 => load_and_run::<32>(graph, result_id, embedding_path, only_last_iteration, c),
        _ => panic!("dim {dim} not covered"),
    }
}

fn load_and_run<const D: usize>(
    graph: &Graph,
    result_id: i64,
    embedding_path: &str,
    only_last_iteration: bool,
    c: &mut Criterion,
) {
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

    crate::runner::profile_datastructures(&embeddings().next_back().unwrap(), &mut group);

    // if only_last_iteration {
    //     if let Some(last_embedding) = iterations.last() {
    //         test_cases.push(Testcase {
    //             iterations: vec![last_embedding.clone()],
    //         });
    //     }
    // } else {
    //     test_cases.push(Testcase { iterations });
    // }
}
