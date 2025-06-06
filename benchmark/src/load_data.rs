use rayon::result;
use rembed::Embedding;
use sqlx::{Pool, Postgres};

pub struct Testcase<'a, const D: usize> {
    pub iterations: Vec<Embedding<'a, D>>,
}

pub struct LoadData<const D: usize> {
    pool: Pool<Postgres>,
    hostname: String,
}

impl<const D: usize> LoadData<D> {
    pub fn new(pool: Pool<Postgres>) -> Self {
        // let test_cases: Vec<_> = Vec::new();
        let hostname = gethostname::gethostname().to_string_lossy().to_string();
        LoadData { pool, hostname }
    }

    pub async fn load_testcase(
        &self,
        only_last_iteration: bool,
        n_range: (usize, usize),
        dim_range: (usize, usize),
    ) -> Result<Vec<Testcase<D>>, Box<dyn std::error::Error>> {
        let test_cases = Vec::new();

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

        // check if the results exist
        for result in position_results {
            let file_path = result.file_path;
            let result_id = result.result_id;

            let path = std::path::Path::new(&file_path);
            if !path.exists() {
                return Err(format!(
                    "File not found: {} \n Please trigger Pull via Command",
                    file_path
                )
                .into());
            }

            let graph_path = result.graph_path;
            let graph_path = std::path::Path::new(&graph_path);
            if !graph_path.exists() {
                return Err(format!(
                    "Graph file not found: {} \n Please trigger Pull via Command",
                    graph_path.display()
                )
                .into());
            }
        }

        // load embeddings from files
        for result in position_results {
            // Get the graph
            let graph = rembed::load_graph(
                &result.graph_path,
                result.embedding_dim as usize,
                result.dim_hint as usize,
            )
            .map_err(|e| format!("Failed to load graph from {}: {}", result.graph_path, e))?;

            // Load the embeddings from the file
            let embeddings: Vec<Embedding<result.embedding_dim>> = iterations
                .iter()
                .map(|x| Embedding {
                    positions: x.coordinates().collect(),
                    graph: &graph,
                })
                .collect();

            if only_last_iteration {
                if let Some(last_embedding) = iterations.last() {
                    test_cases.push(Testcase {
                        iterations: vec![last_embedding.clone()],
                    });
                }
            } else {
                test_cases.push(Testcase { iterations });
            }
        }

        Ok(test_cases)
    }
}
