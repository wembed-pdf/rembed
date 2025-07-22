use rand::seq::IteratorRandom;
use rembed::graph::Graph;
use rembed::parsing::Iteration;
use rembed::parsing::Iterations;
use rembed::parsing::parse_positions_file;

pub struct FScore {}

pub struct StatisticGenerator {}

impl StatisticGenerator {
    pub fn new() -> Self {
        StatisticGenerator {}
    }

    fn neighbors<const D: usize>(
        &self,
        node_id: usize,
        graph: &Graph,
        iteration: &Iteration<D>,
    ) -> Vec<usize> {
        iteration
            .positions
            .iter()
            .enumerate()
            .filter_map(|(j, pos)| {
                if (iteration.positions[node_id].distance_squared(pos) as f64)
                    < (graph.nodes[j].weight * graph.nodes[node_id].weight).powi(2)
                {
                    Some(j)
                } else {
                    None
                }
            })
            .collect()
    }

    fn graph_statistics<const D: usize>(
        &self,
        graph: &Graph,
        iterations: Iterations<D>,
    ) -> Vec<(f64, f64)> {
        let num_nodes = graph.nodes.len();
        let mut found_edges = 0;
        let mut missed_edges = 0;
        let mut found_non_edges = 0;
        let mut stats = Vec::new();
        let mut adjacency_list: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for edge in &graph.edges {
            adjacency_list[edge.0].push(edge.1);
            adjacency_list[edge.1].push(edge.0);
        }
        for iteration in iterations.iterations() {
            // let embedding_neighbours = self.neighbors_parallel(graph, iteration.clone());
            for (i, _) in iteration
                .positions
                .iter()
                .enumerate()
                .choose_multiple(&mut rand::rng(), 500)
            {
                let embedding_neighbours = self.neighbors(i, graph, &iteration);
                for node in embedding_neighbours.iter() {
                    if adjacency_list[i].contains(node) {
                        found_edges += 1;
                    } else {
                        found_non_edges += 1;
                    }
                }
                for neighbor in adjacency_list[i].iter() {
                    if !embedding_neighbours.contains(neighbor) {
                        missed_edges += 1;
                    }
                }
            }
            stats.push((
                found_edges as f64 / (found_edges + missed_edges) as f64,
                found_edges as f64 / (found_edges + found_non_edges) as f64,
            ));
        }
        stats
    }

    fn f1<const D: usize>(&self, graph: &Graph, position_path: &str) -> Vec<f64> {
        let iterations: Iterations<D> = parse_positions_file(position_path).unwrap();
        let stats = self.graph_statistics(graph, iterations);
        let mut f1_scores = Vec::new();

        for (precision, recall) in stats {
            f1_scores.push(2. / (recall.recip() + precision.recip()));
        }

        f1_scores
    }

    pub async fn compute_f_scores(&self, result_id: i64) -> Result<(), Box<dyn std::error::Error>> {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
        let pool = sqlx::PgPool::connect(&database_url).await?;
        let graph_info = sqlx::query!(
                "SELECT graphs.file_path as graph_path, position_results.file_path as position_path, embedding_dim, dim_hint FROM position_results join graphs USING (graph_id) WHERE result_id = $1",
            result_id
        )
        .fetch_one(&pool).await?;
        let graph = rembed::graph::Graph::parse_from_edge_list_file(
            &dbg!(format!("../data/{}", graph_info.graph_path)),
            graph_info.embedding_dim as usize,
            graph_info.dim_hint as usize,
        )?;

        let position_path = format!("../data/{}", graph_info.position_path);

        let f1_scores = match graph_info.embedding_dim {
            2 => self.f1::<2>(&graph, &position_path),
            4 => self.f1::<4>(&graph, &position_path),
            8 => self.f1::<8>(&graph, &position_path),
            16 => self.f1::<16>(&graph, &position_path),
            32 => self.f1::<32>(&graph, &position_path),
            _ => {
                println!(
                    "Skipping unsupported dimension: {}",
                    graph_info.embedding_dim
                );
                Vec::new()
            }
        };
        dbg!(&f1_scores);
        // let f1_scores_json = serde_json::to_string(&f1_scores)?;
        // sqlx::query!(
        //     "UPDATE position_results SET f1_scores = $1 WHERE result_id = $2",
        //     f1_scores_json,
        //     result_id
        // )
        // .execute(&pool)
        // .await?;
        Ok(())
    }
}
