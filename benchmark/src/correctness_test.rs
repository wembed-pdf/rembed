use chrono::{DateTime, Utc};
use rembed::query::SpatialIndex;
use rembed::{NodeId, Query, convert_to_embeddings, data_structures};
use sqlx::{Pool, Postgres};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TestRecord {
    pub result_id: i64,
    pub file_path: String,
    pub created_at: DateTime<Utc>,
}

pub struct CorrectnessTestManager {
    pool: Pool<Postgres>,
    data_directory: String,
}

impl CorrectnessTestManager {
    pub fn new(pool: Pool<Postgres>) -> Self {
        let data_directory = std::env::var("DATA_DIRECTORY").unwrap_or("../data/".to_string());
        Self {
            pool,
            data_directory,
        }
    }

    /// Generate test file for a specific result_id
    pub async fn generate_test(&self, result_id: i64) -> Result<(), Box<dyn std::error::Error>> {
        // Get result info from database
        let result = sqlx::query!(
            "SELECT pr.*, g.file_path as graph_path FROM position_results pr 
             JOIN graphs g USING (graph_id) WHERE result_id = $1",
            result_id
        )
        .fetch_one(&self.pool)
        .await?;

        let pos_path = format!("{}/{}", self.data_directory, result.file_path);
        let graph_path = format!("{}/{}", self.data_directory, result.graph_path);

        // Load graph and positions
        let graph = rembed::graph::Graph::parse_from_edge_list_file(
            &graph_path,
            result.embedding_dim as usize,
            result.dim_hint as usize,
        )?;

        let iterations = match result.embedding_dim {
            2 => self.generate_test_dynamic::<2>(&graph, &pos_path).await?,
            4 => self.generate_test_dynamic::<4>(&graph, &pos_path).await?,
            8 => self.generate_test_dynamic::<8>(&graph, &pos_path).await?,
            16 => self.generate_test_dynamic::<16>(&graph, &pos_path).await?,
            32 => self.generate_test_dynamic::<32>(&graph, &pos_path).await?,
            _ => {
                return Err(
                    format!("Unsupported embedding dimension: {}", result.embedding_dim).into(),
                );
            }
        };

        // Generate test file path
        let test_filename = format!("test_result_{}.bin", result_id);
        let test_file_path = format!("generated/tests/{}", test_filename);
        let full_test_path = format!("{}/{}", self.data_directory, test_file_path);

        // Ensure directory exists
        if let Some(parent) = Path::new(&full_test_path).parent() {
            create_dir_all(parent)?;
        }

        // Write binary test file
        self.write_test_file(&full_test_path, &iterations)?;

        // Store in database
        sqlx::query!(
            "INSERT INTO tests (result_id, file_path) VALUES ($1, $2) 
             ON CONFLICT (result_id) DO UPDATE SET file_path = $2",
            result_id,
            test_file_path
        )
        .execute(&self.pool)
        .await?;

        println!(
            "Generated test file for result_id {}: {}",
            result_id, test_file_path
        );
        Ok(())
    }

    async fn generate_test_dynamic<const D: usize>(
        &self,
        graph: &rembed::graph::Graph,
        pos_path: &str,
    ) -> Result<Vec<Vec<Vec<NodeId>>>, Box<dyn std::error::Error>> {
        let iterations: rembed::parsing::Iterations<D> =
            rembed::parsing::parse_positions_file(pos_path)?;

        let mut all_results = Vec::new();
        let embeddings = convert_to_embeddings(&iterations, graph);

        for embedding in embeddings {
            let mut iteration_results = Vec::new();
            for node_id in 0..embedding.positions.len() {
                let neighbors = embedding.nearest_neighbors(node_id, 1.0);
                iteration_results.push(neighbors);
            }
            all_results.push(iteration_results);
        }

        Ok(all_results)
    }

    fn write_test_file(
        &self,
        file_path: &str,
        iterations: &[Vec<Vec<NodeId>>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);

        // Write number of nodes
        if let Some(first_iteration) = iterations.first() {
            let num_nodes = first_iteration.len() as u64;
            writer.write_all(&num_nodes.to_le_bytes())?;
        } else {
            return Err("No iterations found".into());
        }

        // Write iterations
        for iteration in iterations {
            for node_neighbors in iteration {
                // Write length of neighbor list as u32
                let length = node_neighbors.len() as u32;
                writer.write_all(&length.to_le_bytes())?;

                // Write neighbor IDs as u32
                for &neighbor in node_neighbors {
                    writer.write_all(&(neighbor as u32).to_le_bytes())?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    fn read_test_file(
        &self,
        file_path: &str,
    ) -> Result<Vec<Vec<Vec<NodeId>>>, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);

        // Read number of nodes
        let mut num_nodes_bytes = [0u8; 8];
        reader.read_exact(&mut num_nodes_bytes)?;
        let num_nodes = u64::from_le_bytes(num_nodes_bytes) as usize;

        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        let mut iterations = Vec::new();
        let mut pos = 0;

        while pos < buffer.len() {
            let mut iteration = Vec::new();

            for _ in 0..num_nodes {
                // Read length of neighbor list
                if pos + 4 > buffer.len() {
                    break;
                }
                let length_bytes = [
                    buffer[pos],
                    buffer[pos + 1],
                    buffer[pos + 2],
                    buffer[pos + 3],
                ];
                let length = u32::from_le_bytes(length_bytes) as usize;
                pos += 4;

                let mut neighbors = Vec::new();

                // Read neighbor IDs
                for _ in 0..length {
                    if pos + 4 > buffer.len() {
                        break;
                    }
                    let neighbor_bytes = [
                        buffer[pos],
                        buffer[pos + 1],
                        buffer[pos + 2],
                        buffer[pos + 3],
                    ];
                    let neighbor = u32::from_le_bytes(neighbor_bytes) as NodeId;
                    neighbors.push(neighbor);
                    pos += 4;
                }

                iteration.push(neighbors);
            }

            if !iteration.is_empty() {
                iterations.push(iteration);
            }
        }

        Ok(iterations)
    }

    /// Run correctness tests with configurable options
    pub async fn run_tests(
        &self,
        all_iterations: bool,
        all_graphs: bool,
        result_id_filter: Option<i64>,
        graph_id_filter: Option<i64>,
        dim_filter: Option<i32>,
        run_unit_tests: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if run_unit_tests {
            println!("Running unit tests from main crate...");
            let status = std::process::Command::new("cargo")
                .args(["test", "--manifest-path", "../Cargo.toml"])
                .status()?;

            if !status.success() {
                println!("Unit tests failed!");
                return Err("Unit tests failed".into());
            }
            println!("Unit tests passed ✓");
        }

        if all_graphs {
            println!("Running extensive correctness tests...");
        } else {
            println!("Running quick correctness tests...");
        }

        struct TestResult {
            result_id: i64,
            embedding_dim: i32,
            graph_id: i64,
            processed_n: i64,
        }

        let test_results = sqlx::query_as!(
            TestResult,
            "SELECT result_id, pr.embedding_dim, g.graph_id, g.processed_n
            FROM tests t
            JOIN position_results pr USING (result_id)
            JOIN graphs g USING (graph_id)
            WHERE $1 OR g.processed_n < 5000
            ",
            all_graphs,
        )
        .fetch_all(&self.pool)
        .await?;

        // Apply filters in Rust (simpler than dynamic SQL)
        let filtered_results: Vec<_> = test_results
            .into_iter()
            .filter(|r| {
                if let Some(rid) = result_id_filter {
                    if r.result_id != rid {
                        return false;
                    }
                }
                if let Some(gid) = graph_id_filter {
                    if r.graph_id != gid {
                        return false;
                    }
                }
                if let Some(dim) = dim_filter {
                    if r.embedding_dim != dim {
                        return false;
                    }
                }
                true
            })
            .collect();

        if filtered_results.is_empty() {
            println!("No test files found matching criteria. Run 'generate-test' first.");
            return Ok(());
        }

        // For quick tests, select one graph per dimension
        let final_results: Vec<_> =
            if !all_graphs && result_id_filter.is_none() && graph_id_filter.is_none() {
                let mut per_dim: HashMap<i32, TestResult> = HashMap::new();
                for result in filtered_results {
                    let entry = per_dim.entry(result.embedding_dim);

                    match entry {
                        Entry::Occupied(mut occupied_entry) => {
                            if result.processed_n < occupied_entry.get().processed_n {
                                occupied_entry.insert(result);
                            }
                        }
                        Entry::Vacant(vacant_entry) => {
                            vacant_entry.insert(result);
                        }
                    };
                }
                per_dim.into_values().collect()
            } else {
                filtered_results.into_iter().collect()
            };

        for result in final_results {
            println!(
                "Testing result_id {} (dim={}, n={})",
                result.result_id, result.embedding_dim, result.processed_n
            );

            match result.embedding_dim {
                2 => {
                    self.run_test_for_result::<2>(result.result_id, !all_iterations)
                        .await?
                }
                4 => {
                    self.run_test_for_result::<4>(result.result_id, !all_iterations)
                        .await?
                }
                8 => {
                    self.run_test_for_result::<8>(result.result_id, !all_iterations)
                        .await?
                }
                16 => {
                    self.run_test_for_result::<16>(result.result_id, !all_iterations)
                        .await?
                }
                32 => {
                    self.run_test_for_result::<32>(result.result_id, !all_iterations)
                        .await?
                }
                _ => println!("Skipping unsupported dimension: {}", result.embedding_dim),
            }
        }

        Ok(())
    }

    async fn run_test_for_result<const D: usize>(
        &self,
        result_id: i64,
        last_iteration_only: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get test file info
        let test_record = sqlx::query_as!(
            TestRecord,
            "SELECT result_id, file_path, created_at FROM tests WHERE result_id = $1",
            result_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Get result and graph info
        let result = sqlx::query!(
            "SELECT pr.*, g.file_path as graph_path FROM position_results pr 
             JOIN graphs g USING (graph_id) WHERE pr.result_id = $1",
            result_id
        )
        .fetch_one(&self.pool)
        .await?;

        // Load graph and positions
        let pos_path = format!("{}/{}", self.data_directory, result.file_path);
        let graph_path = format!("{}/{}", self.data_directory, result.graph_path);
        let test_file_path = format!("{}/{}", self.data_directory, test_record.file_path);

        let graph = rembed::graph::Graph::parse_from_edge_list_file(
            &graph_path,
            result.embedding_dim as usize,
            result.dim_hint as usize,
        )?;

        let iterations: rembed::parsing::Iterations<D> =
            rembed::parsing::parse_positions_file(&pos_path)?;

        // Load ground truth
        let ground_truth = self.read_test_file(&test_file_path)?;

        let mut embeddings = convert_to_embeddings(&iterations, &graph);
        // Test each iteration (or just the last one for quick tests)
        let iterations_to_test = if last_iteration_only {
            vec![embeddings.next_back().unwrap()]
        } else {
            embeddings.collect()
        };

        let mut total_errors = 0;

        for (iter_idx, embedding) in iterations_to_test.iter().enumerate() {
            let iteration_idx = if last_iteration_only {
                ground_truth.len() - 1
            } else {
                iter_idx
            };

            if iteration_idx >= ground_truth.len() {
                println!(
                    "Warning: iteration {} not found in ground truth",
                    iteration_idx
                );
                continue;
            }

            let data_structures = data_structures(embedding);

            for structure in data_structures {
                let errors = self.test_structure(
                    structure.as_ref() as &dyn SpatialIndex<D>,
                    &ground_truth[iteration_idx],
                    iteration_idx,
                );
                total_errors += errors;
            }
        }

        if total_errors == 0 {
            println!("✓ All tests passed for result_id {}", result_id);
        } else {
            println!(
                "✗ {} errors found for result_id {}",
                total_errors, result_id
            );
        }

        Ok(())
    }

    fn test_structure<'a, const D: usize>(
        &'a self,
        structure: &'a (dyn rembed::query::SpatialIndex<D> + 'a),
        ground_truth: &'a [Vec<NodeId>],
        iteration: usize,
    ) -> usize {
        let mut errors = 0;

        for node_id in 0..ground_truth.len() {
            let expected: HashSet<NodeId> = ground_truth[node_id].iter().cloned().collect();
            let mut actual: HashSet<NodeId> = structure
                .nearest_neighbors(node_id, 1.0)
                .into_iter()
                .collect();
            actual.retain(|x| x != &node_id);

            if expected.iter().any(|n| !actual.contains(n)) {
                errors += 1;
                if errors < 5 {
                    self.print_diff(&structure.name(), iteration, node_id, &expected, &actual);
                } else if errors == 5 {
                    println!("[ Truncated ]\n")
                }
            }
        }

        if errors == 0 {
            println!("  ✓ {} passed", structure.name());
        } else {
            println!("  ✗ {} failed with {} errors", structure.name(), errors);
        }

        errors
    }

    fn print_diff(
        &self,
        structure_name: &str,
        iteration: usize,
        node_id: NodeId,
        expected: &HashSet<NodeId>,
        actual: &HashSet<NodeId>,
    ) {
        println!(
            "DIFF: {} iteration={} node={}:",
            structure_name, iteration, node_id
        );

        let mut missing: Vec<_> = expected.difference(actual).collect();
        let mut extra: Vec<_> = actual.difference(expected).collect();
        missing.sort_unstable();
        extra.sort_unstable();

        if !missing.is_empty() {
            println!("  Missing: {:?}", missing);
        }
        if !extra.is_empty() {
            println!("  Extra: {:?}", extra);
        }
    }
}
