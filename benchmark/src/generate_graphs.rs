use sha2::{Digest, Sha256};
use sqlx::Postgres;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::process::Command;
use std::str::FromStr;
use tracing::info;

use crate::create_progress_bar;
use crate::job_manager::JobManager;

struct Seed {
    wseed: i32,
    pseed: i32,
    sseed: i32,
}

pub struct Graph {
    pub adjacency_list: Vec<Vec<i32>>,
    pub n: usize,
    pub avg_degree: f64,
}

impl Graph {
    pub fn new(edges: Vec<(i32, i32)>, n: usize) -> Self {
        let mut adjacency_list: Vec<Vec<i32>> = Vec::new();
        for (u, v) in edges {
            if u as usize >= adjacency_list.len() {
                adjacency_list.resize((u + 1) as usize, Vec::new());
            }
            if v as usize >= adjacency_list.len() {
                adjacency_list.resize((v + 1) as usize, Vec::new());
            }
            adjacency_list[u as usize].push(v);
            adjacency_list[v as usize].push(u);
        }
        Self {
            adjacency_list,
            n,
            avg_degree: 0.0,
        }
    }

    pub fn reduce_to_largest_component(&self) -> Self {
        let mut visited = vec![false; self.adjacency_list.len()];
        let mut component = Vec::new();

        for start in 0..self.adjacency_list.len() {
            if !visited[start] {
                let mut stack = vec![start];
                let mut current_component = Vec::new();

                while let Some(node) = stack.pop() {
                    if !visited[node] {
                        visited[node] = true;
                        current_component.push(node as i32);
                        for &neighbor in &self.adjacency_list[node] {
                            if !visited[neighbor as usize] {
                                stack.push(neighbor as usize);
                            }
                        }
                    }
                }

                if current_component.len() > component.len() {
                    component = current_component;
                }
            }
        }

        let id_map: HashMap<i32, i32> = component
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id as i32))
            .collect();
        let component_set: HashSet<i32> = component.iter().copied().collect();
        let n = component.len();
        let edges: Vec<(i32, i32)> = {
            let component_set = &component_set;
            let id_map = &id_map;
            self.adjacency_list
                .iter()
                .enumerate()
                .flat_map(|(u, neighbors)| {
                    neighbors.iter().filter_map(move |&v| {
                        if component_set.contains(&(u as i32)) && component_set.contains(&v) {
                            Some((id_map[&(u as i32)], id_map[&v]))
                        } else {
                            None
                        }
                    })
                })
                .collect()
        };
        Graph::new(edges, n)
    }

    pub fn compute_avg_degree(&mut self) {
        let total_edges: usize = self
            .adjacency_list
            .iter()
            .map(|neighbors| neighbors.len())
            .sum();
        self.avg_degree = total_edges as f64 / self.n as f64;
    }

    fn to_edge_list(&self) -> Vec<(i32, i32)> {
        let mut edges = Vec::new();
        for (u, neighbors) in self.adjacency_list.iter().enumerate() {
            for &v in neighbors {
                if (u as i32) < v {
                    edges.push((u as i32, v));
                }
            }
        }
        edges
    }

    fn to_sorted_edge_string(&self) -> String {
        self.to_edge_list()
            .iter()
            .map(|&(u, v)| format!("{} {}\n", u, v))
            .collect()
    }
}

pub struct GraphGenerator {
    pub girgs_path: String,
    pub output_path: String,
}

impl GraphGenerator {
    pub fn new(girgs_path: String, output_path: String) -> Self {
        Self {
            girgs_path,
            output_path,
        }
    }

    pub async fn generate(&self) -> Result<(), Box<dyn std::error::Error>> {
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
        let pool = sqlx::PgPool::connect(&database_url).await?;

        println!("Generating graphs using GIRGs at: {}", self.girgs_path);
        println!("Output will be saved to: {}", self.output_path);

        let seeds = generate_seeds();
        // let avg_degrees = [15, 20, 25];
        let avg_degrees = [15];
        let n_s = half_log10(1000.0, 1_000_005.0);
        // let dimensions = [2, 3, 4];
        let dimensions = [4];
        // let ples = [2.2, 2.5, 2.8, 8.];
        // let power_law_exponents = [2.2, 2.5, 2.8];
        let power_law_exponents = [2.5];
        // let alphas = [f64::INFINITY, 2., 1.1];
        let alphas = [f64::INFINITY];
        let total_graphs = n_s.len()
            * seeds.len()
            * avg_degrees.len()
            * dimensions.len()
            * power_law_exponents.len()
            * alphas.len();

        let pb = create_progress_bar(total_graphs);

        std::fs::create_dir_all(&self.output_path)?;

        for seed in seeds {
            for &avg_degree in &avg_degrees {
                for &n in &n_s {
                    for dim in dimensions {
                        for ple in power_law_exponents {
                            for alpha in alphas {
                                let mut tx = pool.begin().await?;

                                let existing_graph_id = check_existing_graph(
                                    &mut tx, n, dim, ple, alpha, avg_degree, &seed,
                                )
                                .await?;
                                if existing_graph_id.is_some() {
                                    pb.inc(1);
                                    continue;
                                }

                                pb.set_message(format!(
                                    "Generating graph n={} deg={}",
                                    n, avg_degree
                                ));

                                let temp_filename = format!(
                                    "temp_genhrg_n-{}_deg-{}_dim-{}_ple-{}_alpha-{}_wseed-{}_pseed-{}_sseed-{}",
                                    n,
                                    avg_degree,
                                    dim,
                                    ple,
                                    alpha,
                                    seed.wseed,
                                    seed.pseed,
                                    seed.sseed
                                );
                                let temp_file_path =
                                    format!("{}/{}", self.output_path, temp_filename);

                                let status = self.run_girgs(
                                    &seed,
                                    avg_degree,
                                    n,
                                    &temp_file_path,
                                    dim,
                                    ple,
                                    alpha,
                                )?;
                                let raw_file_path = format!("{}.txt", temp_file_path);

                                if !status.success() {
                                    return Err(format!(
                                        "Failed to generate graph n={} deg={}",
                                        n, avg_degree
                                    )
                                    .into());
                                }

                                let processed_graph = self.process_raw_graph(&raw_file_path)?;

                                let graph_id = insert_graph_with_metrics(
                                    &mut tx,
                                    n,
                                    avg_degree,
                                    dim,
                                    ple,
                                    alpha,
                                    &seed,
                                    processed_graph.n as i32,
                                    processed_graph.avg_degree,
                                )
                                .await?;

                                let final_filename = format!(
                                    "{}_girg_n-{}_deg-{}_dim-{}_ple-{}_alpha-{}_wseed-{}_pseed-{}_sseed-{}",
                                    graph_id,
                                    n,
                                    avg_degree,
                                    dim,
                                    ple,
                                    alpha,
                                    seed.wseed,
                                    seed.pseed,
                                    seed.sseed
                                );
                                let final_file_path = format!(
                                    "{}/generated/graphs/{}",
                                    self.output_path, final_filename
                                );

                                std::fs::write(
                                    &final_file_path,
                                    processed_graph.to_sorted_edge_string(),
                                )?;
                                let checksum = calculate_file_checksum(&final_file_path)?;

                                update_file_path_and_checksum(
                                    &mut tx,
                                    graph_id,
                                    &format!("generated/graphs/{}", final_filename),
                                    &checksum,
                                )
                                .await?;

                                tx.commit().await?;

                                // Create position generation jobs using JobManager
                                let job_manager = JobManager::new(pool.clone());
                                let jobs_created =
                                    job_manager.create_jobs_for_graph(graph_id).await?;

                                std::fs::remove_file(raw_file_path).ok();

                                info!(
                                    "Created graph {} with {} position generation jobs",
                                    graph_id, jobs_created
                                );
                                pb.inc(1);
                            }
                        }
                    }
                }
            }
        }

        pb.finish_with_message("Graph generation complete");
        crate::push_files().await?;
        Ok(())
    }

    fn process_raw_graph(&self, file_path: &str) -> Result<Graph, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let edges: Vec<(i32, i32)> = contents
            .lines()
            .skip(2)
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let u = i32::from_str(parts[0]).ok()?;
                    let v = i32::from_str(parts[1]).ok()?;
                    Some((u, v))
                } else {
                    None
                }
            })
            .collect();

        let mut graph = Graph::new(edges, 0).reduce_to_largest_component();
        graph.compute_avg_degree();
        Ok(graph)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_girgs(
        &self,
        seed: &Seed,
        avg_degree: i32,
        n: i32,
        file_path: &str,
        dim: i32,
        ple: f64,
        alpha: f64,
    ) -> Result<std::process::ExitStatus, std::io::Error> {
        Command::new(&self.girgs_path)
            .stdout(std::process::Stdio::null())
            .arg("-n")
            .arg(n.to_string())
            .arg("-d")
            .arg(dim.to_string())
            .arg("-ple")
            .arg(ple.to_string())
            .arg("-alpha")
            .arg(alpha.to_string())
            .arg("-deg")
            .arg(avg_degree.to_string())
            .arg("-file")
            .arg(file_path)
            .arg("-edge")
            .arg("1")
            .arg("-wseed")
            .arg(seed.wseed.to_string())
            .arg("-pseed")
            .arg(seed.pseed.to_string())
            .arg("-sseed")
            .arg(seed.sseed.to_string())
            .status()
    }
}

async fn check_existing_graph(
    tx: &mut sqlx::Transaction<'static, Postgres>,
    n: i32,
    dim: i32,
    ple: f64,
    alpha: f64,
    avg_degree: i32,
    seed: &Seed,
) -> Result<Option<i64>, sqlx::Error> {
    sqlx::query_scalar!(
        "SELECT graph_id FROM graphs WHERE n = $1
        AND deg = $2 AND dim = $3 AND ple = $4 AND alpha = $5
        AND wseed = $6 AND pseed = $7 AND sseed = $8",
        n,
        avg_degree,
        dim,
        ple,
        alpha,
        seed.wseed,
        seed.pseed,
        seed.sseed
    )
    .fetch_optional(&mut **tx)
    .await
}

#[allow(clippy::too_many_arguments)]
async fn insert_graph_with_metrics(
    tx: &mut sqlx::Transaction<'static, Postgres>,
    original_n: i32,
    original_deg: i32,
    dim: i32,
    ple: f64,
    alpha: f64,
    seed: &Seed,
    processed_n: i32,
    processed_avg_degree: f64,
) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar!(
        r#"
        INSERT INTO graphs (n, deg, dim, ple, alpha, wseed, pseed, sseed, processed_n, processed_avg_degree, file_path, checksum)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING graph_id
        "#,
        original_n, original_deg, dim, ple, alpha, seed.wseed, seed.pseed, seed.sseed, processed_n, processed_avg_degree, "", ""
    ).fetch_one(&mut **tx).await
}

async fn update_file_path_and_checksum(
    tx: &mut sqlx::Transaction<'static, Postgres>,
    graph_id: i64,
    file_path: &str,
    checksum: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "UPDATE graphs SET file_path = $1, checksum = $2 WHERE graph_id = $3",
        file_path,
        checksum,
        graph_id
    )
    .execute(&mut **tx)
    .await?;
    Ok(())
}

fn calculate_file_checksum(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let contents = std::fs::read(file_path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

fn half_log10(start: f64, end: f64) -> Vec<i32> {
    (0..)
        .scan(start, |state, _| {
            if *state > end {
                return None;
            }
            let current = *state as i32;
            *state *= 10f64.powf(0.5);
            Some(current)
        })
        .collect()
}

fn generate_seeds() -> Vec<Seed> {
    let mut seeds = Vec::new();
    for i in 0..3 {
        seeds.push(Seed {
            wseed: 12 + i,
            pseed: 130 + i,
            sseed: 1400 + i,
        });
    }
    seeds
}
