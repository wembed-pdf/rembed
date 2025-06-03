use std::process::Command;
use std::str::FromStr;
use std::{collections::HashMap, io::Read};

use indicatif::ProgressBar;
use rayon::prelude::*;

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

        // Perform DFS to find the largest component
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

        // Create a new graph with the largest component
        // map ids in the component to a new range
        let id_map: HashMap<i32, i32> = component
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id as i32))
            .collect();
        use std::collections::HashSet;
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
}

pub struct GraphGenerator {
    pub wembed_path: String,
    pub graphs_path: String,
    pub output_path: String,
}

impl GraphGenerator {
    pub fn new(wembed_path: String, graphs_path: String, output_path: String) -> Self {
        Self {
            wembed_path,
            graphs_path,
            output_path,
        }
    }

    // remove the top two lines of every graph
    // reduce the graph to its largest component
    // remap the node ids to a new range
    // sort the edge list
    // save the graph to graph path
    pub fn change_graphs(&self, old_graph_path: String) {
        let graph_files = std::fs::read_dir(&old_graph_path)
            .expect("Failed to read graph directory")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();

        println!("Changing graph files from: {}", old_graph_path);
        let pb = ProgressBar::new(graph_files.len() as u64);

        for graph_file in graph_files {
            let mut file = std::fs::File::open(&graph_file).expect("Failed to open graph file");
            let mut lines = String::new();
            file.read_to_string(&mut lines)
                .expect("Failed to read graph file");

            let mut graph = Graph::new(
                lines
                    .lines()
                    .skip(2)
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        let u = i32::from_str(parts[0]).ok()?;
                        let v = i32::from_str(parts[1]).ok()?;
                        Some((u, v))
                    })
                    .collect(),
                0,
            )
            .reduce_to_largest_component();
            graph.compute_avg_degree();

            let new_content = graph
                .to_edge_list()
                .iter()
                .map(|&(u, v)| format!("{} {}\n", u, v))
                .collect::<String>();

            let name = graph_file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
                + "_n-"
                + &graph.n.to_string()
                + "_avg_degree-"
                + &graph.avg_degree.to_string()
                + ".txt";

            let new_path = format!("{}/{}", &self.graphs_path, name);
            // Write the modified content back to the file
            std::fs::write(new_path, new_content).expect("Failed to write modified graph file");
            pb.inc(1);
        }
    }

    pub fn generate(&self) {
        // Get all graph instances from the graphs path
        let graph_files = std::fs::read_dir(&self.graphs_path)
            .expect("Failed to read graph directory")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>();

        println!("Generating positions using WEmbed at: {}", self.wembed_path);
        println!("Graph files found: {}", graph_files.len());
        println!("Output will be saved to: {}", self.output_path);
        let pb = ProgressBar::new(graph_files.len() as u64);
        graph_files.par_iter().for_each(|graph_file| {
            let graph_name = graph_file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            for dim in generate_dim() {
                for dim_hint in generate_dim_hint() {
                    Command::new(&self.wembed_path)
                        .stdout(std::process::Stdio::null())
                        .arg("-i")
                        .arg(graph_file.to_str().unwrap())
                        .arg("--logging-output")
                        .arg(format!(
                            "{}/{}_dim-{}_dim_hint-{}.log",
                            self.output_path, graph_name, dim, dim_hint
                        ))
                        .arg("--dim-hint")
                        .arg("8")
                        .arg("--dim")
                        .arg("8")
                        .arg("--iterations")
                        .arg("1")
                        .status()
                        .expect("Failed to execute WEmbed command");
                    pb.inc(1);
                }
            }
        });
        pb.finish_with_message("Position generation complete");
    }
}

fn generate_dim() -> Vec<usize> {
    vec![8]
}

fn generate_dim_hint() -> Vec<usize> {
    vec![8]
}
