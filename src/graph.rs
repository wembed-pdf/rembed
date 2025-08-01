use std::cmp::max;
use std::collections::HashSet;
use std::fs::read_to_string;
use std::io;

use crate::NodeId;

// A node in the graph
// Each node has a weight, which is degree ^ (d/8)
#[derive(Clone, Debug)]
pub struct Node {
    pub weight: f64,
    pub neighbors: Vec<usize>,
    pub neighbors_set: HashSet<usize>,
}

// A graph structure
// It contains the embedding dimension, nodes, and edges
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<(NodeId, NodeId)>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub const fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Parses a graph from an edge list file.
    /// The file should contain pairs of integers representing edges.
    pub fn parse_from_edge_list_file(
        file_path: &str,
        embedding_dim: usize,
        latent_dim_hint: usize,
    ) -> io::Result<Self> {
        let edges = read_to_string(file_path)
            .unwrap_or_else(|e| panic!("Tried to load file from {file_path}:\n {e:?}"))
            .lines()
            .map(|s| {
                s.split_ascii_whitespace()
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .map(|s| {
                let u = s[0].parse::<usize>().unwrap();
                let v = s[1].parse::<usize>().unwrap();
                (u, v)
            })
            .collect::<Vec<_>>();
        Self::from_edge_list(edges, embedding_dim, latent_dim_hint)
    }

    /// Parses a graph from an edge list.
    /// The file should contain pairs of integers representing edges.
    pub fn from_edge_list(
        edges: Vec<(usize, usize)>,
        embedding_dim: usize,
        latent_dim_hint: usize,
    ) -> io::Result<Self> {
        let mut graph = Graph::new();
        graph.edges = edges;
        let mut node_degree = Vec::new();
        for (u, v) in graph.edges.iter() {
            if node_degree.len() <= max(*u, *v) {
                node_degree.resize(max(*u, *v) + 1, 0);
            }
            node_degree[*u] += 1;
            node_degree[*v] += 1;
        }
        let total_weight: usize = node_degree.iter().sum();
        let weight_norm = node_degree.len() as f64 / total_weight as f64;
        let dim_ratio = embedding_dim as f64 / latent_dim_hint as f64;
        (0..node_degree.len()).for_each(|i| {
            graph.nodes.push(Node {
                // weight = degree ^ (d/8)
                weight: ((node_degree[i] as f64).powf(dim_ratio) * weight_norm)
                    .powf(1. / embedding_dim as f64),
                neighbors: Vec::new(),
                neighbors_set: HashSet::new(),
            });
        });
        for (u, v) in graph.edges.iter() {
            graph.nodes[*u].neighbors.push(*v);
            graph.nodes[*u].neighbors_set.insert(*v);
            graph.nodes[*v].neighbors.push(*u);
            graph.nodes[*v].neighbors_set.insert(*u);
        }

        // TODO: Sort nodes by degree and reassign indices
        Ok(graph)
    }
}

impl crate::query::Graph for Graph {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.nodes[first].neighbors_set.contains(&second)
    }
    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        &self.nodes[index].neighbors
    }
    fn weight(&self, index: NodeId) -> f64 {
        self.nodes[index].weight
    }
}
