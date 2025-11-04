use rand::seq::IteratorRandom;

use crate::{Embedding, NodeId, dvec::DVec};

pub trait Graph {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool;
    fn neighbors(&self, index: NodeId) -> &[NodeId];
    fn weight(&self, index: NodeId) -> f64;
}

pub trait Position<const D: usize> {
    fn position(&self, index: NodeId) -> &DVec<D>;
    fn num_nodes(&self) -> usize;
    fn dim(&self) -> usize {
        D
    }
}
pub trait IndexClone<const D: usize>: SpatialIndex<D> {
    fn clone_box<'a>(&'a self) -> Box<dyn SpatialIndex<D> + 'a>;
    fn clone_box_cloneable<'a>(&self) -> Box<dyn IndexClone<D> + 'a + Sync>
    where
        Self: 'a;
}

impl<const D: usize, T: Clone + Sized + SpatialIndex<D> + Sync> IndexClone<D> for T {
    fn clone_box<'a>(&'a self) -> Box<dyn SpatialIndex<D> + 'a> {
        Box::new(self.clone())
    }
    fn clone_box_cloneable<'a>(&self) -> Box<dyn IndexClone<D> + 'a + Sync>
    where
        T: 'a,
    {
        Box::new(self.clone())
    }
}

pub trait SpatialIndex<const D: usize>: Query + Update<D> + Graph + Position<D> {
    fn name(&self) -> String;

    /// Returns the source code implementation as a string for checksum calculation.
    /// This should include all files that affect the performance of this data structure.
    fn implementation_string(&self) -> &'static str;

    fn checksum(&self) -> String {
        use sha2::Digest;
        let common_files = concat![
            include_str!("dvec.rs"),
            include_str!("../.cargo/config.toml")
        ];
        let implementation = self.implementation_string();
        let hasher = sha2::Sha256::new().chain_update(common_files);
        format!("{:x}", hasher.chain_update(implementation).finalize())
    }
}

// struct Statistics {
//     nodes_queried: usize,
//     branches: usize,
//     dist_checks: usize,
//     final_radius:
// }

pub trait Query {
    /// Return the list of neighbors in a given radius. You are allowed to return results asymmetricallys e.g only nodes to the left of you
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>);
    fn nearest_neighbors_owned(&self, index: usize, radius: f64) -> Vec<NodeId> {
        let mut results = Vec::new();
        self.nearest_neighbors(index, radius, &mut results);
        results
    }
    // fn nearest_neighbors_instrumented(&self, _index: usize, _radius: f64, _stats: &mut Statistics) {
    // }

    /// Runs a batch of nn queries and makes the result symmetric
    fn nearest_neighbors_batched(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        let mut results = vec![vec![]; indices.len()];
        for &index in indices {
            for other_node_id in self.nearest_neighbors_owned(index, 1.) {
                results[other_node_id].push(index);
                results[index].push(other_node_id);
            }
        }
        for vec in &mut results {
            vec.sort_unstable();
            vec.dedup();
        }
        results
    }
}

pub trait Update<const D: usize> {
    fn update_positions(&mut self, postions: &[DVec<D>]);
}

pub trait Embedder<'a, const D: usize>: Query + Update<D> + Graph + Position<D> {
    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        self.nearest_neighbors(index, 1., result);
        let pos = self.position(index);
        let weight = self.weight(index);

        result.retain(|&x| {
            index != x
                && !self.is_connected(index, x)
                && (self.position(x).distance_squared(pos) as f64)
                    < (weight * self.weight(x)).powi(2)
        });
    }
    fn attracting_nodes(&self, index: usize) -> Vec<usize> {
        self.neighbors(index).to_vec()
    }

    fn new(embedding: &crate::Embedding<'a, D>) -> Self;
    fn from_graph(graph: &'a crate::graph::Graph) -> Self
    where
        Self: Sized,
    {
        Self::new(&Embedding {
            positions: Vec::new(),
            graph,
        })
    }

    fn graph_statistics(&self) -> (f64, f64) {
        let ids: Vec<_> = (0..(self.num_nodes())).collect();
        let results = self.nearest_neighbors_batched(&ids);
        let mut found_edges = 0;
        let mut missed_edges = 0;
        let mut found_non_edges = 0;
        for (i, close_nodes) in results
            .iter()
            .enumerate()
            .choose_multiple(&mut rand::rng(), 500)
        {
            for close_node in close_nodes {
                if self.is_connected(i, *close_node) {
                    found_edges += 1;
                } else if (self
                    .position(i)
                    .distance_squared(self.position(*close_node)) as f64)
                    < (self.weight(*close_node) * self.weight(i)).powi(2)
                {
                    found_non_edges += 1;
                }
            }
            for neighbor in self.neighbors(i) {
                if close_nodes.contains(neighbor) {
                    continue;
                }
                missed_edges += 1;
            }
        }

        (
            found_edges as f64 / (found_edges + missed_edges) as f64,
            found_edges as f64 / (found_edges + found_non_edges).max(1) as f64,
        )
    }
    fn f1(&self) -> f64 {
        let (percision, recall) = self.graph_statistics();
        2. / (recall.recip() + percision.recip())
    }
}
