use crate::{NodeId, dvec::DVec};

pub trait Graph {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool;
    fn neighbors(&self, index: NodeId) -> &[NodeId];
    fn weight(&self, index: NodeId) -> f64;
}

pub trait Position<const D: usize> {
    fn position(&self, index: NodeId) -> &DVec<D>;
    fn dim(&self) -> usize {
        D
    }
}
pub trait IndexClone<const D: usize>: SpatialIndex<D> {
    fn clone_box<'a>(&'a self) -> Box<dyn SpatialIndex<D> + 'a>;
}

impl<const D: usize, T: Clone + Sized + SpatialIndex<D>> IndexClone<D> for T {
    fn clone_box<'a>(&'a self) -> Box<dyn SpatialIndex<D> + 'a> {
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
        let common_files = concat![include_str!("dvec.rs")];
        let implementation = self.implementation_string();
        let hasher = sha2::Sha256::new().chain_update(common_files);
        format!("{:x}", hasher.chain_update(implementation).finalize())
    }
}

pub trait Query {
    /// Return the list of neighbors in a given radius. You are allowed to return results asymmetricallys e.g only nodes to the left of you
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize>;

    /// Runs a batch of nn queries and makes the result symmetric
    fn nearest_neighbors_batched(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        let mut results = vec![vec![]; indices.len()];
        for &index in indices {
            for other_node_id in self.nearest_neighbors(index, 1.) {
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

pub trait Embedder<const D: usize>: Query + Update<D> + Graph + Position<D> {
    fn calculate_step(&mut self, dt: f64) {
        todo!("Not implemented yet {}", dt);
    }
    fn repelling_nodes(&self, index: usize) -> Vec<usize> {
        let mut result = self.nearest_neighbors(index, 1.);
        let pos = self.position(index);
        let weight = self.weight(index);
        // todo consider graph edges
        result.retain(|&x| {
            (self.position(x).distance_squared(pos) as f64) < (weight * self.weight(x)).powi(2)
        });

        result
    }
    fn attracting_nodes(&self, index: usize) -> Vec<usize> {
        self.neighbors(index).to_vec()
    }
}
