use crate::{NodeId, dvec::DVec};

pub trait Graph {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool;
    fn neighbors(&self, index: NodeId) -> &[NodeId];
    fn weight(&self, index: NodeId) -> f64;
}

pub trait Position<const D: usize> {
    fn position(&self, index: NodeId) -> &DVec<D>;
}

pub trait Query {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize>;
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
        let nn_count = result.len();
        result.retain(|&x| {
            (self.position(x).distance_squared(pos) as f64) < (weight * self.weight(x)).powi(2)
        });

        // println!("nn_count: {nn_count}, filtered: {}", result.len());
        result
    }
    fn attracting_nodes(&self, index: usize) -> Vec<usize> {
        self.neighbors(index).to_vec()
    }
}
