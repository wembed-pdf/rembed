use crate::{
    ATree, NodeId,
    dvec::DVec,
    query::{Embedder, Graph, Position, Query, SpatialIndex, Update},
    random_projection_lsh::RandomProjectionLsh,
};
use std::sync::Mutex;

pub struct MeasuredLSH<'a, const D: usize> {
    pub lsh: RandomProjectionLsh<'a, D>,
    pub ground_truth: ATree<'a, D>,
    recall_measurements: Mutex<Vec<f64>>,
}

impl<'a, const D: usize> Clone for MeasuredLSH<'a, D> {
    fn clone(&self) -> Self {
        Self {
            lsh: self.lsh.clone(),
            ground_truth: self.ground_truth.clone(),
            recall_measurements: Mutex::new(Vec::new()), // Fresh measurements for clone
        }
    }
}

impl<'a, const D: usize> MeasuredLSH<'a, D> {
    pub fn new(lsh: RandomProjectionLsh<'a, D>, ground_truth: ATree<'a, D>) -> Self {
        Self {
            lsh,
            ground_truth,
            recall_measurements: Mutex::new(Vec::new()),
        }
    }

    /// Get the average recall across all measurements
    pub fn average_recall(&self) -> f64 {
        let measurements = self.recall_measurements.lock().unwrap();
        if measurements.is_empty() {
            return 0.0;
        }
        measurements.iter().sum::<f64>() / measurements.len() as f64
    }

    /// Clear measurements (call at start of each iteration if tracking per-iteration)
    pub fn clear_measurements(&mut self) {
        self.recall_measurements.lock().unwrap().clear();
    }
}

// Delegate most trait implementations to the LSH structure
impl<'a, const D: usize> Graph for MeasuredLSH<'a, D> {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.lsh.is_connected(first, second)
    }
    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.lsh.neighbors(index)
    }
    fn weight(&self, index: NodeId) -> f64 {
        self.lsh.weight(index)
    }
}

impl<'a, const D: usize> Position<D> for MeasuredLSH<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        self.lsh.position(index)
    }
    fn num_nodes(&self) -> usize {
        self.lsh.num_nodes()
    }
}

impl<'a, const D: usize> Update<D> for MeasuredLSH<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], last_delta: Option<f64>) {
        self.lsh.update_positions(positions, last_delta);
        self.ground_truth.update_positions(positions, last_delta);
    }
}

impl<'a, const D: usize> Query for MeasuredLSH<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<usize>) {
        self.lsh.nearest_neighbors(index, radius, results)
    }
}

impl<'a, const D: usize> SpatialIndex<D> for MeasuredLSH<'a, D> {
    fn name(&self) -> String {
        format!("measured-{}", self.lsh.name())
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("measured_lsh.rs")
    }
}

impl<'a, const D: usize> Embedder<'a, D> for MeasuredLSH<'a, D> {
    fn new(_embedding: &crate::Embedding<'a, D>) -> Self {
        panic!("MeasuredLSH requires explicit construction with both LSH and ground truth")
    }

    // Override repelling_nodes to measure recall
    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        // Get ground truth repelling nodes
        let mut ground_truth_nodes = Vec::new();
        self.ground_truth
            .repelling_nodes(index, &mut ground_truth_nodes);

        // Get LSH repelling nodes
        self.lsh.repelling_nodes(index, result);

        // Compute recall: |LSH ∩ ground_truth| / |ground_truth|
        if !ground_truth_nodes.is_empty() {
            let ground_truth_set: std::collections::HashSet<_> =
                ground_truth_nodes.iter().copied().collect();
            let lsh_set: std::collections::HashSet<_> = result.iter().copied().collect();

            let intersection_size = ground_truth_set.intersection(&lsh_set).count();
            let recall = intersection_size as f64 / ground_truth_nodes.len() as f64;

            // Store measurement
            self.recall_measurements.lock().unwrap().push(recall);
        }

        // Return LSH results (so embedding uses LSH, not ground truth)
    }
}
