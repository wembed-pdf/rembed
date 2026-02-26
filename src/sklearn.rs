use std::marker::PhantomData;

use sklearn::{SklearnBallTreeIndex, SklearnKDTreeIndex};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

/// Wrapper for scikit-learn's KDTree implementation
pub struct SklearnKDTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    index: Option<SklearnKDTreeIndex>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, const D: usize> Clone for SklearnKDTree<'a, D> {
    fn clone(&self) -> Self {
        Self::new(&Embedding {
            positions: self.positions.clone(),
            graph: self.graph,
        })
    }
}

impl<'a, const D: usize> SklearnKDTree<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut wrapper = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            index: None,
            _phantom: PhantomData,
        };
        wrapper.update_positions(&embedding.positions, None);
        wrapper
    }

    fn positions_to_flat_array(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.positions.len() * D);
        for pos in &self.positions {
            for &component in &pos.components {
                flat.push(component);
            }
        }
        flat
    }

    pub fn is_valid(&self) -> bool {
        self.index.is_some()
    }

    pub fn point_count(&self) -> usize {
        self.index.as_ref().map(|i| i.point_count()).unwrap_or(0)
    }

    pub fn dimensions(&self) -> usize {
        self.index.as_ref().map(|i| i.dimensions()).unwrap_or(D)
    }
}

impl<'a, const D: usize> Graph for SklearnKDTree<'a, D> {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.graph.is_connected(first, second)
    }

    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.graph.neighbors(index)
    }

    fn weight(&self, index: NodeId) -> f64 {
        self.graph.weight(index)
    }
}

impl<'a, const D: usize> Position<D> for SklearnKDTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for SklearnKDTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();

        if positions.is_empty() {
            self.index = None;
            return;
        }

        let flat_points = self.positions_to_flat_array();
        match SklearnKDTreeIndex::create(&flat_points, positions.len(), D, 40) {
            Ok(idx) => self.index = Some(idx),
            Err(e) => {
                eprintln!("Warning: Failed to create sklearn KDTree: {}", e);
                self.index = None;
            }
        }
    }
}

impl<'a, const D: usize> Query for SklearnKDTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let tree = match &self.index {
            Some(t) => t,
            None => return,
        };

        if index >= self.positions.len() {
            return;
        }

        let own_position = &self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius = radius * own_weight.powi(2);

        let query_point: Vec<f32> = own_position.components.to_vec();

        match tree.radius_search(&query_point, scaled_radius) {
            Ok(result) => {
                for neighbor_idx in result.indices {
                    if neighbor_idx != index {
                        results.push(neighbor_idx);
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: sklearn KDTree query failed: {}", e);
            }
        }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for SklearnKDTree<'a, D> {
    fn name(&self) -> String {
        "sklearn_kdtree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("sklearn.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for SklearnKDTree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}

// Send + Sync: sklearn crate handles GIL acquisition for thread safety
unsafe impl<'a, const D: usize> Send for SklearnKDTree<'a, D> {}
unsafe impl<'a, const D: usize> Sync for SklearnKDTree<'a, D> {}

/// Wrapper for scikit-learn's BallTree implementation
pub struct SklearnBallTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    index: Option<SklearnBallTreeIndex>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, const D: usize> Clone for SklearnBallTree<'a, D> {
    fn clone(&self) -> Self {
        Self::new(&Embedding {
            positions: self.positions.clone(),
            graph: self.graph,
        })
    }
}

impl<'a, const D: usize> SklearnBallTree<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut wrapper = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            index: None,
            _phantom: PhantomData,
        };
        wrapper.update_positions(&embedding.positions, None);
        wrapper
    }

    fn positions_to_flat_array(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.positions.len() * D);
        for pos in &self.positions {
            for &component in &pos.components {
                flat.push(component);
            }
        }
        flat
    }

    pub fn is_valid(&self) -> bool {
        self.index.is_some()
    }

    pub fn point_count(&self) -> usize {
        self.index.as_ref().map(|i| i.point_count()).unwrap_or(0)
    }

    pub fn dimensions(&self) -> usize {
        self.index.as_ref().map(|i| i.dimensions()).unwrap_or(D)
    }
}

impl<'a, const D: usize> Graph for SklearnBallTree<'a, D> {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.graph.is_connected(first, second)
    }

    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.graph.neighbors(index)
    }

    fn weight(&self, index: NodeId) -> f64 {
        self.graph.weight(index)
    }
}

impl<'a, const D: usize> Position<D> for SklearnBallTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for SklearnBallTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();

        if positions.is_empty() {
            self.index = None;
            return;
        }

        let flat_points = self.positions_to_flat_array();
        match SklearnBallTreeIndex::create(&flat_points, positions.len(), D, 40) {
            Ok(idx) => self.index = Some(idx),
            Err(e) => {
                eprintln!("Warning: Failed to create sklearn BallTree: {}", e);
                self.index = None;
            }
        }
    }
}

impl<'a, const D: usize> Query for SklearnBallTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let tree = match &self.index {
            Some(t) => t,
            None => return,
        };

        if index >= self.positions.len() {
            return;
        }

        let own_position = &self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius = radius * own_weight.powi(2);

        let query_point: Vec<f32> = own_position.components.to_vec();

        match tree.radius_search(&query_point, scaled_radius) {
            Ok(result) => {
                for neighbor_idx in result.indices {
                    if neighbor_idx != index {
                        results.push(neighbor_idx);
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: sklearn BallTree query failed: {}", e);
            }
        }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for SklearnBallTree<'a, D> {
    fn name(&self) -> String {
        "sklearn_balltree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("sklearn.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for SklearnBallTree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}

// Send + Sync: sklearn crate handles GIL acquisition for thread safety
unsafe impl<'a, const D: usize> Send for SklearnBallTree<'a, D> {}
unsafe impl<'a, const D: usize> Sync for SklearnBallTree<'a, D> {}
