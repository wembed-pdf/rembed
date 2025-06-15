use sif_kdtree::KdTree;
use sif_kdtree::Object;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

pub struct Data<const D: usize>(usize, [f64; D]);

impl<const D: usize> Object for Data<D> {
    type Point = [f64; D];

    fn position(&self) -> &Self::Point {
        &self.1
    }
}

pub struct SIF<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub kdtree: KdTree<Data<D>, Vec<Data<D>>>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> SIF<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let index = KdTree::new(
            embedding
                .positions
                .iter()
                .enumerate()
                .map(|(i, pos)| Data::<D>(i, pos.components.map(|x| x as f64)))
                .collect::<Vec<_>>(),
        );
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            kdtree: index,
            max_weights: Vec::new(),
        };
        tree.update_positions(&embedding.positions);
        tree
    }
}

impl<'a, const D: usize> Clone for SIF<'a, D> {
    fn clone(&self) -> Self {
        let kdtree = KdTree::new(
            self.kdtree
                .iter()
                .map(|data| Data::<D>(data.0, data.1))
                .collect::<Vec<_>>(),
        );
        Self {
            positions: self.positions.clone(),
            graph: self.graph,
            kdtree: kdtree,
            max_weights: self.max_weights.clone(),
        }
    }
}

impl<'a, const D: usize> Graph for SIF<'a, D> {
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

impl<'a, const D: usize> Position<D> for SIF<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for SIF<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
        self.kdtree = KdTree::new(
            self.positions
                .iter()
                .enumerate()
                .map(|(i, pos)| Data::<D>(i, pos.components.map(|x| x as f64)))
                .collect::<Vec<_>>(),
        );
    }
}

impl<'a, const D: usize> Query for SIF<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius = radius * own_weight.powi(2);

        let mut results = Vec::with_capacity(16);

        // Convert own_position to [f64; D]
        let own_position_f64: [f64; D] = own_position.components.map(|x| x as f64);

        let _ = self.kdtree.look_up(
            &sif_kdtree::WithinDistance::new(own_position_f64, scaled_radius),
            |nn| {
                let data = nn.0 as usize;
                if data != index {
                    let other_pos = &self.positions[data];
                    let other_weight = self.weight(data);
                    if own_position.distance_squared(other_pos)
                        <= (own_weight * other_weight).powi(2) as f32
                    {
                        results.push(data);
                    }
                }
                std::ops::ControlFlow::Continue::<()>(())
            },
        );
        results
    }
}

impl<'a, const D: usize> SpatialIndex<D> for SIF<'a, D> {
    fn name(&self) -> String {
        "sif".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("sif.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for SIF<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
