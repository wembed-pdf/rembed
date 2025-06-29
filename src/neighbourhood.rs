use std::collections::HashMap;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
use neighbourhood::KdTree;

pub struct Neihbourhood<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub tree: KdTree<f32, D>,
    pub map: HashMap<[u32; D], NodeId>,
}

impl<'a, const D: usize> Clone for Neihbourhood<'a, D> {
    fn clone(&self) -> Self {
        let mut points: Vec<[f32; D]> = Vec::with_capacity(self.positions.len());
        for pos in self.positions.iter() {
            let point: [f32; D] = pos.components.map(|x| x as f32);
            points.push(point);
        }
        let tree = KdTree::new(points);
        Self {
            positions: self.positions.clone(),
            graph: self.graph,
            tree,
            map: self.map.clone(),
        }
    }
}

impl<'a, const D: usize> Neihbourhood<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            tree: KdTree::new(vec![[0. as f32; D]; 2]),
            map: HashMap::new(),
        };
        tree.update_positions(&embedding.positions);
        tree
    }
}

impl<'a, const D: usize> Graph for Neihbourhood<'a, D> {
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

impl<'a, const D: usize> Position<D> for Neihbourhood<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for Neihbourhood<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
        self.map.clear();
        let mut points: Vec<[f32; D]> = Vec::with_capacity(self.positions.len());
        for pos in self.positions.iter() {
            let point: [f32; D] = pos.components.map(|x| x as f32);
            self.map.insert(
                core::array::from_fn(|i| point[i].to_bits()),
                self.map.len() as NodeId,
            );
            points.push(point);
        }
        self.tree = KdTree::new(points);
    }
}

impl<'a, const D: usize> Query for Neihbourhood<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius_squared = (radius * own_weight.powi(2)) as f32;

        self.tree
            .neighbourhood(&own_position.components, scaled_radius_squared)
            .into_iter()
            .for_each(|nn| {
                if let Some(&node_id) = self.map.get(&core::array::from_fn(|i| nn[i].to_bits())) {
                    if node_id != index {
                        let other_pos = &self.positions[node_id];
                        let other_weight = self.weight(node_id);
                        if own_position.distance_squared(other_pos)
                            <= (own_weight * other_weight).powi(2) as f32
                        {
                            results.push(node_id as usize);
                        }
                    }
                }
            });
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Neihbourhood<'a, D> {
    fn name(&self) -> String {
        "neighbourhood".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("neighbourhood.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Neihbourhood<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
