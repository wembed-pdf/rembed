use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
use kdtree::KdTree;

#[derive(Clone)]
pub struct KDTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub tree: KdTree<f32, usize, [f32; D]>,
}

impl<'a, const D: usize> KDTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            tree: KdTree::new(0),
        };
        tree.update_positions(&embedding.positions);
        tree
    }
}

impl<'a, const D: usize> Graph for KDTree<'a, D> {
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

impl<'a, const D: usize> Position<D> for KDTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for KDTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
        self.tree = KdTree::new(D);
        for (i, pos) in self.positions.iter().enumerate() {
            // Convert DVec to [f32; D] for the KdTree
            let point: [f32; D] = pos.components.map(|x| x as f32);
            self.tree
                .add(point, i)
                .expect("Failed to add point to KdTree");
        }
    }
}

impl<'a, const D: usize> Query for KDTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius_squared = (radius * own_weight.powi(4)) as f32;

        let mut results = Vec::with_capacity(16);

        self.tree
            .within(
                &own_position.components,
                scaled_radius_squared,
                &kdtree::distance::squared_euclidean,
            )
            .into_iter()
            .for_each(|nn| {
                for data in nn.iter() {
                    let data_index = *data.1 as usize;
                    if data_index == index {
                        continue; // Skip the own node
                    }
                    let other_pos = &self.positions[data_index];
                    let other_weight = self.weight(data_index);
                    if own_position.distance_squared(other_pos)
                        <= (own_weight * other_weight).powi(2) as f32
                    {
                        results.push(data_index);
                    }
                }
            });

        results
    }
}

impl<'a, const D: usize> SpatialIndex<D> for KDTree<'a, D> {
    fn name(&self) -> String {
        "kd_tree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("kd_tree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for KDTree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
