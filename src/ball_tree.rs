use ball_tree::BallTree;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct WBallTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub ball_tree: BallTree<[f64; D], usize>,
}

impl<'a, const D: usize> WBallTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let empty_positions = Vec::new();
        let empty_indices = Vec::new();
        let ball_tree = BallTree::new(empty_positions, empty_indices);
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            ball_tree: ball_tree,
        };
        tree.update_positions(&embedding.positions);
        tree
    }
}

impl<'a, const D: usize> Graph for WBallTree<'a, D> {
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

impl<'a, const D: usize> Position<D> for WBallTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for WBallTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
        let ball_tree_positions: Vec<[f64; D]> = self
            .positions
            .iter()
            .map(|pos| pos.components.map(|x| x as f64))
            .collect();
        let ball_tree_indices: Vec<usize> = (0..self.positions.len()).collect();
        self.ball_tree = BallTree::new(ball_tree_positions, ball_tree_indices);
    }
}

impl<'a, const D: usize> Query for WBallTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius_squared = (radius * own_weight.powi(2)) as f32;

        let mut query = self.ball_tree.query();

        let query_position: [f64; D] = own_position.components.map(|x| x as f64);
        query
            .nn_within(&query_position, scaled_radius_squared as f64)
            .into_iter()
            .for_each(|nn| {
                let data = *nn.2;
                let other_pos = &self.positions[data];
                let other_weight = self.weight(data);
                if own_position.distance_squared(other_pos)
                    <= (own_weight * other_weight).powi(2) as f32
                {
                    results.push(data);
                }
            });
    }
}

impl<'a, const D: usize> SpatialIndex<D> for WBallTree<'a, D> {
    fn name(&self) -> String {
        "ball_tree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("ball_tree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for WBallTree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
