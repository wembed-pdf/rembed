use kiddo::{ImmutableKdTree, SquaredEuclidean};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct Kiddo<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub kdtree: ImmutableKdTree<f32, D>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> Kiddo<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            kdtree: ImmutableKdTree::new_from_slice(&[]),
            max_weights: Vec::new(),
        };
        tree.update_positions(&embedding.positions, None);
        tree
    }
}

impl<'a, const D: usize> Graph for Kiddo<'a, D> {
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

impl<'a, const D: usize> Position<D> for Kiddo<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for Kiddo<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        let second_positions: Vec<_> = positions.iter().map(|p| p.components).collect();
        self.positions = positions.to_vec();
        self.kdtree = ImmutableKdTree::new_from_slice(&second_positions);
        // for (i, pos) in self.positions.iter().enumerate() {
        //     self.kdtree.add(&pos.components, i as u64);
        // }
    }
}

impl<'a, const D: usize> Query for Kiddo<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius_squared = (radius * own_weight.powi(2)).powi(2) as f32;

        self.kdtree
            .within_unsorted::<SquaredEuclidean>(&own_position.components, scaled_radius_squared)
            .into_iter()
            .for_each(|nn| {
                let data = nn.item as usize;
                if data == index {
                    return;
                }
                results.push(data);
                // let other_weight = self.weight(data);
                // if nn.distance <= (own_weight * other_weight).powi(2) as f32 {
                //     results.push(data);
                // }
            });
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Kiddo<'a, D> {
    fn name(&self) -> String {
        "kiddo".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("kiddo.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Kiddo<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
