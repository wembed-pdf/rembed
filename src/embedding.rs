use crate::{
    NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex},
};

#[derive(Clone)]
pub struct Embedding<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
}

impl<'a, const D: usize> crate::query::Graph for Embedding<'a, D> {
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
impl<'a, const D: usize> query::Position<D> for Embedding<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}
impl<'a, const D: usize> query::Update<D> for Embedding<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>], _: Option<f64>) {
        self.positions = postions.to_vec();
    }
}

impl<const D: usize> Query for Embedding<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let graph = self.graph;
        let own_weight = self.weight(index);
        let own_position = self.position(index);

        for (i, (node, position)) in graph
            .nodes
            .iter()
            .zip(self.positions.iter())
            .enumerate()
            .take(index)
        {
            let weight = own_weight * node.weight;
            let distance = own_position.distance_squared(position);
            if (distance as f64) < weight.powi(2) * radius {
                results.push(i);
            }
        }
    }
}
impl<const D: usize> SpatialIndex<D> for Embedding<'_, D> {
    fn name(&self) -> String {
        String::from("brute-force")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("embedding.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Embedding<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        embedding.clone()
    }
}
