use crate::{NodeId, Query, dvec::DVec, query};

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
impl<const D: usize> Query for Embedding<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let mut output = Vec::new();
        let graph = self.graph;
        let positions = &self.positions;
        let own_weight = graph.nodes[index].weight;
        let own_position = positions[index];

        for (i, (node, position)) in graph.nodes.iter().zip(positions.iter()).enumerate() {
            let weight = own_weight * node.weight;
            let distance = own_position.distance_squared(position);
            if distance < weight.powi(2) * radius {
                output.push(i);
            }
        }
        output
    }
}
impl<'a, const D: usize> query::Update<D> for Embedding<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        self.positions = postions.to_vec();
    }
}
impl<'a, const D: usize> query::Position<D> for Embedding<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
}

impl<'a, const D: usize> query::Embedder<D> for Embedding<'a, D> {}
