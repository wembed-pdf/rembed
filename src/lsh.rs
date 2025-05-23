use std::collections::{HashMap, HashSet};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, Update},
};

#[derive(Clone)]
pub struct Lsh<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub weight_threshold: f64,
    pub map: HashMap<[i32; D], Vec<NodeId>>,
}

impl<'a, const D: usize> crate::query::Graph for Lsh<'a, D> {
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
impl<'a, const D: usize> query::Position<D> for Lsh<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
}
impl<'a, const D: usize> query::Update<D> for Lsh<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        self.positions = postions.to_vec();
        for (id, pos) in self.positions.iter().enumerate() {
            for rounded_pos in nstar(pos) {
                self.map.entry(rounded_pos).or_default().push(id);
            }
        }
    }
}

impl<const D: usize> Query for Lsh<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        if self.weight(index) >= self.weight_threshold {
            self.heavy_nn(index, radius)
        } else {
            self.light_nn(index)
        }
    }
}

impl<'a, const D: usize> Lsh<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let Embedding { positions, graph } = embedding;
        let mut new = Self {
            positions: positions.clone(),
            graph,
            weight_threshold: 1.,
            // TODO: use different hash function
            map: HashMap::with_capacity(positions.len()),
        };
        new.update_positions(&positions);
        new
    }

    fn heavy_nn(&self, index: usize, radius: f64) -> Vec<usize> {
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

    fn light_nn(&self, index: usize) -> Vec<usize> {
        let neighbors: HashSet<NodeId> = nstar(self.position(index))
            .flat_map(|x| self.map.get(&x))
            .flatten()
            .copied()
            .collect();
        neighbors.into_iter().collect()
    }
}

fn nstar<const D: usize>(pos: &DVec<D>) -> impl Iterator<Item = [i32; D]> {
    let start = pos.map(|x| x.round());
    let iter = (0..D).flat_map(move |x| [start + DVec::unit(x), start - DVec::unit(x)]);
    iter.chain(std::iter::once(start)).map(|x| x.round())
}

impl<'a, const D: usize> query::Embedder<D> for Lsh<'a, D> {}
