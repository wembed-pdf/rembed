use rstar::{RStarInsertionStrategy, RTree, RTreeParams, primitives::GeomWithData};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, Update},
};

type PointPosition<const D: usize> = GeomWithData<[f32; D], usize>;

fn compute_weight_class(graph: &impl Graph, index: usize) -> usize {
    let weight = graph.neighbors(index).len();
    let mut i = 0;
    while (1 << i) <= weight {
        i += 1;
    }
    i
}

pub struct LargeNodeParameters;

impl RTreeParams for LargeNodeParameters {
    const MIN_SIZE: usize = 10;
    const MAX_SIZE: usize = 100;
    const REINSERTION_COUNT: usize = 40;
    type DefaultInsertionStrategy = RStarInsertionStrategy;
}

pub struct WRTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a dyn Graph,
    pub rtrees: Vec<RTree<PointPosition<D>, LargeNodeParameters>>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> WRTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            rtrees: Vec::new(),
            max_weights: Vec::new(),
        }
    }
}

impl<'a, const D: usize> Graph for WRTree<'a, D> {
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

impl<'a, const D: usize> Position<D> for WRTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
}

impl<'a, const D: usize> Update<D> for WRTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        let mut weight_classes: Vec<Vec<usize>> = Vec::new();
        let mut max_weights: Vec<f64> = Vec::new();

        for (i, position) in positions.iter().enumerate() {
            self.positions[i] = *position;
            let weight_class = compute_weight_class(self, i);
            if weight_class >= weight_classes.len() {
                weight_classes.resize(weight_class + 1, Vec::new());
                max_weights.resize(weight_class + 1, 0.0);
            }
            weight_classes[weight_class].push(i);
            max_weights[weight_class] = f64::max(max_weights[weight_class], self.weight(i));
        }

        self.max_weights = max_weights;
        self.rtrees.clear();
        for weight_class in weight_classes.iter() {
            let entries = weight_class
                .iter()
                .map(|&index| PointPosition::new(positions[index].components, index))
                .collect::<Vec<_>>();
            self.rtrees.push(RTree::bulk_load_with_params(entries));
        }
    }
}

impl<'a, const D: usize> Query for WRTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let own_position = self.positions[index].components;
        let own_weight = self.weight(index);
        let mut result = Vec::new();

        for weight_class in 0..self.rtrees.len() {
            // Calculate the radius based on the maximum weight of the class
            let radius = (own_weight * self.max_weights[weight_class]).powi(2);

            result.extend(
                self.rtrees[weight_class]
                    .locate_within_distance(own_position, radius as f32)
                    .map(|node| node.data)
                    .collect::<Vec<_>>(),
            );
        }

        result.retain(|&node| {
            // Check the distance to the own node
            let own_weight = self.weight(index);
            let own_position = self.positions[index];
            let distance = own_position.distance_squared(&self.positions[node]);
            let weight = own_weight * self.weight(node);
            distance < weight.powi(2) as f32 && node != index
        });
        result

        // let query_radius =
        //     radius * (self.weight(index) * self.max_weights[weight_class] as f64).powi(2);
        // self.rtrees
        //     .iter()
        //     .flat_map(|tree| {
        //         tree.locate_within_distance(pos, query_radius)
        //             .map(|pt| pt.data)
        //     })
        //     .collect::<Vec<usize>>()
        //     .into_iter()
        //     .filter(|&id| id != index)
        //     .collect()
    }

    fn name(&self) -> String {
        "Weighted R-Tree".to_string()
    }
}

impl<'a, const D: usize> query::Embedder<D> for WRTree<'a, D> {}
