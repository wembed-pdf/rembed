use rstar::{RStarInsertionStrategy, RTree, RTreeParams, primitives::GeomWithData};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
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

#[derive(Clone)]
pub struct LargeNodeParameters;

impl RTreeParams for LargeNodeParameters {
    const MIN_SIZE: usize = 10;
    const MAX_SIZE: usize = 100;
    const REINSERTION_COUNT: usize = 40;
    type DefaultInsertionStrategy = RStarInsertionStrategy;
}

#[derive(Clone)]
pub struct WRTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub rtrees: Vec<RTree<PointPosition<D>, LargeNodeParameters>>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> WRTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            rtrees: Vec::new(),
            max_weights: Vec::new(),
        };
        tree.update_positions(&embedding.positions);
        tree
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

        self.positions = positions.to_vec();

        for i in 0..positions.len() {
            let weight_class = compute_weight_class(self, i);
            if weight_class >= weight_classes.len() {
                weight_classes.resize(weight_class + 1, Vec::new());
                max_weights.resize(weight_class + 1, 0.0);
            }
            weight_classes[weight_class].push(i);
            max_weights[weight_class] = f64::max(max_weights[weight_class], self.weight(i));
        }

        // push to all hevier weight classes
        for i in 0..positions.len() {
            let weight_class = compute_weight_class(self, i);
            for class in weight_classes.iter_mut().skip(weight_class + 1) {
                class.push(i);
            }
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
        let own_position = self.positions[index];
        let own_weight = self.weight(index);

        let weight_class = compute_weight_class(self, index);
        let radius = radius * own_weight.powi(4);

        let mut results = Vec::with_capacity(16);

        for node in
            self.rtrees[weight_class].locate_within_distance(own_position.components, radius as f32)
        {
            if node.data == index {
                continue;
            }

            let node_weight = self.weight(node.data);

            if own_position.distance_squared(&self.positions[node.data])
                <= (own_weight * node_weight).powi(2) as f32
            {
                results.push(node.data);
            }
        }

        results
    }
}
impl<'a, const D: usize> SpatialIndex<D> for WRTree<'a, D> {
    fn name(&self) -> String {
        "Weighted R-Tree".to_string()
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("wrtree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<D> for WRTree<'a, D> {}
