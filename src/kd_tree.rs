use kiddo::{KdTree, SquaredEuclidean};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

fn compute_weight_class(graph: &impl Graph, index: usize) -> usize {
    let weight = graph.neighbors(index).len();
    let mut i = 0;
    while (1 << i) <= weight {
        i += 1;
    }
    i
}

#[derive(Clone)]
pub struct WKDTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub kdtree_classes: Vec<KdTree<f32, D>>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> WKDTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            kdtree_classes: Vec::new(),
            max_weights: Vec::new(),
        };
        tree.update_positions(&embedding.positions);
        tree
    }
}

impl<'a, const D: usize> Graph for WKDTree<'a, D> {
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

impl<'a, const D: usize> Position<D> for WKDTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for WKDTree<'a, D> {
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

        // push to all heavier weight classes
        for i in 0..positions.len() {
            let weight_class = compute_weight_class(self, i);
            for class in weight_classes.iter_mut().skip(weight_class + 1) {
                class.push(i);
            }
        }

        self.max_weights = max_weights;
        self.kdtree_classes.clear();

        for class in weight_classes.iter() {
            let mut tree = KdTree::new();
            for &index in class {
                tree.add(&positions[index].components, index as u64);
            }
            self.kdtree_classes.push(tree);
        }
    }
}

fn squared_euclidean<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

impl<'a, const D: usize> Query for WKDTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let weight_class = compute_weight_class(self, index);
        let scaled_radius_squared = (radius * own_weight.powi(4)).powi(2) as f32;

        let mut results = Vec::with_capacity(16);

        self.kdtree_classes[weight_class]
            .within::<SquaredEuclidean>(&own_position.components, scaled_radius_squared)
            .into_iter()
            .for_each(|nn| {
                let data = nn.item as usize;
                if data == index {
                    return;
                }
                let other_pos = &self.positions[data];
                let other_weight = self.weight(data);
                if own_position.distance_squared(other_pos)
                    <= (own_weight * other_weight).powi(2) as f32
                {
                    results.push(data);
                }
            });

        results
    }
}

impl<'a, const D: usize> SpatialIndex<D> for WKDTree<'a, D> {
    fn name(&self) -> String {
        "Weighted KD-Tree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("kd_tree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<D> for WKDTree<'a, D> {}
