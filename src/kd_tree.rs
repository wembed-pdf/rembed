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
        tree.update_positions(&embedding.positions, None);
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
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();
        self.tree = KdTree::new(D);
        for (i, pos) in self.positions.iter().enumerate() {
            // Convert DVec to [f32; D] for the KdTree
            let point: [f32; D] = pos.components.map(|x| x);
            self.tree
                .add(point, i)
                .expect("Failed to add point to KdTree");
        }
    }
}

impl<'a, const D: usize> Query<D> for KDTree<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<usize>) {
        let radius_squared = (radius * radius) as f32;
        self.tree
            .within(
                &pos.components,
                radius_squared,
                &kdtree::distance::squared_euclidean,
            )
            .into_iter()
            .for_each(|nn| {
                for data in nn.iter() {
                    results.push(*data.1);
                }
            });
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
