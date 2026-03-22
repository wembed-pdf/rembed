use crate::{
    Embedding, NodeId,
    dvec::DVec,
    query::{self, SpatialIndex},
};

pub use atree::simd;

#[derive(Clone)]
pub struct ATree<'a, const D: usize> {
    pub tree: atree::ATree<D>,
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
}

impl<const D: usize> crate::query::Graph for ATree<'_, D> {
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

impl<const D: usize> query::Position<D> for ATree<'_, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<const D: usize> query::Update<D> for ATree<'_, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            for (old_pos, pos) in self.positions.iter_mut().zip(positions.iter()) {
                *old_pos = *pos;
            }
        }

        let raw_positions: Vec<[f32; D]> = positions.iter().map(|p| p.components).collect();
        self.tree.update(&raw_positions);
    }
}

impl<const D: usize> crate::Query<D> for ATree<'_, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        assert_eq!(self.positions.len(), self.tree.len());
        self.tree.query_radius(&pos.components, radius as f32, results);
    }
}

impl<const D: usize> SpatialIndex<D> for ATree<'_, D> {
    fn name(&self) -> String {
        String::from("atree")
    }
    fn implementation_string(&self) -> &'static str {
        concat!(
            include_str!("../atree/src/lib.rs"),
            include_str!("../atree/src/simd.rs")
        )
    }
}

impl<'a, const D: usize> ATree<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let raw_positions: Vec<[f32; D]> =
            embedding.positions.iter().map(|p| p.components).collect();
        ATree {
            tree: atree::ATree::new(&raw_positions),
            positions: embedding.positions.clone(),
            graph: embedding.graph,
        }
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for ATree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}
