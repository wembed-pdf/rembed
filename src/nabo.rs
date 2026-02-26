use nabo::KDTree;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct Nabo<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub kdtree: KDTree<f32, DVec<D>>,
    pub max_weights: Vec<f64>,
}

impl<const D: usize> nabo::Point<f32> for DVec<D> {
    fn set(&mut self, i: u32, value: nabo::NotNan<f32>) {
        self[i as usize] = value.into_inner();
    }

    fn get(&self, i: u32) -> nabo::NotNan<f32> {
        // unsafe { nabo::NotNan::new_unchecked(self[i as usize]) }
        nabo::NotNan::new(self[i as usize]).unwrap()
    }

    const DIM: u32 = D as u32;
}

impl<'a, const D: usize> Nabo<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            kdtree: KDTree::new(&embedding.positions),
            max_weights: Vec::new(),
        };
        tree.update_positions(&embedding.positions, None);
        tree
    }
}

impl<'a, const D: usize> Graph for Nabo<'a, D> {
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

impl<'a, const D: usize> Position<D> for Nabo<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for Nabo<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();
        self.kdtree = KDTree::new(positions);
    }
}

impl<'a, const D: usize> Query<D> for Nabo<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let new_results = self.kdtree.knn_advanced(
            self.positions.len() as u32,
            &pos,
            nabo::CandidateContainer::BinaryHeap,
            &nabo::Parameters {
                epsilon: 0.,
                max_radius: radius as f32,
                allow_self_match: false,
                sort_results: false,
            },
            None,
        );
        for neighbour in new_results {
            results.push(neighbour.index as usize);
        }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Nabo<'a, D> {
    fn name(&self) -> String {
        "nabo".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("nabo.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Nabo<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
