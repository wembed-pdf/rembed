use crate::graph::Graph;
use crate::query::Graph as _;
use crate::NodeId;

use super::dyn_vec::DynVec;

/// Dynamic-dimension spatial index wrapping `sprk::DynSprk`.
#[derive(Clone)]
pub struct DynDynSprk<'a> {
    pub tree: sprk::DynSprk,
    pub positions: Vec<DynVec>,
    pub graph: &'a Graph,
    _dim: usize,
}

impl<'a> DynDynSprk<'a> {
    pub fn new(dim: usize, positions: &[DynVec], graph: &'a Graph) -> Self {
        let flat: Vec<f32> = positions
            .iter()
            .flat_map(|p| &p.components)
            .copied()
            .collect();
        DynDynSprk {
            tree: sprk::DynSprk::new(dim, &flat),
            positions: positions.to_vec(),
            graph,
            _dim: dim,
        }
    }

    pub fn update_positions(&mut self, positions: &[DynVec], _last_delta: Option<f64>) {
        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            for (old, new) in self.positions.iter_mut().zip(positions) {
                old.components.copy_from_slice(&new.components);
            }
        }
        let flat: Vec<f32> = positions
            .iter()
            .flat_map(|p| &p.components)
            .copied()
            .collect();
        self.tree.update(&flat);
    }

    pub fn query_radius(&self, pos: &DynVec, radius: f64, results: &mut Vec<NodeId>) {
        self.tree
            .query_radius(&pos.components, radius as f32, results);
    }

    pub fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let pos = &self.positions[index];
        let scaled_radius = radius * self.graph.weight(index).powi(2);
        self.query_radius(pos, scaled_radius, results);
    }
}
