use acap::{NearestNeighbors, vp::FlatVpTree};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone, Debug)]
pub struct DataPoint<const D: usize> {
    index: usize,
    position: DVec<D>,
}

impl<const D: usize> acap::Proximity for DataPoint<D> {
    type Distance = f32;

    fn distance(&self, other: &Self) -> Self::Distance {
        self.position.distance(&other.position)
    }
}

pub struct VPTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub vptree: FlatVpTree<DataPoint<D>>,
}

impl<'a, const D: usize> Clone for VPTree<'a, D> {
    fn clone(&self) -> Self {
        let data = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, pos)| DataPoint {
                index: i,
                position: pos.clone(),
            })
            .collect::<Vec<_>>();
        let vptree = FlatVpTree::balanced(data);
        Self {
            positions: self.positions.clone(),
            graph: self.graph,
            vptree,
        }
    }
}

impl<'a, const D: usize> VPTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let positions = embedding.positions.clone();
        let data_points = positions
            .iter()
            .enumerate()
            .map(|(i, pos)| DataPoint {
                index: i,
                position: pos.clone(),
            })
            .collect::<Vec<_>>();
        let vptree = FlatVpTree::balanced(data_points);
        Self {
            positions,
            graph: embedding.graph,
            vptree,
        }
    }
}

impl<'a, const D: usize> Graph for VPTree<'a, D> {
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

impl<'a, const D: usize> Position<D> for VPTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for VPTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
        let data_points = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, pos)| DataPoint {
                index: i,
                position: pos.clone(),
            })
            .collect::<Vec<_>>();
        self.vptree = FlatVpTree::balanced(data_points);
    }
}

impl<'a, const D: usize> Query for VPTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius_squared = (radius * own_weight.powi(2)) as f32;

        let query_point = DataPoint {
            index,
            position: own_position,
        };

        self.vptree
            .k_nearest_within(&query_point, self.positions.len(), scaled_radius_squared)
            .into_iter()
            .filter(|nn| {
                let data = nn.item.index;
                if data == index {
                    return false;
                }
                let other_pos = &self.positions[data];
                let other_weight = self.weight(data);
                own_position.distance_squared(other_pos)
                    <= (own_weight * other_weight).powi(2) as f32
            })
            .for_each(|nn| {
                results.push(nn.item.index);
            });
    }
}

impl<'a, const D: usize> SpatialIndex<D> for VPTree<'a, D> {
    fn name(&self) -> String {
        "vptree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("vptree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for VPTree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
