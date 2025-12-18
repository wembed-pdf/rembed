use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::prelude::DistL2;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

pub struct HNSWTree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub hnsw: Hnsw<
        'a,
        f32,
        DistL2, // DistL2 implements Distance<[f32]>
    >,
}

impl<'a, const D: usize> Clone for HNSWTree<'a, D> {
    fn clone(&self) -> Self {
        let positions = self.positions.clone();
        let hnsw = Hnsw::new(32, positions.len(), 4, 100, DistL2::default());
        for (i, pos) in positions.iter().enumerate() {
            hnsw.insert((&pos.components, i));
        }
        Self {
            positions,
            graph: self.graph,
            hnsw,
        }
    }
}

impl<'a, const D: usize> HNSWTree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let positions = embedding.positions.clone();
        let mut tree = Self {
            positions,
            graph: embedding.graph,
            hnsw: Hnsw::new(5, embedding.positions.len(), 3, 100, DistL2::default()),
        };
        tree.update_positions(&embedding.positions);
        tree
    }
}

impl<'a, const D: usize> Graph for HNSWTree<'a, D> {
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

impl<'a, const D: usize> Position<D> for HNSWTree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for HNSWTree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
        self.hnsw = Hnsw::new(8, self.positions.len(), 3, 10, DistL2::default());

        let data: Vec<(Vec<f32>, usize)> = positions
            .iter()
            .enumerate()
            .map(|(i, pos)| (pos.components.to_vec(), i))
            .collect();
        let data_refs: Vec<(&Vec<f32>, usize)> = data.iter().map(|(v, i)| (v, *i)).collect();
        self.hnsw.parallel_insert(data_refs.as_slice());
    }
}

impl<'a, const D: usize> Query for HNSWTree<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius_squared = (radius * own_weight.powi(2)) as f32;

        if self.weight(index) > 1.2 {
            // bruteforce search if own weight is too high
            // self.positions.iter().enumerate().for_each(|(i, pos)| {
            //     if i != index
            //         && own_position.distance_squared(pos)
            //             <= (own_weight * self.weight(i)).powi(2) as f32
            //     {
            //         results.push(i);
            //     }
            // });
            let mut knbn = 50; // initial number of neighbors to search for
            let ef_search = 100;

            let neighbors = self.hnsw.search(&own_position.components, knbn, ef_search);
            for neighbor in neighbors {
                results.push(neighbor.d_id);
            }
            return;
        }

        // if self.weight(index) > 1.0 {
        //     self.positions.iter().enumerate().for_each(|(i, d)| {
        //         results.push(i);
        //     });
        // } else {
        let mut knbn = 5; // initial number of neighbors to search for
        let ef_search = 10;

        let neighbors = self.hnsw.search(&own_position.components, knbn, ef_search);
        for neighbor in neighbors {
            results.push(neighbor.d_id);
        }
        // }

        // let mut knbn = 1; // initial number of neighbors to search for
        // let mut ef_search = 1; // size of the dynamic candidate list
        // let max_knbn = 1; // Limit to prevent infinite loop

        // loop {
        //     let neighbors = self.hnsw.search(&own_position.components, knbn, ef_search);

        //     // How many neighbors are inside radius?
        //     let mut outside_count = 0;
        //     for neighbor in neighbors {
        //         let data = neighbor.d_id;
        //         if results.contains(&data) {
        //             continue; // Skip already found neighbors
        //         }
        //         if data == index {
        //             continue; // Skip self
        //         }
        //         let other_pos = &self.positions[data];
        //         let other_weight = self.weight(data);
        //         if own_position.distance_squared(other_pos)
        //             <= (own_weight * other_weight).powi(2) as f32
        //         {
        //             results.push(data);
        //         } else {
        //             outside_count += 1;
        //         }
        //     }
        //     break;

        //     if outside_count > 1 || knbn >= max_knbn {
        //         break; // If too many neighbors are outside the radius, stop searching
        //     }

        //     // Otherwise, increase knbn exponentially and try again
        //     knbn = (knbn * 2).min(max_knbn);
        //     ef_search = (ef_search * 2).min(max_knbn);
        // }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for HNSWTree<'a, D> {
    fn name(&self) -> String {
        "hnsw".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("hnsw.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for HNSWTree<'a, D> {
    fn new(embedding: &Embedding<'a, D>) -> Self {
        HNSWTree::new(embedding.clone())
    }
}
