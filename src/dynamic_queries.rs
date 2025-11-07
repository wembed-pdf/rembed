use std::sync::Mutex;

use crate::{
    NodeId, Query,
    atree::ATree,
    dvec::DVec,
    query::{self, IndexClone, SpatialIndex, Update},
};

pub struct DynamicQuery<'a, const D: usize> {
    query_cache: Vec<Mutex<Option<Vec<usize>>>>,
    structure: Box<dyn IndexClone<D> + 'a + Sync>,
    positions: Vec<DVec<D>>,
    query_buffer: f64,
    over_query_radius: f64,
    overquery: bool,
}

impl<'a, const D: usize> Clone for DynamicQuery<'a, D> {
    fn clone(&self) -> Self {
        Self {
            query_cache: empty_cache(self.query_cache.len()),
            structure: self.structure.clone_box_cloneable(),
            positions: self.positions.clone(),
            query_buffer: self.query_buffer.clone(),
            over_query_radius: self.over_query_radius.clone(),
            overquery: self.overquery,
        }
    }
}

fn empty_cache(len: usize) -> Vec<Mutex<Option<Vec<usize>>>> {
    (0..len).map(|_| Mutex::new(None)).collect()
}

impl<'a, const D: usize> crate::query::Graph for DynamicQuery<'a, D> {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.structure.is_connected(first, second)
    }

    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.structure.neighbors(index)
    }

    fn weight(&self, index: NodeId) -> f64 {
        self.structure.weight(index)
    }
}

impl<'a, const D: usize> query::Position<D> for DynamicQuery<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}
impl<'a, const D: usize> query::Update<D> for DynamicQuery<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        if postions.is_empty() {
            return;
        }
        let max_deviation = self
            .positions
            .iter()
            .zip(postions)
            .map(|(old, new)| old.distance(new))
            .max_by(|a, b| f32::total_cmp(a, b))
            .unwrap_or_default() as f64
            * 2.;

        self.structure.update_positions(postions);
        self.positions = postions.to_vec();
        if 1. + max_deviation < self.over_query_radius {
            self.overquery = true;
        } else {
            return;
        }
        if self.query_buffer - max_deviation < 1. {
            // println!("recomputing after pos diff {}", max_deviation);
            for cache in self.query_cache.iter_mut() {
                let mut guard = cache.lock().unwrap();
                *guard = None;
            }
            // self.query_cache = {
            //     let indices: &[usize] = &(0..postions.len()).collect::<Vec<_>>();
            //     let mut results = vec![vec![]; indices.len()];
            //     for &index in indices {
            //         for other_node_id in self
            //             .structure
            //             .nearest_neighbors_owned(index, self.over_query_radius as f64)
            //         {
            //             results[other_node_id].push(index);
            //             results[index].push(other_node_id);
            //         }
            //     }
            //     for vec in &mut results {
            //         vec.sort_unstable();
            //         vec.dedup();
            //     }
            //     results
            // };

            self.query_buffer = self.over_query_radius;
        } else {
            // println!("reusing previous query");
            self.query_buffer -= max_deviation;
        }
    }
}

impl<'a, const D: usize> Query for DynamicQuery<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<usize>) {
        if !self.overquery {
            return self.structure.nearest_neighbors(index, radius, results);
        }
        assert!(
            radius <= self.query_buffer,
            "query_buffer: {}",
            self.query_buffer
        );
        let mut guard = self.query_cache[index].lock().unwrap();
        if let Some(nodes) = &*guard {
            results.extend_from_slice(nodes);
        } else {
            let new_nodes = self
                .structure
                .nearest_neighbors_owned(index, self.over_query_radius as f64);
            results.extend_from_slice(&new_nodes);
            *guard = Some(new_nodes);
        }
    }
}
impl<'a, const D: usize> SpatialIndex<D> for DynamicQuery<'a, D> {
    fn name(&self) -> String {
        String::from("dynamic queries")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("dynamic_queries.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for DynamicQuery<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        let mut query = DynamicQuery {
            query_cache: empty_cache(embedding.positions.len()),
            structure: Box::new(ATree::new(embedding)),
            positions: vec![],
            query_buffer: 0.,
            over_query_radius: 1.1,
            overquery: false,
        };
        query.update_positions(&embedding.positions);
        // assert_ne!(query.query_cache.len(), 0);
        query
    }
}
