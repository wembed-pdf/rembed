use std::sync::Mutex;

use rayon::prelude::*;

use crate::dvec::Vector;
use crate::graph::Graph;
use crate::query::Graph as _;
use crate::NodeId;

use super::EmbedIndex;
use super::dyn_sprk::DynDynSprk;
use super::dyn_vec::DynVec;

/// Caching query wrapper for [`DynDynSprk`], implementing [`EmbedIndex`].
///
/// Dynamic-dimension equivalent of `DynamicQuery<'a, D, Sprk<'a, D>>`.
pub struct DynDynamicQuery<'a> {
    query_cache: Vec<Mutex<Vec<usize>>>,
    structure: DynDynSprk<'a>,
    positions: Vec<DynVec>,
    query_buffer: f64,
    over_query_radius: f64,
    overquery: bool,
    cache_empty: bool,
}

impl Clone for DynDynamicQuery<'_> {
    fn clone(&self) -> Self {
        Self {
            query_cache: empty_cache(self.query_cache.len()),
            structure: self.structure.clone(),
            positions: self.positions.clone(),
            query_buffer: self.query_buffer,
            over_query_radius: self.over_query_radius,
            cache_empty: false,
            overquery: self.overquery,
        }
    }
}

// SAFETY: The Mutex<Vec<usize>> fields are properly synchronized.
unsafe impl Sync for DynDynamicQuery<'_> {}

fn empty_cache(len: usize) -> Vec<Mutex<Vec<usize>>> {
    (0..len).map(|_| Mutex::new(Vec::new())).collect()
}

impl<'a> DynDynamicQuery<'a> {
    pub fn new(dim: usize, positions: &[DynVec], graph: &'a Graph) -> Self {
        let structure = DynDynSprk::new(dim, positions, graph);
        let mut query = DynDynamicQuery {
            query_cache: empty_cache(positions.len()),
            structure,
            positions: vec![],
            query_buffer: 0.,
            over_query_radius: 1.1,
            overquery: false,
            cache_empty: true,
        };
        query.do_update_positions(positions, None);
        query
    }

    fn do_update_positions(&mut self, positions: &[DynVec], last_delta: Option<f64>) {
        if positions.is_empty() {
            return;
        }
        let max_deviation = last_delta.unwrap_or(10.);

        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            for (old, new) in self.positions.iter_mut().zip(positions) {
                old.components.copy_from_slice(&new.components);
            }
        }

        if 1. + max_deviation < self.over_query_radius {
            self.overquery = true;
            self.cache_empty = false;
        } else {
            self.overquery = false;
        }
        if self.query_buffer - max_deviation < 1. {
            self.structure.update_positions(positions, last_delta);
            self.cache_empty = true;
            for cache in self.query_cache.iter_mut() {
                cache.get_mut().unwrap().clear();
            }
            self.query_buffer = self.over_query_radius;
        } else {
            self.query_buffer -= max_deviation;
        }
    }

    fn do_nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<usize>) {
        if !self.overquery {
            return self.structure.nearest_neighbors(index, radius, results);
        }
        assert!(
            radius <= self.query_buffer,
            "query_buffer: {}",
            self.query_buffer
        );
        let mut guard = self.query_cache[index].lock().unwrap();

        let pos = &self.positions[index];
        let weight = self.structure.graph.weight(index);
        let remaining_radius = self.query_buffer;
        let filter = |&id: &usize| {
            index != id
                && (weight > self.structure.graph.weight(id)
                    || (weight == self.structure.graph.weight(id) && index > id))
                && (self.positions[id].distance_squared(pos) as f64)
                    < (weight * self.structure.graph.weight(id)).powi(2) * remaining_radius
                && !self.structure.graph.is_connected(index, id)
        };
        let radius_one = |&id: &usize| {
            (self.positions[id].distance_squared(pos) as f64)
                < (weight * self.structure.graph.weight(id)).powi(2)
        };
        let pos_filter = |&id: &usize| {
            (self.positions[id].distance_squared(pos) as f64)
                < (weight * self.structure.graph.weight(id)).powi(2) * remaining_radius
        };

        if !self.cache_empty {
            guard.retain(pos_filter);
            results.extend(guard.iter().filter(|x| radius_one(x)).cloned());
        } else {
            assert!(guard.is_empty());
            self.structure
                .nearest_neighbors(index, self.over_query_radius, &mut guard);
            guard.retain(filter);
            results.extend(guard.iter().filter(|x| radius_one(x)).cloned());
        }
    }

    fn do_repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        if !self.cache_empty || self.overquery {
            self.do_nearest_neighbors(index, 1., result);
        } else {
            self.do_nearest_neighbors(index, 1., result);
            let pos = &self.positions[index];
            let weight = self.structure.graph.weight(index);

            result.retain(|&x| {
                index != x
                    && (weight > self.structure.graph.weight(x)
                        || weight == self.structure.graph.weight(x) && index > x)
                    && (self.positions[x].distance_squared(pos) as f64)
                        < (weight * self.structure.graph.weight(x)).powi(2)
                    && !self.structure.graph.is_connected(index, x)
            });
        }
    }

    fn nearest_neighbors_batched(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        let n = indices.len();
        // Run all NN queries in parallel
        let per_node: Vec<Vec<usize>> = indices
            .par_iter()
            .map(|&index| {
                let mut owned = Vec::new();
                self.structure.nearest_neighbors(index, 1., &mut owned);
                owned
            })
            .collect();

        // Symmetrize: merge forward edges and reverse edges
        let mut results = vec![vec![]; n];
        for (index, neighbors) in per_node.into_iter().enumerate() {
            for other in neighbors {
                results[index].push(other);
                results[other].push(index);
            }
        }
        results.par_iter_mut().for_each(|vec| {
            vec.sort_unstable();
            vec.dedup();
        });
        results
    }
}

impl EmbedIndex for DynDynamicQuery<'_> {
    type Vec = DynVec;

    fn position(&self, index: NodeId) -> &DynVec {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.structure.positions.len()
    }

    fn weight(&self, index: NodeId) -> f64 {
        self.structure.graph.weight(index)
    }

    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.structure.graph.is_connected(first, second)
    }

    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.structure.graph.neighbors(index)
    }

    fn update_positions(&mut self, positions: &[DynVec], last_delta: Option<f64>) {
        self.do_update_positions(positions, last_delta);
    }

    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        self.do_repelling_nodes(index, result);
    }

    fn graph_statistics(&self) -> (f64, f64) {
        let ids: Vec<_> = (0..self.num_nodes()).collect();
        let results = self.nearest_neighbors_batched(&ids);

        // Count total edges in graph
        let total_edges: usize =
            (0..self.num_nodes()).map(|i| self.neighbors(i).len()).sum::<usize>() / 2;

        // precision, recall
        let (found_edges, found_non_edges) = results
            .par_iter()
            .enumerate()
            .map(|(i, close_nodes)| {
                let mut edges = 0usize;
                let mut non_edges = 0usize;
                for &close_node in close_nodes {
                    if i == close_node {
                        continue;
                    }
                    let within_dist =
                        (self.positions[i].distance_squared(&self.positions[close_node]) as f64)
                            < (self.weight(close_node) * self.weight(i)).powi(2);

                    if self.is_connected(i, close_node) && within_dist {
                        edges += 1;
                    } else if within_dist {
                        non_edges += 1;
                    }
                }
                (edges, non_edges)
            })
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        // Adjust for double-counting (since results are symmetric)
        let found_edges = found_edges / 2;
        let found_non_edges = found_non_edges / 2;

        (
            found_edges as f64 / (found_edges + found_non_edges).max(1) as f64,
            found_edges as f64 / total_edges as f64,
        )
    }
}
