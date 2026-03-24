use std::sync::Mutex;

use crate::{
    NodeId, Query,
    dvec::DVec,
    query::{self, Embedder, Graph, Position, SpatialIndex, Update},
};

pub struct DynamicQuery<'a, const D: usize, ID: Embedder<'a, D>> {
    query_cache: Vec<Mutex<Vec<(usize, f32)>>>,
    structure: ID,
    positions: Vec<DVec<D>>,
    query_buffer: f64,
    over_query_radius: f64,
    overquery: bool,
    cache_empty: bool,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, const D: usize, ID: Embedder<'a, D> + Clone> Clone for DynamicQuery<'a, D, ID> {
    fn clone(&self) -> Self {
        Self {
            query_cache: empty_cache(self.query_cache.len()),
            structure: self.structure.clone(),
            positions: self.positions.clone(),
            query_buffer: self.query_buffer,
            over_query_radius: self.over_query_radius,
            cache_empty: false,
            overquery: self.overquery,
            _phantom: std::marker::PhantomData,
        }
    }
}

fn empty_cache(len: usize) -> Vec<Mutex<Vec<(usize, f32)>>> {
    (0..len).map(|_| Mutex::new(Vec::new())).collect()
}

impl<'a, const D: usize, ID: Embedder<'a, D>> crate::query::Graph for DynamicQuery<'a, D, ID> {
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

impl<'a, const D: usize, ID: Embedder<'a, D>> query::Position<D> for DynamicQuery<'a, D, ID> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.structure.num_nodes()
    }
}
impl<'a, const D: usize, ID: Embedder<'a, D>> query::Update<D> for DynamicQuery<'a, D, ID> {
    fn update_positions(&mut self, positions: &[DVec<D>], last_delta: Option<f64>) {
        if positions.is_empty() {
            return;
        }
        // let max_deviation = last_delta.unwrap_or(10.) * 2.;
        // let max_deviation = last_delta.unwrap_or(10.) * 2.;
        let max_deviation = last_delta.unwrap_or(10.);

        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            for (old_pos, pos) in self.positions.iter_mut().zip(positions.iter()) {
                *old_pos = *pos;
            }
        }

        // self.positions = postions.to_vec();
        if 1. + max_deviation < self.over_query_radius {
            // println!("over query");
            self.overquery = true;
            self.cache_empty = false;
        } else {
            self.overquery = false;
            // self.structure.update_positions(positions, last_delta);
            // return;
        }
        // return;
        if self.query_buffer - max_deviation < 1. {
            self.structure.update_positions(positions, last_delta);
            // println!("recomputing after pos diff {}", max_deviation);
            self.cache_empty = true;
            for cache in self.query_cache.iter_mut() {
                cache.get_mut().unwrap().clear();
            }

            self.query_buffer = self.over_query_radius;
        } else {
            // println!("reusing previous query");
            self.query_buffer -= max_deviation;
        }
    }
}

impl<'a, const D: usize, ID: Embedder<'a, D>> Query<D> for DynamicQuery<'a, D, ID> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        self.structure.query_radius(pos, radius, results);
    }
    fn nearest_neighbors_with_distances(
        &self,
        index: usize,
        radius: f64,
        results: &mut Vec<NodeId>,
    ) -> impl Iterator<Item = (usize, f32)>
    where
        Self: Sized,
    {
        if !self.overquery {
            // TODO find out why this assert fails
            // assert!(self.query_cache[index].lock().unwrap().is_empty());
            let collected: Vec<_> = self
                .structure
                .nearest_neighbors_with_distances(index, radius, results)
                .collect();
            return collected.into_iter();
        }
        assert!(
            radius <= self.query_buffer,
            "query_buffer: {}",
            self.query_buffer
        );
        let mut guard = self.query_cache[index].lock().unwrap();

        let pos = self.position(index);
        let weight = self.weight(index);
        let remaining_radius = self.query_buffer;
        let filter = |&(id, dist): &(usize, f32)| {
            !self.structure.is_connected(index, id)
                && (weight > self.weight(id) || (weight == self.weight(id) && index > id))
                && (dist as f64) < (weight * self.weight(id)).powi(2) * remaining_radius
        };
        let radius_one = |&(id, _): &(usize, f32)| {
            let dist = self.position(id).distance_squared(pos);
            ((dist as f64) < (weight * self.weight(id)).powi(2)).then(|| (id, dist))
        };
        let radius_one_precomp = |&(id, dist): &(usize, f32)| {
            (dist < (weight * self.weight(id)).powi(2) as f32).then(|| (id, dist))
        };
        let pos = |&(id, _): &(usize, f32)| {
            (self.position(id).distance_squared(pos) as f64)
                < (weight * self.weight(id)).powi(2) * remaining_radius
        };

        if !self.cache_empty {
            guard.retain(pos);
            let collected: Vec<_> = guard.iter().filter_map(|x| radius_one(x)).collect();
            return collected.into_iter();
        } else {
            assert!(guard.is_empty());
            guard.extend(
                self.structure
                    .nearest_neighbors_with_distances(index, self.over_query_radius, results)
                    .filter(filter),
            );
            let collected: Vec<_> = guard.iter().filter_map(|x| radius_one_precomp(x)).collect();
            collected.into_iter()
        }
    }
}
impl<'a, const D: usize, ID: Embedder<'a, D>> SpatialIndex<D> for DynamicQuery<'a, D, ID> {
    fn name(&self) -> String {
        String::from("dynamic queries")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("dynamic_queries.rs")
    }
}

impl<'a, const D: usize, ID: Embedder<'a, D>> query::Embedder<'a, D> for DynamicQuery<'a, D, ID> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        let mut query = DynamicQuery {
            query_cache: empty_cache(embedding.positions.len()),
            structure: ID::new(embedding),
            positions: vec![],
            query_buffer: 0.,
            over_query_radius: 1.1,
            overquery: false,
            cache_empty: true,
            _phantom: std::marker::PhantomData,
        };
        query.update_positions(&embedding.positions, None);
        // assert_ne!(query.query_cache.len(), 0);
        query
    }

    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        if !self.cache_empty || self.overquery {
            self.nearest_neighbors(index, 1., result);
        } else {
            // TODO: find out why this fails
            // assert!(self.query_cache[index].lock().unwrap().is_empty());

            self.nearest_neighbors(index, 1., result);
            let pos = self.position(index);
            let weight = self.weight(index);

            result.retain(|&x| {
                index != x
                    && !self.is_connected(index, x)
                    // TODO: remove dedup from embedder
                    && (weight > self.weight(x) || weight == self.weight(x) && index > x)
                    && (self.position(x).distance_squared(pos) as f64)
                        < (weight * self.weight(x)).powi(2)
            });
        }
    }
}
