use std::sync::Mutex;

use crate::{
    NodeId, Query,
    dvec::DVec,
    query::{self, Embedder, Graph, Position, SpatialIndex, Update},
};

pub struct DynamicQuery<'a, const D: usize, ID: Embedder<'a, D>> {
    query_cache: Vec<Mutex<Vec<usize>>>,
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
            query_buffer: self.query_buffer.clone(),
            over_query_radius: self.over_query_radius.clone(),
            cache_empty: true,
            overquery: self.overquery,
            _phantom: std::marker::PhantomData,
        }
    }
}

fn empty_cache(len: usize) -> Vec<Mutex<Vec<usize>>> {
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
        let max_deviation = last_delta.unwrap_or(10.) * 2.;

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
            self.structure.update_positions(positions, last_delta);
            return;
        }
        // return;
        if self.query_buffer - max_deviation < 1. {
            self.structure.update_positions(positions, last_delta);
            // println!("recomputing after pos diff {}", max_deviation);
            self.cache_empty = true;
            for cache in self.query_cache.iter_mut() {
                let mut guard = cache.lock().unwrap();
                guard.clear();
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

impl<'a, const D: usize, ID: Embedder<'a, D>> Query for DynamicQuery<'a, D, ID> {
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

        let pos = self.position(index);
        let weight = self.weight(index);
        let remaining_radius = self.query_buffer;
        let filter = |&id: &usize| {
            !self.structure.is_connected(index, id)
                && (self.position(id).distance_squared(&pos) as f64)
                    < (weight * self.weight(id)).powi(2) * remaining_radius
        };
        let radius_one = |&id: &usize| {
            (self.position(id).distance_squared(&pos) as f64) < (weight * self.weight(id)).powi(2)
        };
        let pos = |&id: &usize| {
            (self.position(id).distance_squared(&pos) as f64)
                < (weight * self.weight(id)).powi(2) * remaining_radius
        };

        if !self.cache_empty {
            guard.retain(pos);
            results.extend(guard.iter().filter(|x| radius_one(x)).cloned());
        } else {
            self.structure
                .nearest_neighbors(index, self.over_query_radius as f64, &mut guard);
            guard.retain(filter);
            results.extend(guard.iter().filter(|x| radius_one(x)).cloned());
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
            over_query_radius: 1.2,
            overquery: false,
            cache_empty: true,
            _phantom: std::marker::PhantomData,
        };
        query.update_positions(&embedding.positions, None);
        // assert_ne!(query.query_cache.len(), 0);
        query
    }

    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        self.nearest_neighbors(index, 1., result);
    }
}

/// **DANGER:** This abstraction invokes UB and is exposed for the insane fun
///             of it only. Do not ever try this in a production program.
///
/// On x86_64, it momentarily disables subnormal math using forbidden MXCSR
/// operations until the current scope is exited. On other CPUs it does nothing.
struct DenormalsGuard {
    old_mxcsr: u32,
}
//
impl DenormalsGuard {
    /// Set up the forbidden denormals-disabling magic
    fn new() -> Self {
        #[expect(
            deprecated,
            reason = "Deprecated due to UB, which is exactly what our foolish selves are looking for here"
        )]
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64;
            let old_mxcsr = x86_64::_mm_getcsr();
            // Set FTZ flag (0x8000) and DAZ flag (0x0040)
            x86_64::_mm_setcsr(old_mxcsr | 0x8040);
            Self { old_mxcsr }
        }
        #[cfg(not(target_arch = "x86_64"))]
        Self { old_mxcsr: 0 }
    }
}
//
impl Drop for DenormalsGuard {
    fn drop(&mut self) {
        #[expect(
            deprecated,
            reason = "Deprecated due to UB, which is exactly what our foolish selves are looking for here"
        )]
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.old_mxcsr);
        }
    }
}
