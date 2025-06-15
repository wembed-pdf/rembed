use std::marker::PhantomData;
use std::ptr;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
use nanoflann::*;

pub struct NanoflannIndexWrapper<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    index: *mut NanoflannIndex,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, const D: usize> Clone for NanoflannIndexWrapper<'a, D> {
    fn clone(&self) -> Self {
        Self::new(&Embedding {
            positions: self.positions.clone(),
            graph: self.graph,
        })
    }
}

impl<'a, const D: usize> NanoflannIndexWrapper<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut wrapper = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            index: ptr::null_mut(),
            _phantom: PhantomData,
        };
        wrapper.update_positions(&embedding.positions);
        wrapper
    }

    fn positions_to_flat_array(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.positions.len() * D);
        for pos in &self.positions {
            for &component in &pos.components {
                flat.push(component);
            }
        }
        flat
    }

    /// Check if the index is valid
    pub fn is_valid(&self) -> bool {
        !self.index.is_null()
    }

    /// Get the number of points in the index
    pub fn point_count(&self) -> usize {
        if self.index.is_null() {
            0
        } else {
            unsafe { nanoflann_point_count(self.index) }
        }
    }

    /// Get the dimensionality of the index
    pub fn dimensions(&self) -> usize {
        if self.index.is_null() {
            D
        } else {
            unsafe { nanoflann_dimensions(self.index) }
        }
    }

    /// Get estimated memory usage
    pub fn memory_usage(&self) -> usize {
        if self.index.is_null() {
            std::mem::size_of::<Self>()
        } else {
            unsafe { nanoflann_memory_usage(self.index) }
        }
    }
}

impl<'a, const D: usize> Drop for NanoflannIndexWrapper<'a, D> {
    fn drop(&mut self) {
        if !self.index.is_null() {
            unsafe {
                nanoflann_destroy_index(self.index);
            }
            self.index = ptr::null_mut();
        }
    }
}

// Additional convenience methods specific to nanoflann
impl<'a, const D: usize> NanoflannIndexWrapper<'a, D> {
    /// Perform exact k-nearest neighbor search
    /// Returns vector of (index, squared_distance) pairs
    pub fn knn_search(&self, query_point: &DVec<D>, k: usize) -> Vec<(usize, f32)> {
        if !self.is_valid() || k == 0 {
            return Vec::new();
        }

        let query_flat: Vec<f32> = query_point.components.iter().copied().collect();
        let mut indices = vec![0usize; k];
        let mut distances_squared = vec![0f32; k];

        let num_found = unsafe {
            nanoflann_knn_search(
                self.index,
                query_flat.as_ptr(),
                k,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
            )
        };

        indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .take(num_found)
            .collect()
    }

    /// Perform radius search with raw results (no weight filtering)
    /// Returns vector of (index, squared_distance) pairs
    pub fn radius_search_raw(&self, query_point: &DVec<D>, radius: f32) -> Vec<(usize, f32)> {
        if !self.is_valid() {
            return Vec::new();
        }

        let query_flat: Vec<f32> = query_point.components.iter().copied().collect();
        let radius_squared = (radius * radius) + 10.;

        // Allocate buffers for results
        const MAX_RESULTS: usize = 100000;
        let mut indices = vec![0usize; MAX_RESULTS];
        let mut distances_squared = vec![0f32; MAX_RESULTS];

        let num_found = unsafe {
            nanoflann_radius_search(
                self.index,
                query_flat.as_ptr(),
                radius_squared,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
                MAX_RESULTS,
            )
        };

        indices
            .into_iter()
            .zip(distances_squared.into_iter())
            .take(num_found)
            .collect()
    }

    /// Create index with custom leaf size
    pub fn with_leaf_size(embedding: Embedding<'a, D>, leaf_max_size: usize) -> Self {
        let positions = embedding.positions.clone();
        let mut wrapper = Self {
            positions: positions.clone(),
            graph: embedding.graph,
            index: ptr::null_mut(),
            _phantom: PhantomData,
        };

        if !positions.is_empty() {
            let flat_points = wrapper.positions_to_flat_array();
            unsafe {
                wrapper.index =
                    nanoflann_create_index(flat_points.as_ptr(), positions.len(), D, leaf_max_size);
            }
        }

        wrapper
    }
}

// Implement Send and Sync if the underlying C++ library is thread-safe
// Note: This assumes nanoflann queries are thread-safe, which they typically are
// for read-only operations, but you should verify this for your use case
unsafe impl<'a, const D: usize> Send for NanoflannIndexWrapper<'a, D> {}
unsafe impl<'a, const D: usize> Sync for NanoflannIndexWrapper<'a, D> {}

impl<'a, const D: usize> Graph for NanoflannIndexWrapper<'a, D> {
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

impl<'a, const D: usize> Position<D> for NanoflannIndexWrapper<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for NanoflannIndexWrapper<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();

        // Destroy old index if it exists
        if !self.index.is_null() {
            unsafe {
                nanoflann_destroy_index(self.index);
            }
            self.index = ptr::null_mut();
        }

        // Create new index if we have positions
        if !positions.is_empty() {
            let flat_points = self.positions_to_flat_array();
            unsafe {
                self.index = nanoflann_create_index(
                    flat_points.as_ptr(),
                    positions.len(),
                    D,
                    10, // leaf_max_size
                );
            }

            if self.index.is_null() {
                eprintln!("Warning: Failed to create nanoflann index");
            }
        }
    }
}

impl<'a, const D: usize> Query for NanoflannIndexWrapper<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        if !self.is_valid() || index >= self.positions.len() {
            return Vec::new();
        }

        let own_position = self.positions[index];
        let own_weight = self.weight(index);
        let scaled_radius = radius * own_weight.powi(2);
        let scaled_radius_squared = (scaled_radius * scaled_radius) as f32;

        // Convert query point to flat array
        let query_point: Vec<f32> = own_position.components.iter().copied().collect();

        // Allocate buffers for results - estimate reasonable max size
        const MAX_RESULTS: usize = 1000;
        let mut indices = vec![0usize; MAX_RESULTS];
        let mut distances_squared = vec![0f32; MAX_RESULTS];

        let num_found = unsafe {
            nanoflann_radius_search(
                self.index,
                query_point.as_ptr(),
                scaled_radius_squared,
                indices.as_mut_ptr(),
                distances_squared.as_mut_ptr(),
                MAX_RESULTS,
            )
        };

        // Filter results and apply weight-based distance check
        let mut result = Vec::new();
        for i in 0..num_found {
            let neighbor_idx = indices[i];
            if neighbor_idx == index {
                continue; // Skip self
            }

            let other_pos = &self.positions[neighbor_idx];
            let other_weight = self.weight(neighbor_idx);

            // Apply the same distance check as other implementations
            if own_position.distance_squared(other_pos)
                <= (own_weight * other_weight).powi(2) as f32
            {
                result.push(neighbor_idx);
            }
        }

        result
    }
}

impl<'a, const D: usize> SpatialIndex<D> for NanoflannIndexWrapper<'a, D> {
    fn name(&self) -> String {
        "nanoflann".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("nanoflann.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for NanoflannIndexWrapper<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}
