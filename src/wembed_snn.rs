use std::marker::PhantomData;
use std::ptr;

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
use wembed_snn::*;

pub struct WembedSnnWrapper<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    index: *mut WembedSnnIndex,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, const D: usize> Clone for WembedSnnWrapper<'a, D> {
    fn clone(&self) -> Self {
        Self::new(&Embedding {
            positions: self.positions.clone(),
            graph: self.graph,
        })
    }
}

impl<'a, const D: usize> WembedSnnWrapper<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut wrapper = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            index: ptr::null_mut(),
            _phantom: PhantomData,
        };
        wrapper.update_positions(&embedding.positions, None);
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
            unsafe { wembed_snn_point_count(self.index) }
        }
    }

    /// Get the dimensionality of the index
    pub fn dimensions(&self) -> usize {
        if self.index.is_null() {
            D
        } else {
            unsafe { wembed_snn_dimensions(self.index) }
        }
    }
}

impl<'a, const D: usize> Drop for WembedSnnWrapper<'a, D> {
    fn drop(&mut self) {
        if !self.index.is_null() {
            unsafe {
                wembed_snn_destroy_index(self.index);
            }
            self.index = ptr::null_mut();
        }
    }
}

// Implement Send and Sync - SNN queries are thread-safe for read-only operations
unsafe impl<'a, const D: usize> Send for WembedSnnWrapper<'a, D> {}
unsafe impl<'a, const D: usize> Sync for WembedSnnWrapper<'a, D> {}

impl<'a, const D: usize> Graph for WembedSnnWrapper<'a, D> {
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

impl<'a, const D: usize> Position<D> for WembedSnnWrapper<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for WembedSnnWrapper<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();

        // Destroy old index if it exists
        if !self.index.is_null() {
            unsafe {
                wembed_snn_destroy_index(self.index);
            }
            self.index = ptr::null_mut();
        }

        // Create new index if we have positions
        if !positions.is_empty() {
            let flat_points = self.positions_to_flat_array();
            unsafe {
                self.index = wembed_snn_create_index(flat_points.as_ptr(), positions.len(), D);
            }

            if self.index.is_null() {
                eprintln!("Warning: Failed to create wembed SNN index");
            }
        }
    }
}

impl<'a, const D: usize> Query<D> for WembedSnnWrapper<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        if !self.is_valid() {
            return;
        }

        let query_point: Vec<f32> = pos.components.iter().copied().collect();

        let mut search_result = unsafe {
            wembed_snn_radius_search(self.index, query_point.as_ptr(), radius as f32)
        };

        if !search_result.indices.is_null() && search_result.count > 0 {
            let indices =
                unsafe { std::slice::from_raw_parts(search_result.indices, search_result.count) };
            results.extend(indices.iter().copied());
            unsafe {
                wembed_snn_free_result(&mut search_result);
            }
        }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for WembedSnnWrapper<'a, D> {
    fn name(&self) -> String {
        "wembed_snn".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("wembed_snn.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for WembedSnnWrapper<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}
