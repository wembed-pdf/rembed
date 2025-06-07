use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, Update},
};

use nalgebra::{DMatrix, DVector, SMatrix, SquareMatrix};
use std::{
    collections::{HashMap, HashSet},
    ops::Mul,
};

fn compute_weight_class(graph: &impl Graph, index: usize) -> usize {
    let weight = graph.neighbors(index).len();
    let mut i = 0;
    while (1 << i) <= weight {
        i += 1;
    }
    i
}

#[derive(Clone)]
pub struct SNN<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    graph: &'a crate::graph::Graph,
    first_dim: Vec<usize>,
    pub v: SMatrix<f64, D, D>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> SNN<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            first_dim: Vec::new(),
            v: SMatrix::<f64, D, D>::identity(),
            max_weights: Vec::new(),
        }
    }
}

impl<'a, const D: usize> Graph for SNN<'a, D> {
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

impl<'a, const D: usize> Position<D> for SNN<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
}

impl<'a, const D: usize> Update<D> for SNN<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.v = SMatrix::<f64, D, D>::identity();
        self.positions = positions.to_vec();

        self.first_dim.clear();

        // create a vector of the indices of the positions
        let mut indices: Vec<usize> = (0..self.positions.len()).collect();
        // sort the indices based on the first dimension of the positions
        indices.sort_by(|&a, &b| {
            self.positions[a][0]
                .partial_cmp(&self.positions[b][0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.first_dim = indices;

        //TODO
        // let x = DMatrix::<f32>::from_fn(positions.len(), D, |r, c| positions[r].components[c]);

        // // Print the variance of the original positions
        // let var_before = x.row_variance();
        // println!("Variance BEFORE PCA : {:?}", var_before);
    }
}

impl<'a, const D: usize> Query for SNN<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let query_radius = radius * self.weight(index).powi(2);

        let mut result = Vec::new();

        let first_dim_lower = self.position(index)[0] - query_radius as f32;
        let first_dim_upper = self.position(index)[0] + query_radius as f32;

        // find the range of indices in the first dimension using binary search
        let start = self
            .first_dim
            .binary_search_by(|probe| {
                self.positions[*probe][0]
                    .partial_cmp(&first_dim_lower)
                    .unwrap_or(std::cmp::Ordering::Greater)
            })
            .unwrap_or_else(|x| x);
        let end = self
            .first_dim
            .binary_search_by(|probe| {
                self.positions[*probe][0]
                    .partial_cmp(&first_dim_upper)
                    .unwrap_or(std::cmp::Ordering::Less)
            })
            .unwrap_or_else(|x| x);

        for i in start..end {
            let pos = &self.positions[self.first_dim[i]];
            if pos.distance_squared(&self.positions[index]) < radius as f32
                && self.first_dim[i] != index
            {
                result.push(self.first_dim[i]);
            } else if self.weight(index) > 1.0 {
                // If the node has a weight greater than 1, we need to check the distance
                let weight = self.weight(self.first_dim[i]);
                let distance = pos.distance_squared(&self.positions[index]);
                if distance < (weight * self.weight(index)).powi(2) as f32
                    && self.first_dim[i] != index
                {
                    result.push(self.first_dim[i]);
                }
            }
        }
        result
    }

    fn name(&self) -> String {
        "Weighted R-Tree".to_string()
    }
}

impl<'a, const D: usize> query::Embedder<D> for SNN<'a, D> {}
