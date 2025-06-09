use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

use nalgebra::{DMatrix, SMatrix};

#[derive(Clone)]
pub struct SNN<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    graph: &'a crate::graph::Graph,
    projected: Vec<(usize, DVec<D>)>,
    pub v: SMatrix<f32, D, D>,
    pub max_weights: Vec<f64>,
}

impl<'a, const D: usize> SNN<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut snn = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            projected: Vec::new(),
            v: SMatrix::<f32, D, D>::identity(),
            max_weights: Vec::new(),
        };
        snn.update_positions(&embedding.positions);
        snn
    }

    pub fn get_query_sequence(&self) -> Vec<usize> {
        // Get the sequence of indices sorted by their first dimension in the projected space
        self.projected.iter().map(|(i, _)| *i).collect()
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
        // self.v = SMatrix::<f64, D, D>::identity();
        // self.positions = positions.to_vec();

        let x = DMatrix::<f32>::from_fn(positions.len(), D, |r, c| positions[r].components[c]);

        // Print the variance of the original positions
        // let var_before = x.row_variance();
        // println!("Variance BEFORE PCA : {:?}", var_before);

        // // ---------------------------------------------------------------- center

        let mean = x.row_mean();
        let mean_mat = DMatrix::<f32>::from_fn(positions.len(), D, |_, c| mean[c]);
        let centred = x.clone() - mean_mat;

        // // ---------------------------------------------------------------- normalise

        // Normalize each row (position) to unit length
        // for r in 0..centred.nrows() {
        //     let mut row = centred.row_mut(r);
        //     let norm = row.norm();
        //     if norm > f32::EPSILON {
        //         row.scale_mut(1.0 / norm);
        //     }
        // }

        // // ---------------------------------------------------------------- SVD
        let svd = nalgebra::linalg::SVD::new(centred, false, true);
        let v_t = svd
            .v_t
            .expect("requested V^T in SVD constructor but got None");

        // let u = svd.u.expect("requested U in SVD constructor but got None");

        // let scaling

        // // let singulars = svd.singular_values; // already sorted ↓
        // // println!("Singular values     : {:?}", singulars);

        // // ---------------------------------------------------------------- project

        // self.v = v_t.transpose(); // Store the transpose of V for later use

        // // Transpose and save to self.v
        for i in 0..D {
            for j in 0..D {
                self.v[(i, j)] = v_t[(j, i)];
            }
        }

        // self.v = SMatrix::<f32, D, D>::identity();

        let projected = &x * self.v;

        self.projected = projected
            .row_iter()
            .enumerate()
            .map(|(i, row)| {
                let mut arr = [0.0f32; D];
                for (j, v) in row.iter().enumerate() {
                    arr[j] = *v;
                }
                (i, DVec::from(arr))
            })
            .collect();

        // sort the indices based on the first dimension of the positions
        self.projected.sort_by(|&a, &b| {
            a.1[0]
                .partial_cmp(&b.1[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // print variance of indices first dimension
        // let var_after = projected.row_variance();
        // println!("Variance AFTER PCA : {:?}", var_after);
    }
}

impl<'a, const D: usize> Query for SNN<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let query_radius = radius * self.weight(index).powi(2);

        let mut result = Vec::new();

        // project the position of the index
        let pos_vec =
            nalgebra::SVector::<f32, D>::from_row_slice(&self.positions[index].components);
        let projected_position_vec = pos_vec.transpose() * self.v;
        let projected_position_array: [f32; D] = projected_position_vec.transpose().into();
        let projected_position = DVec::from(projected_position_array);

        let first_dim_lower = projected_position[0] - query_radius as f32;
        let first_dim_upper = projected_position[0] + query_radius as f32;

        // find the range of indices in the first dimension using binary search
        let start = self
            .projected
            .binary_search_by(|probe| {
                probe.1[0]
                    .partial_cmp(&first_dim_lower)
                    .unwrap_or(std::cmp::Ordering::Greater)
            })
            .unwrap_or_else(|x| x);
        let end = self
            .projected
            .binary_search_by(|probe| {
                probe.1[0]
                    .partial_cmp(&first_dim_upper)
                    .unwrap_or(std::cmp::Ordering::Less)
            })
            .unwrap_or_else(|x| x);

        let query_radius = query_radius.powi(2) as f32;
        for (i, pos) in &self.projected[start..end] {
            if i != &index && pos.distance_squared(&projected_position) <= query_radius {
                result.push(*i);
            }
        }

        result
    }
}
impl<'a, const D: usize> SpatialIndex<D> for SNN<'a, D> {
    fn name(&self) -> String {
        "SNN".to_string()
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("snn.rs")
    }
}

impl<'a, const D: usize> query::Embedder<D> for SNN<'a, D> {}
