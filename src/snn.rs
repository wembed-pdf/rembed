use crate::{
    Embedding, NodeId,
    dvec::DVec,
    query::{self, SpatialIndex, Update},
};
use sprk::simd::PDVec;
use sprk::svd::Svd;

const W: usize = 8;

#[derive(Clone)]
pub struct Snn<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    /// PDVecs sorted by principal-axis projection, storing original positions and IDs
    pdvecs: Vec<PDVec<D, W, f32, u32>>,
    /// Min principal-axis projection per PDVec group
    group_min: Vec<f32>,
    /// SVD for extracting principal axis
    svd: Svd<D, f32>,
    /// Principal axis (first singular vector)
    principal_axis: [f32; D],
    /// Mean of all positions
    mean: [f32; D],
}

impl<'a, const D: usize> Snn<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut snn = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            pdvecs: Vec::new(),
            group_min: Vec::new(),
            svd: Svd::new(),
            principal_axis: [0.0; D],
            mean: [0.0; D],
        };
        snn.update_positions(&embedding.positions, None);
        snn
    }

    fn build_index(&mut self) {
        let n = self.positions.len();
        if n == 0 {
            return;
        }

        let raw_positions: Vec<[f32; D]> = self.positions.iter().map(|p| p.components).collect();

        // Compute mean
        self.mean = [0.0; D];
        let inv_n = 1.0 / n as f32;
        for pos in &raw_positions {
            for j in 0..D {
                self.mean[j] += pos[j];
            }
        }
        for m in &mut self.mean {
            *m *= inv_n;
        }

        // Compute SVD to find principal axis
        self.svd.compute_svd(&raw_positions);

        // Extract principal axis by probing SVD projection
        self.principal_axis = {
            let origin_proj = self.svd.project(&self.mean);
            let mut axis = [0.0f32; D];
            for j in 0..D {
                let mut shifted = self.mean;
                shifted[j] += 1.0;
                let proj = self.svd.project(&shifted);
                axis[j] = proj[0] - origin_proj[0];
            }
            let norm: f32 = axis.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for a in &mut axis {
                    *a /= norm;
                }
            }
            axis
        };

        // Project all points onto principal axis
        let mut projections: Vec<(f32, usize)> = raw_positions
            .iter()
            .enumerate()
            .map(|(i, pos)| {
                let proj: f32 = (0..D)
                    .map(|j| (pos[j] - self.mean[j]) * self.principal_axis[j])
                    .sum();
                (proj, i)
            })
            .collect();

        projections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Build PDVecs grouped by W, with per-group min projection
        let num_groups = n.div_ceil(W);
        self.pdvecs = Vec::with_capacity(num_groups);
        self.group_min = Vec::with_capacity(num_groups);

        for chunk_start in (0..n).step_by(W) {
            let chunk_end = (chunk_start + W).min(n);

            let pdvec = PDVec::<D, W, f32, u32>::new((chunk_start..chunk_end).map(|si| {
                let orig_id = projections[si].1;
                (raw_positions[orig_id], orig_id)
            }));
            self.pdvecs.push(pdvec);
            self.group_min.push(projections[chunk_start].0);
        }
    }
}

impl<const D: usize> crate::query::Graph for Snn<'_, D> {
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

impl<const D: usize> query::Position<D> for Snn<'_, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<const D: usize> query::Update<D> for Snn<'_, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();
        self.build_index();
    }
}

impl<const D: usize> crate::Query<D> for Snn<'_, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        if self.pdvecs.is_empty() {
            return;
        }

        let radius_f32 = radius as f32;
        let radius_sq_half = radius_f32 * radius_f32 * 0.5 + 1e-2;

        // Project query onto principal axis
        let sv_q: f32 = (0..D)
            .map(|j| (pos.components[j] - self.mean[j]) * self.principal_axis[j])
            .sum();

        // Binary search on group_min:
        // - Start one group before the first whose min > sv_q - radius
        //   (that prior group could still contain points within range)
        // - End at the first group whose min > sv_q + radius
        let left = self
            .group_min
            .partition_point(|&min_p| min_p <= sv_q - radius_f32)
            .saturating_sub(1);
        let right = self
            .group_min
            .partition_point(|&min_p| min_p <= sv_q + radius_f32)
            .min(self.pdvecs.len());

        // Precompute ||q||²/2 for dist_half_squared
        let q_squared_half: f32 = pos.components.iter().map(|&x| x * x).sum::<f32>() * 0.5;

        for pdvec in &self.pdvecs[left..right] {
            let distances = pdvec.dist_half_squared(pos.components, q_squared_half);
            let (count, ids, _) = pdvec.compress(distances, radius_sq_half);
            for i in 0..count {
                results.push(ids[i] as usize);
            }
        }
    }
}

impl<const D: usize> SpatialIndex<D> for Snn<'_, D> {
    fn name(&self) -> String {
        String::from("snn")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("snn.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Snn<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}
