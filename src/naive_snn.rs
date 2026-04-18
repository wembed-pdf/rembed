use crate::{
    Embedding, NodeId,
    dvec::DVec,
    query::{self, SpatialIndex, Update},
};
use sprk::svd::Svd;

#[derive(Clone)]
pub struct NaiveSnn<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    /// Positions sorted by principal-axis projection
    sorted_positions: Vec<DVec<D>>,
    /// Original IDs in sorted order
    sorted_ids: Vec<u32>,
    /// Per-point principal-axis projection (sorted)
    sort_vals: Vec<f32>,
    /// SVD for extracting principal axis
    svd: Svd<D, f32>,
    /// Principal axis (first singular vector)
    principal_axis: [f32; D],
    /// Mean of all positions
    mean: [f32; D],
}

impl<'a, const D: usize> NaiveSnn<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut snn = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            sorted_positions: Vec::new(),
            sorted_ids: Vec::new(),
            sort_vals: Vec::new(),
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

        self.sort_vals = projections.iter().map(|&(v, _)| v).collect();
        self.sorted_ids = projections.iter().map(|&(_, id)| id as u32).collect();
        self.sorted_positions = projections
            .iter()
            .map(|&(_, id)| self.positions[id])
            .collect();
    }
}

impl<const D: usize> crate::query::Graph for NaiveSnn<'_, D> {
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

impl<const D: usize> query::Position<D> for NaiveSnn<'_, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<const D: usize> query::Update<D> for NaiveSnn<'_, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();
        self.build_index();
    }
}

impl<const D: usize> crate::Query<D> for NaiveSnn<'_, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        if self.sort_vals.is_empty() {
            return;
        }

        let radius_f32 = radius as f32;
        let radius_sq = radius_f32 * radius_f32;

        // Project query onto principal axis
        let sv_q: f32 = (0..D)
            .map(|j| (pos.components[j] - self.mean[j]) * self.principal_axis[j])
            .sum();

        // Binary search for candidate range
        let left = self.sort_vals.partition_point(|&v| v < sv_q - radius_f32);
        let right = self.sort_vals.partition_point(|&v| v <= sv_q + radius_f32);

        // Scan candidates with scalar distance
        for i in left..right {
            if pos.distance_squared(&self.sorted_positions[i]) <= radius_sq {
                results.push(self.sorted_ids[i] as usize);
            }
        }
    }
}

impl<const D: usize> SpatialIndex<D> for NaiveSnn<'_, D> {
    fn name(&self) -> String {
        String::from("naive_snn")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("naive_snn.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for NaiveSnn<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}
