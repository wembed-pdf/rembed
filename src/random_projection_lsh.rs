use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct RandomProjectionLsh<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,

    // LSH-specific fields
    hash_tables: Vec<FxHashMap<u64, Vec<NodeId>>>,
    random_hyperplanes: Vec<Vec<DVec<D>>>,
    num_tables: usize,
    num_projections: usize,
}

impl<'a, const D: usize> RandomProjectionLsh<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        Self::new_with_params(embedding, None, None)
    }

    pub fn new_with_params(
        embedding: Embedding<'a, D>,
        num_tables: Option<usize>,
        num_projections: Option<usize>,
    ) -> Self {
        let num_tables = num_tables.unwrap_or_else(|| Self::default_num_tables(D));
        let num_projections = num_projections.unwrap_or_else(|| Self::default_num_projections(D));

        let mut lsh = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            hash_tables: vec![FxHashMap::default(); num_tables],
            random_hyperplanes: Vec::new(),
            num_tables,
            num_projections,
        };

        lsh.update_positions(&embedding.positions, None);
        lsh
    }

    fn default_num_tables(d: usize) -> usize {
        match d {
            0..=4 => 5,
            5..=16 => 1,
            17..=64 => 15,
            _ => 20,
        }
    }

    fn default_num_projections(d: usize) -> usize {
        match d {
            0..=4 => 8,
            5..=16 => 48,
            17..=64 => 24,
            _ => 20,
        }
    }

    fn generate_hyperplanes(&self) -> Vec<Vec<DVec<D>>> {
        use rand::Rng;
        let mut rng = rand::rng();

        (0..self.num_tables)
            .map(|_| {
                (0..self.num_projections)
                    .map(|_| {
                        // Generate random vector with Gaussian-like distribution
                        let components = DVec::from_fn(|_| rng.random_range(-1.0..1.0_f32));
                        // Normalize to unit vector
                        let mag = components.magnitude();
                        if mag > 0.0 {
                            components / mag
                        } else {
                            DVec::unit(0)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn compute_hash(&self, position: &DVec<D>, table_idx: usize) -> u64 {
        let hyperplanes = &self.random_hyperplanes[table_idx];
        let mut hash: u64 = 0;

        for (bit_idx, hyperplane) in hyperplanes.iter().enumerate() {
            if bit_idx >= 64 {
                break; // u64 has only 64 bits
            }
            // If dot product is positive, set bit to 1
            if position.dot(hyperplane) >= 0.0 {
                hash |= 1u64 << bit_idx;
            }
        }

        hash
    }
}

impl<'a, const D: usize> Graph for RandomProjectionLsh<'a, D> {
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

impl<'a, const D: usize> Position<D> for RandomProjectionLsh<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for RandomProjectionLsh<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();

        // Generate new random hyperplanes
        self.random_hyperplanes = self.generate_hyperplanes();

        // Clear and rebuild all hash tables
        self.hash_tables = vec![FxHashMap::default(); self.num_tables];

        // Hash all points into all L tables
        for (node_id, position) in self.positions.iter().enumerate() {
            for table_idx in 0..self.num_tables {
                let hash_code = self.compute_hash(position, table_idx);

                self.hash_tables[table_idx]
                    .entry(hash_code)
                    .or_default()
                    .push(node_id);
            }
        }
    }
}

impl<'a, const D: usize> Query<D> for RandomProjectionLsh<'a, D> {
    fn query_radius(&self, pos: DVec<D>, _radius: f64, results: &mut Vec<NodeId>) {
        // Special case for single table - no deduplication needed
        if self.num_tables == 1 {
            let hash_code = self.compute_hash(&pos, 0);

            if let Some(bucket) = self.hash_tables[0].get(&hash_code) {
                results.extend(bucket.iter().copied());
            }
            return;
        }

        // Multiple tables - use FxHashSet to avoid duplicate candidates
        let mut candidates = FxHashSet::default();

        // Query all L hash tables
        for table_idx in 0..self.num_tables {
            let hash_code = self.compute_hash(&pos, table_idx);

            // Get candidates from exact hash match
            if let Some(bucket) = self.hash_tables[table_idx].get(&hash_code) {
                candidates.extend(bucket.iter().copied());
            }
        }

        results.extend(candidates);
    }
}

impl<'a, const D: usize> SpatialIndex<D> for RandomProjectionLsh<'a, D> {
    fn name(&self) -> String {
        format!("rp-lsh-L{}-K{}", self.num_tables, self.num_projections)
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("random_projection_lsh.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for RandomProjectionLsh<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
