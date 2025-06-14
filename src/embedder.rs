use crate::{
    NodeId,
    dvec::DVec,
    graph::Graph,
    query::{Embedder, Graph as GraphTrait, Query},
};
use rand::{Rng, rngs::SmallRng};
use rayon::prelude::*;

/// Configuration options for the embedder
#[derive(Clone, Debug)]
pub struct EmbedderOptions {
    pub learning_rate: f64,
    pub cooling_factor: f64,
    pub max_iterations: usize,
    pub min_position_change: f64,
    pub attraction_scale: f64,
    pub repulsion_scale: f64,
}

impl Default for EmbedderOptions {
    fn default() -> Self {
        Self {
            learning_rate: 10.0,
            cooling_factor: 0.99,
            max_iterations: 1000,
            min_position_change: 1e-8,
            attraction_scale: 1.0,
            repulsion_scale: 1.0,
        }
    }
}

/// Adam optimizer for gradient descent
pub struct AdamOptimizer<const D: usize> {
    m: Vec<DVec<D>>, // First moment estimates
    v: Vec<DVec<D>>, // Second moment estimates
    t: usize,        // Time step

    learning_rate: f64,
    cooling_factor: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl<const D: usize> AdamOptimizer<D> {
    pub fn new(num_nodes: usize, learning_rate: f64, cooling_factor: f64) -> Self {
        Self {
            m: vec![DVec::zero(); num_nodes],
            v: vec![DVec::zero(); num_nodes],
            t: 0,
            learning_rate,
            cooling_factor,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    pub fn update(&mut self, positions: &mut [DVec<D>], forces: &[DVec<D>]) {
        self.t += 1;
        let cooling = self.cooling_factor.powi(self.t as i32) as f32;

        for i in 0..positions.len() {
            // Update biased first moment estimate
            self.m[i] = self.m[i] * (self.beta1 as f32) + forces[i] * ((1.0 - self.beta1) as f32);

            // Update biased second moment estimate
            let force_squared = forces[i].map(|x| x * x);
            self.v[i] =
                self.v[i] * (self.beta2 as f32) + force_squared * ((1.0 - self.beta2) as f32);

            // Compute bias-corrected moments
            let m_hat = self.m[i] / ((1.0 - self.beta1.powi(self.t as i32)) as f32);
            let v_hat = self.v[i] / ((1.0 - self.beta2.powi(self.t as i32)) as f32);

            // Update parameters
            let update = m_hat * (cooling * self.learning_rate as f32)
                / v_hat.map(|v| (v.sqrt() + self.epsilon as f32));
            positions[i] += update;
        }
    }

    pub fn reset(&mut self) {
        self.t = 0;
        for i in 0..self.m.len() {
            self.m[i] = DVec::zero();
            self.v[i] = DVec::zero();
        }
    }
}

/// Main weighted embedder with generic spatial index
pub struct WEmbedder<SI: Query, const D: usize> {
    // Node data
    positions: Vec<DVec<D>>,
    weights: Vec<f64>,
    forces: Vec<DVec<D>>,
    old_positions: Vec<DVec<D>>,
    positions_log: Vec<(u64, Vec<DVec<D>>)>,

    // Spatial index
    spatial_index: SI,

    // Optimizer
    optimizer: AdamOptimizer<D>,

    // Configuration
    options: EmbedderOptions,
    iteration: usize,
}

impl<'a, SI: Embedder<'a, D> + Clone + Sync, const D: usize> WEmbedder<SI, D> {
    pub fn random(seed: u64, graph: &'a Graph, options: EmbedderOptions) -> Self {
        let mut spatial_index = SI::from_graph(graph);
        let n = graph.nodes.len();

        // Initialize random positions
        let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(seed);
        let cube_side = (n as f64).powf(1.0 / D as f64);
        let positions: Vec<DVec<D>> = (0..n)
            .map(|_| {
                let components: [f32; D] =
                    std::array::from_fn(|_| rng.random_range(0.0..cube_side) as f32);
                DVec::new(components)
            })
            .collect();
        spatial_index.update_positions(&positions);

        Self::new(spatial_index, options)
    }
    pub fn new(spatial_index: SI, options: EmbedderOptions) -> Self {
        let n = spatial_index.num_nodes();
        let learning_rate = options.learning_rate;
        let cooling_factor = options.cooling_factor;

        let positions: Vec<_> = (0..n).map(|node| *spatial_index.position(node)).collect();

        // Extract weights from graph
        let weights: Vec<f64> = (0..n).map(|node| spatial_index.weight(node)).collect();

        Self {
            positions,
            weights,
            forces: vec![DVec::zero(); n],
            old_positions: vec![DVec::zero(); n],
            positions_log: Vec::new(),
            spatial_index,
            optimizer: AdamOptimizer::new(n, learning_rate, cooling_factor),
            options,
            iteration: 0,
        }
    }

    /// Run the embedding algorithm until convergence or max iterations
    pub fn embed(&mut self) -> Vec<DVec<D>> {
        self.optimizer.reset();

        loop {
            println!("Iteration {}", self.iteration);
            self.iteration += 1;

            self.calculate_step();

            // Check convergence
            if self.check_convergence() || self.iteration >= self.options.max_iterations {
                break;
            }
        }

        self.positions.clone()
    }

    pub fn calculate_step(&mut self) {
        // Save old positions
        self.old_positions.clone_from(&self.positions);
        if self.iteration % 10 == 0 {
            self.positions_log
                .push((self.iteration as u64, self.old_positions.clone()));
        }

        // Clear forces
        self.forces.iter_mut().for_each(|f| *f = DVec::zero());

        // Update spatial index
        self.update_spatial_index();

        // Calculate forces
        self.calculate_attraction_forces();
        self.calculate_repulsion_forces();

        // Update positions
        self.optimizer.update(&mut self.positions, &self.forces);
    }

    fn update_spatial_index(&mut self) {
        self.spatial_index.update_positions(&self.positions);
    }

    fn calculate_attraction_forces(&mut self) {
        // Calculate forces for each node in parallel using neighbor lists
        let forces: Vec<DVec<D>> = (0..self.positions.len())
            .into_par_iter()
            .map(|v| {
                let mut force = DVec::zero();

                // Get neighbors from graph
                let neighbors = GraphTrait::neighbors(&self.spatial_index, v);

                // Calculate attraction force for each neighbor
                for &u in neighbors {
                    let f = self.attraction_force(v, u);
                    force += f;
                }

                force
            })
            .collect();

        // Update the forces
        self.forces = forces;
    }

    fn attraction_force(&self, u: NodeId, v: NodeId) -> DVec<D> {
        let pos_u = self.positions[u];
        let pos_v = self.positions[v];

        let direction = pos_v - pos_u;
        let distance = direction.magnitude();

        if distance == 0.0 {
            // Random displacement if positions are identical
            let mut rng = rand::rng();
            return DVec::from_fn(|_| rng.random_range(-0.01..0.01));
        }

        let weight_factor = self.weights[u] * self.weights[v];
        let weighted_distance = distance as f64 / weight_factor;

        if weighted_distance <= 1.0 {
            // Already close enough
            DVec::zero()
        } else {
            // Attraction force
            direction * (self.options.attraction_scale / (distance as f64 * weight_factor)) as f32
        }
    }

    fn calculate_repulsion_forces(&mut self) {
        // Stage 1: Query nearest neighbors for all nodes in parallel
        let mut repelling_candidates: Vec<Vec<NodeId>> = (0..self.positions.len())
            .into_par_iter()
            .map(|v| {
                // Find nearby nodes that might repel
                let mut candidates = self.spatial_index.repelling_nodes(v);

                candidates.sort_unstable();
                candidates.dedup();

                candidates
            })
            .collect();

        for (i, candidates) in repelling_candidates.clone().iter().enumerate() {
            for candidate in candidates {
                repelling_candidates[*candidate].push(i);
            }
        }

        repelling_candidates.par_iter_mut().for_each(|candidates| {
            candidates.sort_unstable();
            candidates.dedup();
        });

        // Stage 2: Calculate repulsion forces in parallel
        let new_forces: Vec<DVec<D>> = repelling_candidates
            .par_iter()
            .enumerate()
            .map(|(v, candidates)| {
                let mut force = self.forces[v]; // Start with existing attraction force

                // Add repulsion forces from all candidates
                for &u in candidates {
                    let f = self.repulsion_force(v, u);
                    force += f;
                }

                force
            })
            .collect();

        // Update the forces
        self.forces = new_forces;
    }

    fn repulsion_force(&self, v: NodeId, u: NodeId) -> DVec<D> {
        let pos_v = self.positions[v];
        let pos_u = self.positions[u];

        let direction = pos_v - pos_u;
        let distance = direction.magnitude();

        if distance == 0.0 {
            // Random displacement if positions are identical
            let mut rng = rand::rng();
            return DVec::from_fn(|_| rng.random_range(-0.01..0.01));
        }

        let weight_factor = self.weights[v] * self.weights[u];
        let weighted_distance = distance as f64 / weight_factor;

        if weighted_distance > 1.0 {
            // Far enough apart
            DVec::zero()
        } else {
            // Repulsion force
            direction * (self.options.repulsion_scale / (distance as f64 * weight_factor)) as f32
        }
    }

    fn check_convergence(&self) -> bool {
        let (sum_norm_squared, sum_diff_squared) = self
            .positions
            .iter()
            .zip(&self.old_positions)
            .map(|(new_pos, old_pos)| {
                let diff = *new_pos - *old_pos;
                (old_pos.magnitude_squared(), diff.magnitude_squared())
            })
            .fold((0.0, 0.0), |(sum_norm, sum_diff), (norm, diff)| {
                (sum_norm + norm as f64, sum_diff + diff as f64)
            });

        if sum_norm_squared == 0.0 {
            return false;
        }

        let relative_change = sum_diff_squared / sum_norm_squared;
        relative_change < self.options.min_position_change
    }

    /// Get the current positions
    pub fn positions(&self) -> &[DVec<D>] {
        &self.positions
    }

    /// Get the history of positions
    pub fn history(&self) -> &[(u64, Vec<DVec<D>>)] {
        &self.positions_log
    }

    /// Get the current iteration count
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}
