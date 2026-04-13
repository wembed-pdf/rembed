use std::sync::Mutex;

use crate::{
    NodeId,
    dvec::Vector,
    dyn_embed::EmbedIndex,
    graph::Graph,
    query::Embedder,
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
    pub print_timings: bool,
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
            print_timings: false,
        }
    }
}

/// Adam optimizer for gradient descent, generic over vector type.
pub struct AdamOptimizer<V: Vector> {
    m: Vec<V>,    // First moment estimates
    v: Vec<V>,    // Second moment estimates
    t: usize,     // Time step
    dim: usize,

    learning_rate: f64,
    cooling_factor: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl<V: Vector> AdamOptimizer<V> {
    pub fn new(num_nodes: usize, dim: usize, learning_rate: f64, cooling_factor: f64) -> Self {
        Self {
            m: vec![V::zero(dim); num_nodes],
            v: vec![V::zero(dim); num_nodes],
            t: 0,
            dim,
            learning_rate,
            cooling_factor,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    pub fn update(&mut self, positions: &mut [V], forces: &[V]) {
        self.t += 1;
        let cooling = self.cooling_factor.powi(self.t as i32) as f32;

        for i in 0..positions.len() {
            // Update biased first moment estimate
            self.m[i] =
                self.m[i].clone() * (self.beta1 as f32) + forces[i].clone() * ((1.0 - self.beta1) as f32);

            // Update biased second moment estimate
            let force_squared = forces[i].map(|x| x * x);
            self.v[i] =
                self.v[i].clone() * (self.beta2 as f32) + force_squared * ((1.0 - self.beta2) as f32);

            // Compute bias-corrected moments
            let m_hat = self.m[i].clone() / ((1.0 - self.beta1.powi(self.t as i32)) as f32);
            let v_hat = self.v[i].clone() / ((1.0 - self.beta2.powi(self.t as i32)) as f32);

            // Update parameters
            let update = m_hat * (cooling * self.learning_rate as f32)
                / v_hat.map(|v| v.sqrt() + self.epsilon as f32);
            positions[i] += update;
        }
    }

    pub fn reset(&mut self) {
        self.t = 0;
        for i in 0..self.m.len() {
            self.m[i] = V::zero(self.dim);
            self.v[i] = V::zero(self.dim);
        }
    }
}

/// Main weighted embedder, generic over the spatial index via [`EmbedIndex`].
pub struct WEmbedder<SI: EmbedIndex> {
    // Node data
    positions: Vec<SI::Vec>,
    weights: Vec<f64>,
    forces: Vec<SI::Vec>,
    old_positions: Vec<SI::Vec>,
    positions_log: Vec<(u64, Vec<SI::Vec>)>,

    // Helpers for symmetrification
    query_cache: Vec<Vec<NodeId>>,
    repulsion_mutexes: Vec<Mutex<Vec<usize>>>,

    // Spatial index
    pub spatial_index: SI,

    // Optimizer
    optimizer: AdamOptimizer<SI::Vec>,

    dim: usize,

    // Configuration
    options: EmbedderOptions,
    iteration: usize,
    last_relative_change: Option<f64>,
    print_timings: bool,
}

/// Constructor for const-generic spatial indices that implement `Embedder<'a, D>`.
impl<'a, SI, const D: usize> WEmbedder<SI>
where
    SI: Embedder<'a, D> + EmbedIndex<Vec = crate::dvec::DVec<D>>,
{
    pub fn random(seed: u64, graph: &'a Graph, options: EmbedderOptions) -> Self {
        use crate::dvec::DVec;
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
        let spatial_index = SI::new(&crate::Embedding { positions, graph });

        Self::new(spatial_index, options)
    }
}

impl<SI: EmbedIndex> WEmbedder<SI> {
    pub fn new(spatial_index: SI, options: EmbedderOptions) -> Self {
        let n = spatial_index.num_nodes();
        let learning_rate = options.learning_rate;
        let cooling_factor = options.cooling_factor;
        let dim = if n > 0 {
            spatial_index.position(0).dim()
        } else {
            0
        };

        let positions: Vec<_> = (0..n)
            .map(|node| spatial_index.position(node).clone())
            .collect();

        // Extract weights from graph
        let weights: Vec<f64> = (0..n).map(|node| spatial_index.weight(node)).collect();

        Self {
            positions,
            weights,
            forces: vec![SI::Vec::zero(dim); n],
            old_positions: vec![SI::Vec::zero(dim); n],
            positions_log: Vec::new(),
            query_cache: vec![Vec::with_capacity(10); n],
            repulsion_mutexes: (0..n).map(|_| Mutex::new(Vec::with_capacity(10))).collect(),
            spatial_index,
            optimizer: AdamOptimizer::new(n, dim, learning_rate, cooling_factor),
            print_timings: options.print_timings,
            dim,
            options,
            iteration: 0,
            last_relative_change: None,
        }
    }

    /// Run the embedding algorithm until convergence or max iterations
    pub fn embed(&mut self) -> Vec<SI::Vec> {
        self.embed_with_callback(|_| {})
    }
    /// Run the embedding algorithm until convergence or max iterations
    pub fn embed_with_callback(&mut self, mut callback: impl FnMut(&Self)) -> Vec<SI::Vec> {
        self.optimizer.reset();

        loop {
            callback(self);
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
        let update_start = std::time::Instant::now();
        // Save old positions
        self.old_positions.clone_from(&self.positions);
        if self.iteration.is_multiple_of(10) {
            self.positions_log
                .push((self.iteration as u64, self.old_positions.clone()));
        }

        // Clear forces
        self.forces
            .iter_mut()
            .for_each(|f| *f = SI::Vec::zero(self.dim));
        let reset = update_start.elapsed();

        // Update spatial index
        self.update_spatial_index();
        let update_end = update_start.elapsed();

        // Calculate forces
        self.calculate_attraction_forces();
        let attraction_end = update_start.elapsed();
        self.calculate_repulsion_forces();
        let repulsion_end = update_start.elapsed();

        // Update positions
        self.optimizer.update(&mut self.positions, &self.forces);
        let optimizer_update = update_start.elapsed();

        if self.iteration.is_multiple_of(100) && self.print_timings {
            println!("reset: {}μs", reset.as_micros());
            println!("update index: {}ms", (update_end - reset).as_millis());
            println!(
                "attraction: {}ms",
                (attraction_end - update_end).as_millis()
            );
            println!(
                "repulsion: {}ms",
                (repulsion_end - attraction_end).as_millis()
            );
            println!("adam: {}μs", (optimizer_update - repulsion_end).as_micros());
            println!("total {}ms", update_start.elapsed().as_millis());
        }
    }

    fn update_spatial_index(&mut self) {
        self.spatial_index
            .update_positions(&self.positions, self.last_relative_change);
    }

    fn calculate_attraction_forces(&mut self) {
        let dim = self.dim;
        // Calculate forces for each node in parallel using neighbor lists
        let forces: Vec<SI::Vec> = (0..self.positions.len())
            .into_par_iter()
            .map(|v| {
                let mut force = SI::Vec::zero(dim);

                // Get neighbors from graph
                let neighbors = self.spatial_index.neighbors(v);

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

    fn attraction_force(&self, u: NodeId, v: NodeId) -> SI::Vec {
        let pos_u = self.positions[u].clone();
        let pos_v = self.positions[v].clone();

        let direction = pos_v - pos_u;
        let distance = direction.magnitude();

        if distance == 0.0 {
            // Random displacement if positions are identical
            let mut rng = rand::rng();
            return SI::Vec::from_fn(self.dim, |_| rng.random_range(-0.01..0.01));
        }

        let weight_factor = self.weights[u] * self.weights[v];
        let weighted_distance = distance as f64 / weight_factor;

        if weighted_distance <= 1.0 {
            // Already close enough
            SI::Vec::zero(self.dim)
        } else {
            // Attraction force
            direction * (self.options.attraction_scale / (distance as f64 * weight_factor)) as f32
        }
    }

    pub fn print_stats(&self) {
        let (percision, recall) = self.spatial_index.graph_statistics();
        let f1 = 2. / (recall.recip() + percision.recip());
        eprintln!("i: , percision: {percision:.3}, recall: {recall:.3}, f1: {f1:.3}");
    }

    fn calculate_repulsion_forces(&mut self) {
        self.repulsion_mutexes
            .iter()
            .for_each(|mutex| mutex.lock().unwrap().clear());

        // Stage 1: Query nearest neighbors for all nodes in parallel
        (0..self.positions.len())
            .into_par_iter()
            .zip(self.query_cache.par_iter_mut())
            .for_each(|(v, cache)| {
                cache.clear();
                // Find nearby nodes that might repel
                self.spatial_index.repelling_nodes(v, cache);

                for candidate in cache {
                    self.repulsion_mutexes[*candidate].lock().unwrap().push(v);
                }
            });

        self.repulsion_mutexes
            .par_iter()
            .zip(self.query_cache.par_iter_mut())
            .for_each(|(candidates, cache)| {
                cache.extend(candidates.lock().unwrap().drain(..));
            });

        // Stage 2: Calculate repulsion forces in parallel
        let new_forces: Vec<SI::Vec> = self
            .query_cache
            .par_iter()
            .enumerate()
            .map(|(v, results)| {
                let mut force = self.forces[v].clone(); // Start with existing attraction force

                // Add repulsion forces from all candidates
                for &u in results {
                    let f = self.repulsion_force(v, u);
                    force += f;
                }

                force
            })
            .collect();

        // Update the forces
        self.forces = new_forces;
    }

    fn repulsion_force(&self, v: NodeId, u: NodeId) -> SI::Vec {
        let pos_v = self.positions[v].clone();
        let pos_u = self.positions[u].clone();

        let direction = pos_v - pos_u;
        let distance = direction.magnitude();

        if distance == 0.0 {
            // Random displacement if positions are identical
            let mut rng = rand::rng();
            return SI::Vec::from_fn(self.dim, |_| rng.random_range(-0.01..0.01));
        }

        let weight_factor = self.weights[v] * self.weights[u];
        let weighted_distance = distance as f64 / weight_factor;

        if weighted_distance > 1.0 {
            // Far enough apart
            SI::Vec::zero(self.dim)
        } else {
            // Repulsion force
            direction * (self.options.repulsion_scale / (distance as f64 * weight_factor)) as f32
        }
    }

    fn check_convergence(&mut self) -> bool {
        let (sum_norm_squared, sum_diff_squared, max_squared) = self
            .positions
            .iter()
            .zip(&self.old_positions)
            .map(|(new_pos, old_pos)| {
                let diff = new_pos.clone() - old_pos.clone();
                (old_pos.magnitude_squared(), diff.magnitude_squared())
            })
            .fold(
                (0.0, 0.0, 0f64),
                |(sum_norm, sum_diff, max), (norm, diff)| {
                    (
                        sum_norm + norm as f64,
                        sum_diff + diff as f64,
                        max.max(diff as f64),
                    )
                },
            );

        if sum_norm_squared == 0.0 {
            return false;
        }

        let relative_change = sum_diff_squared / sum_norm_squared;
        self.last_relative_change = Some(max_squared.sqrt());
        relative_change < self.options.min_position_change
    }

    /// Get the current positions
    pub fn positions(&self) -> &[SI::Vec] {
        &self.positions
    }

    /// Get the history of positions
    pub fn history(&self) -> &[(u64, Vec<SI::Vec>)] {
        &self.positions_log
    }

    /// Get the current iteration count
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the current query_cache
    pub fn query_cache(&self) -> &[Vec<NodeId>] {
        &self.query_cache
    }

    // Get the last position delta
    pub fn last_pos_delta(&self) -> &Option<f64> {
        &self.last_relative_change
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Embedding,
        dvec::DVec,
        graph::Graph,
        query::{Embedder, Graph as _},
    };

    use super::{EmbedderOptions, WEmbedder};

    #[test]
    fn check_convergence() {
        let node1 = DVec::new([-0.1, 0.0]);
        let node2 = DVec::new([0.0, 1.0]);
        let node3 = DVec::new([0.1, 0.0]);

        let edges = vec![(0, 1), (1, 2)];

        let graph = Graph::from_edge_list(edges, 2, 2).unwrap();

        let embedding = Embedding {
            positions: vec![node1, node2, node3],
            graph: &graph,
        };
        let mut options = EmbedderOptions::default();
        options.learning_rate = 1.0;

        let mut embedder = WEmbedder::new(embedding, options);

        for i in 0..1000 {
            println!("\n\nIteration {i} \n\n");
            for a in [0, 1, 2] {
                println!("node {a} pos: {:?}", embedder.positions[a]);
                for b in [0, 1, 2] {
                    if b == a {
                        continue;
                    }
                    if graph.is_connected(a, b) {
                        println!("attraction to {b} {:?}", embedder.attraction_force(a, b));
                    } else {
                        println!("repulsion to {b} {:?}", embedder.repulsion_force(a, b));
                    }
                }
            }
            embedder.calculate_step();
            let (percision, recall) = embedder.spatial_index.graph_statistics();
            let f1 = 2. / (recall.recip() + percision.recip());
            println!("i: , percision: {percision}, recall: {recall}, f1: {f1}");
            if f1 == 1. {
                if i > 10 {
                    panic!();
                }
                break;
            }
        }
    }
    #[test]
    fn knowledge_graph() {
        let nodes = [
            ((0, 0), "Pdf"),
            ((-1, 0), "Theory"),
            ((-1, 1), "Seperators"),
            ((-2, 1), "O(n)"),
            ((-1, -1), "Lower Bounds"),
            ((0, -1), "Literature"),
            ((-1, -2), "Weighted Space"),
            ((0, -2), "Math. Bounds"),
            ((1, -2), "Ex. Implementations"),
            ((1, 0), "Implementation"),
            ((1, -1), "Symmetric Queries"),
            ((1, 1), "Optimization"),
            ((2, 1), "GPU"),
            ((3, 1), "Dim Reduction"),
            ((2, 0), "Persistance"),
            ((3, -1), "Weight classes"),
            ((2, -1), "Radius Reduction"),
            ((2, -2), "Grid"),
            ((3, -2), "Tree"),
            ((0, 1), "Evaluation"),
            ((1, 2), "Existing Libs"),
            ((0, 3), "KD"),
            ((1, 3), "RTree"),
            ((2, 3), "BallTree"),
            ((3, 3), "VP"),
            ((4, 3), "SNN"),
            ((-1, 2), "Benchmarking"),
            ((-2, 2), "Perf Events"),
            ((-1, 3), "Database"),
            ((-2, 3), "Testing"),
            ((-3, 3), "Plotting"),
        ];

        let edges = vec![
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (0, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (0, 9),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (16, 17),
            (16, 18),
            (0, 19),
            (19, 20),
            (20, 21),
            (20, 22),
            (20, 23),
            (20, 24),
            (20, 25),
            (19, 26),
            (26, 27),
            (26, 28),
            (26, 29),
            (26, 30),
        ];

        let graph = Graph::from_edge_list(edges.clone(), 2, 5).unwrap();

        let positions = nodes
            .iter()
            .map(|x| [x.0.0 as f32, x.0.1 as f32].into())
            .collect();

        let embedding = Embedding {
            positions,
            graph: &graph,
        };
        let mut options = EmbedderOptions::default();
        options.learning_rate = 0.8;
        options.repulsion_scale = 6.;

        let mut embedder = WEmbedder::new(embedding, options);

        for i in 0..1000 {
            embedder.calculate_step();
            let (percision, recall) = embedder.spatial_index.graph_statistics();
            let f1 = 2. / (recall.recip() + percision.recip());
            println!("i: {i}, percision: {percision}, recall: {recall}, f1: {f1}");
            if f1 == 1. {
                if i > 600 {
                    for (i, pos) in embedder.positions.iter().enumerate() {
                        println!(
                            "node(({} * scale_x, {} * scale_y), \"{}\", name: <l{}>),",
                            pos[0], pos[1], nodes[i].1, i
                        );
                    }
                    for (from, to) in &edges {
                        println!(
                            "edge(<l{}>, <l{}>), // ({}, {})",
                            from, to, nodes[*from].1, nodes[*to].1,
                        )
                    }
                    panic!();
                }
                break;
            }
        }
    }
}
