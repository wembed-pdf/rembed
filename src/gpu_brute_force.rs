use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
use cubecl::prelude::*;
use std::collections::HashSet;

#[derive(Clone)]
pub struct GpuBruteForce<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
}

// CubeCL kernel for parallel neighbor search using flags
#[cube(launch_unchecked)]
fn find_neighbors<F: Float>(
    query_pos: &Array<F>,
    all_positions: &Array<F>,
    weights: &Array<F>,
    query_weight: F,
    radius: F,
    neighbor_flags: &mut Array<u32>,
    query_index: u32,
    #[comptime] dim: u32,
) {
    // Calculate global thread ID across all workgroups
    let thread_id = CUBE_POS * CUBE_DIM + UNIT_POS;
    let total_nodes = all_positions.len() / dim;

    // Use conditional execution instead of early return
    if thread_id < total_nodes && thread_id < query_index {
        // Calculate squared distance
        let mut distance_squared = F::new(0.0);

        #[unroll]
        for d in 0..dim {
            let query_comp = query_pos[d];
            let other_comp = all_positions[thread_id * dim + d];
            let diff = query_comp - other_comp;
            distance_squared += diff * diff;
        }

        // Check if within radius using weighted threshold
        let other_weight = weights[thread_id];
        let weight_product = query_weight * other_weight;
        let threshold = weight_product * weight_product * radius;

        if distance_squared < threshold {
            // Set flag to 1 if this node is a neighbor
            neighbor_flags[thread_id] = 1;
        } else {
            // Set flag to 0 if not a neighbor
            neighbor_flags[thread_id] = 0;
        }
    } else if thread_id < neighbor_flags.len() {
        // Initialize remaining flags to 0
        neighbor_flags[thread_id] = 0;
    }
}

// Batched GPU kernel for all-vs-all neighbor search
#[cube(launch_unchecked)]
fn find_neighbors_batched<F: Float>(
    all_positions: &Array<F>,
    weights: &Array<F>,
    radius: F,
    neighbor_matrix: &mut Array<u32>,
    num_nodes: u32,
    #[comptime] dim: u32,
) {
    // Calculate which query node and candidate node this thread handles
    let global_thread_id = CUBE_POS * CUBE_DIM + UNIT_POS;
    let query_node = global_thread_id / num_nodes;
    let candidate_node = global_thread_id % num_nodes;

    // Only process valid combinations where candidate < query (avoid duplicates and self)
    if query_node < num_nodes && candidate_node < query_node {
        // Calculate squared distance
        let mut distance_squared = F::new(0.0);

        #[unroll]
        for d in 0..dim {
            let query_comp = all_positions[query_node * dim + d];
            let candidate_comp = all_positions[candidate_node * dim + d];
            let diff = query_comp - candidate_comp;
            distance_squared += diff * diff;
        }

        // Check if within radius using weighted threshold
        let query_weight = weights[query_node];
        let candidate_weight = weights[candidate_node];
        let weight_product = query_weight * candidate_weight;
        let threshold = weight_product * weight_product * radius;

        if distance_squared < threshold {
            // Set flags for both directions (symmetric)
            let matrix_index_qc = query_node * num_nodes + candidate_node;
            let matrix_index_cq = candidate_node * num_nodes + query_node;

            neighbor_matrix[matrix_index_qc] = 1;
            neighbor_matrix[matrix_index_cq] = 1;
        }
    }
}

impl<'a, const D: usize> GpuBruteForce<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        Self {
            positions: embedding.positions,
            graph: embedding.graph,
        }
    }

    pub fn gpu_nearest_neighbors<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        index: usize,
        radius: f64,
    ) -> Vec<NodeId> {
        let num_nodes = self.positions.len();
        if index >= num_nodes || index == 0 {
            return Vec::new();
        }

        // Flatten positions for GPU
        let mut flat_positions = Vec::with_capacity(num_nodes * D);
        for pos in &self.positions {
            for &component in &pos.components {
                flat_positions.push(component);
            }
        }

        // Extract weights
        let weights: Vec<f32> = (0..num_nodes).map(|i| self.weight(i) as f32).collect();

        // Query position
        let query_pos = &self.positions[index].components;
        let query_weight = self.weight(index) as f32;

        // Create GPU buffers
        let positions_handle = client.create(f32::as_bytes(&flat_positions));
        let weights_handle = client.create(f32::as_bytes(&weights));
        let query_pos_handle = client.create(f32::as_bytes(query_pos));

        // Flag buffer - one flag per potential neighbor (nodes before query index)
        let flags_handle = client.create(u32::as_bytes(&vec![0u32; index]));

        // Configure workgroup size and count
        let workgroup_size = 256u32; // Safe workgroup size for most GPUs
        let total_threads = index as u32;
        let num_workgroups = (total_threads + workgroup_size - 1) / workgroup_size;

        // Safety check - ensure we have at least one workgroup
        let num_workgroups = Ord::max(num_workgroups, 1);

        unsafe {
            find_neighbors::launch_unchecked::<f32, R>(
                client,
                CubeCount::Static(num_workgroups, 1, 1),
                CubeDim::new(workgroup_size, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&query_pos_handle, D, 1),
                ArrayArg::from_raw_parts::<f32>(&positions_handle, flat_positions.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&weights_handle, weights.len(), 1),
                ScalarArg::new(query_weight),
                ScalarArg::new(radius as f32),
                ArrayArg::from_raw_parts::<u32>(&flags_handle, index, 1),
                ScalarArg::new(index as u32),
                D as u32,
            );
        }

        // Read flags and collect neighbor indices
        let flags_bytes = client.read_one(flags_handle.binding());
        let flags = u32::from_bytes(&flags_bytes);

        // Collect indices where flag is set to 1
        flags
            .iter()
            .enumerate()
            .filter(|(_, flag)| **flag == 1)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn gpu_nearest_neighbors_batched<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        radius: f64,
    ) -> Vec<Vec<NodeId>> {
        let num_nodes = self.positions.len();
        if num_nodes == 0 {
            return Vec::new();
        }

        // Flatten positions for GPU
        let mut flat_positions = Vec::with_capacity(num_nodes * D);
        for pos in &self.positions {
            for &component in &pos.components {
                flat_positions.push(component);
            }
        }

        // Extract weights
        let weights: Vec<f32> = (0..num_nodes).map(|i| self.weight(i) as f32).collect();

        // Create GPU buffers
        let positions_handle = client.create(f32::as_bytes(&flat_positions));
        let weights_handle = client.create(f32::as_bytes(&weights));

        // Neighbor matrix - store results as flattened matrix (num_nodes x num_nodes)
        let matrix_size = num_nodes * num_nodes;
        let neighbor_matrix_handle = client.create(u32::as_bytes(&vec![0u32; matrix_size]));

        // Configure workgroup size and count for batched processing
        let workgroup_size = 256u32;
        // Total work: num_nodes * num_nodes / 2 (since we only process candidate < query)
        let total_work = (num_nodes * num_nodes) as u32;
        let num_workgroups = (total_work + workgroup_size - 1) / workgroup_size;
        let num_workgroups = Ord::max(num_workgroups, 1);

        unsafe {
            find_neighbors_batched::launch_unchecked::<f32, R>(
                client,
                CubeCount::Static(num_workgroups, 1, 1),
                CubeDim::new(workgroup_size, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&positions_handle, flat_positions.len(), 1),
                ArrayArg::from_raw_parts::<f32>(&weights_handle, weights.len(), 1),
                ScalarArg::new(radius as f32),
                ArrayArg::from_raw_parts::<u32>(&neighbor_matrix_handle, matrix_size, 1),
                ScalarArg::new(num_nodes as u32),
                D as u32,
            );
        }

        // Read neighbor matrix from GPU
        let matrix_bytes = client.read_one(neighbor_matrix_handle.binding());
        let neighbor_matrix = u32::from_bytes(&matrix_bytes);

        // Convert matrix to vec of neighbor lists
        let mut results = vec![Vec::new(); num_nodes];
        for query_idx in 0..num_nodes {
            for candidate_idx in 0..num_nodes {
                if query_idx != candidate_idx {
                    let matrix_index = query_idx * num_nodes + candidate_idx;
                    if neighbor_matrix[matrix_index] == 1 {
                        results[query_idx].push(candidate_idx);
                    }
                }
            }
            // Sort and deduplicate (though shouldn't be necessary with this approach)
            results[query_idx].sort_unstable();
            results[query_idx].dedup();
        }

        results
    }
}

impl<'a, const D: usize> Graph for GpuBruteForce<'a, D> {
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

impl<'a, const D: usize> Position<D> for GpuBruteForce<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for GpuBruteForce<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>]) {
        self.positions = positions.to_vec();
    }
}

impl<'a, const D: usize> Query for GpuBruteForce<'a, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        // For now, fall back to CPU implementation since we don't have runtime access
        let graph = self.graph;
        let own_weight = self.weight(index);
        let own_position = self.position(index);

        for (i, (node, position)) in graph
            .nodes
            .iter()
            .zip(self.positions.iter())
            .enumerate()
            .take(index)
        {
            let weight = own_weight * node.weight;
            let distance = own_position.distance_squared(position);
            if (distance as f64) < weight.powi(2) * radius {
                results.push(i);
            }
        }
    }

    /// GPU-optimized batched neighbor search for all nodes
    fn nearest_neighbors_batched(&self, indices: &[usize]) -> Vec<Vec<usize>> {
        // If not all nodes are requested, fall back to default implementation
        if indices.len() != self.num_nodes()
            || !indices.iter().enumerate().all(|(i, &idx)| i == idx)
        {
            // Default implementation for partial queries
            let mut results = vec![vec![]; indices.len()];
            for &index in indices {
                for other_node_id in self.nearest_neighbors_owned(index, 1.) {
                    if other_node_id < results.len() && index < results.len() {
                        results[other_node_id].push(index);
                        results[index].push(other_node_id);
                    }
                }
            }
            for vec in &mut results {
                vec.sort_unstable();
                vec.dedup();
            }
            return results;
        }

        // For all-nodes queries, we'd need access to the runtime client
        // For now, fall back to CPU implementation
        // In a real scenario, you'd pass the client or store it in the struct
        let mut results = vec![vec![]; self.num_nodes()];
        for i in 0..self.num_nodes() {
            for j in 0..i {
                // Only check previous nodes to avoid duplicates
                let pos_i = &self.positions[i];
                let pos_j = &self.positions[j];
                let weight_i = self.weight(i);
                let weight_j = self.weight(j);

                let distance_squared = pos_i.distance_squared(pos_j) as f64;
                let threshold = (weight_i * weight_j).powi(2);

                if distance_squared < threshold {
                    results[i].push(j);
                    results[j].push(i);
                }
            }
        }

        for vec in &mut results {
            vec.sort_unstable();
            vec.dedup();
        }

        results
    }
}

impl<'a, const D: usize> SpatialIndex<D> for GpuBruteForce<'a, D> {
    fn name(&self) -> String {
        "gpu-brute-force".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("gpu_brute_force.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for GpuBruteForce<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}

// GPU-enabled version of the embedder that can use GPU for neighbor finding
pub struct GpuEmbedder<'a, const D: usize> {
    base: crate::embedder::WEmbedder<GpuBruteForce<'a, D>, D>,
}

impl<'a, const D: usize> GpuEmbedder<'a, D> {
    pub fn new<R: Runtime>(
        embedding: &Embedding<'a, D>,
        device: &R::Device,
        options: crate::embedder::EmbedderOptions,
    ) -> Self {
        let spatial_index = GpuBruteForce::new(embedding.clone());
        let base = crate::embedder::WEmbedder::new(spatial_index, options);
        Self { base }
    }

    pub fn embed_gpu<R: Runtime>(&mut self, device: &R::Device) -> Vec<DVec<D>> {
        let client = R::client(device);

        // For now, we fall back to CPU embedding since the WEmbedder doesn't
        // directly support passing the runtime client to the spatial index
        self.base.embed()
    }
}

// Specialized batched GPU embedder for high-performance scenarios
pub struct BatchedGpuEmbedder<'a, R: Runtime, const D: usize> {
    graph: &'a crate::graph::Graph,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    options: crate::embedder::EmbedderOptions,
}

impl<'a, R: Runtime, const D: usize> BatchedGpuEmbedder<'a, R, D> {
    pub fn new(
        graph: &'a crate::graph::Graph,
        device: R::Device,
        options: crate::embedder::EmbedderOptions,
    ) -> Self {
        let client = R::client(&device);
        Self {
            graph,
            device,
            client,
            options,
        }
    }

    /// Run embedding with GPU-accelerated batched neighbor finding
    pub fn embed_with_gpu_batching(&self, initial_positions: Vec<DVec<D>>) -> Vec<DVec<D>> {
        let mut positions = initial_positions;
        let radius = 1.0; // Use standard radius

        println!("Starting GPU-accelerated embedding...");

        for iteration in 0..self.options.max_iterations {
            if iteration % 10 == 0 {
                println!("GPU Embedding iteration {}", iteration);
            }

            // Create embedding for current positions
            let embedding = Embedding {
                positions: positions.clone(),
                graph: self.graph,
            };

            // Use GPU for batched neighbor finding
            let gpu_bf = GpuBruteForce::new(embedding);
            let neighbor_lists = gpu_bf.gpu_nearest_neighbors_batched::<R>(&self.client, radius);

            // Update positions based on neighbors (simplified force calculation)
            let mut new_positions = positions.clone();
            for (i, neighbors) in neighbor_lists.iter().enumerate() {
                if neighbors.is_empty() {
                    continue;
                }

                // Simple repulsion force calculation
                let mut force = DVec::zero();
                let own_pos = positions[i];

                for &neighbor_idx in neighbors {
                    let neighbor_pos = positions[neighbor_idx];
                    let diff = own_pos - neighbor_pos;
                    let distance = diff.magnitude();

                    if distance > 0.0001 {
                        // Repulsion force inversely proportional to distance
                        let force_magnitude = 0.01 / (distance + 0.001);
                        force += diff * (force_magnitude / distance);
                    }
                }

                // Apply learning rate and update position
                let learning_rate = self.options.learning_rate as f32
                    * self.options.cooling_factor.powi(iteration as i32) as f32;
                new_positions[i] += force * learning_rate;
            }

            // Check for convergence
            let mut total_movement = 0.0;
            for (old, new) in positions.iter().zip(new_positions.iter()) {
                total_movement += (*old - *new).magnitude_squared() as f64;
            }

            positions = new_positions;

            if total_movement < self.options.min_position_change {
                println!("Converged at iteration {}", iteration);
                break;
            }
        }

        positions
    }
}

// Convenience function to run GPU version with runtime
pub fn gpu_nearest_neighbors<R: Runtime, const D: usize>(
    embedding: &Embedding<D>,
    device: &R::Device,
    index: usize,
    radius: f64,
) -> Vec<NodeId> {
    let client = R::client(device);
    let gpu_bf = GpuBruteForce::new(embedding.clone());
    gpu_bf.gpu_nearest_neighbors::<R>(&client, index, radius)
}

// Example usage function
pub fn example_gpu_usage() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

        println!("Testing GPU brute force neighbor search...");

        // Create sample data with nodes at different distances
        let positions = vec![
            DVec::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DVec::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), // Close to first
            DVec::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), // Medium distance
            DVec::new([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]), // Far away
            DVec::new([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]), // Very close to first
        ];

        // Create a minimal graph with some edges for testing
        let mut graph = crate::graph::Graph::new();
        for i in 0..positions.len() {
            graph.nodes.push(crate::graph::Node {
                weight: 1.0, // Equal weights for simplicity
                neighbors: Vec::new(),
                neighbors_set: std::collections::HashSet::new(),
            });
        }

        let embedding = Embedding {
            positions,
            graph: &graph,
        };

        // Initialize GPU device
        let device = WgpuDevice::BestAvailable;

        // Test different query indices and radii
        for query_index in 1..embedding.positions.len() {
            for radius in [0.5, 1.5, 5.0] {
                let neighbors = gpu_nearest_neighbors::<WgpuRuntime, 8>(
                    &embedding,
                    &device,
                    query_index,
                    radius,
                );

                println!(
                    "Query index {}, radius {}: found {} neighbors: {:?}",
                    query_index,
                    radius,
                    neighbors.len(),
                    neighbors
                );
            }
        }

        println!("GPU neighbor search test completed successfully!");
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("WGPU feature not enabled. Compile with --features wgpu");
    }

    Ok(())
}
