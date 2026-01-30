use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal, Uniform};
use rembed::dvec::DVec;
use rembed::graph::{Graph, Node};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum PointDistribution {
    Normal { mean: f32, std_dev: f32 },
    Uniform { min: f32, max: f32 },
}

impl PointDistribution {
    pub fn name(&self) -> &'static str {
        match self {
            PointDistribution::Normal { .. } => "normal",
            PointDistribution::Uniform { .. } => "uniform",
        }
    }

    pub fn standard_normal() -> Self {
        PointDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    pub fn unit_uniform() -> Self {
        PointDistribution::Uniform { min: 0.0, max: 1.0 }
    }
}

/// Generate D-dimensional points with the specified distribution
pub fn generate_points<const D: usize>(
    count: usize,
    distribution: &PointDistribution,
    seed: u64,
) -> Vec<DVec<D>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(count);

    match distribution {
        PointDistribution::Normal { mean, std_dev } => {
            for _ in 0..count {
                let components: [f32; D] = std::array::from_fn(|_| {
                    let sample: f32 = StandardNormal.sample(&mut rng);
                    sample * std_dev + mean
                });
                points.push(DVec::new(components));
            }
        }
        PointDistribution::Uniform { min, max } => {
            let uniform =
                Uniform::new(*min, *max).expect("Invalid uniform distribution parameters");
            for _ in 0..count {
                let components: [f32; D] = std::array::from_fn(|_| uniform.sample(&mut rng));
                points.push(DVec::new(components));
            }
        }
    }

    points
}

/// Create a minimal graph with no edges and uniform weights
/// All nodes have weight = 1.0 and no neighbors
pub fn create_minimal_graph(node_count: usize, weight: f64) -> Graph {
    let nodes: Vec<Node> = (0..node_count)
        .map(|_| Node {
            weight,
            neighbors: Vec::new(),
            neighbors_set: HashSet::new(),
        })
        .collect();

    Graph {
        nodes,
        edges: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_points_normal() {
        let points = generate_points::<2>(100, &PointDistribution::standard_normal(), 42);
        assert_eq!(points.len(), 100);

        // Check mean is approximately 0
        let mean_x: f32 = points.iter().map(|p| p[0]).sum::<f32>() / points.len() as f32;
        let mean_y: f32 = points.iter().map(|p| p[1]).sum::<f32>() / points.len() as f32;
        assert!(mean_x.abs() < 0.5, "Mean X should be close to 0");
        assert!(mean_y.abs() < 0.5, "Mean Y should be close to 0");
    }

    #[test]
    fn test_generate_points_uniform() {
        let points = generate_points::<2>(100, &PointDistribution::unit_uniform(), 42);
        assert_eq!(points.len(), 100);

        // Check all points are in [0, 1]
        for point in &points {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
            assert!(point[1] >= 0.0 && point[1] <= 1.0);
        }
    }

    #[test]
    fn test_create_minimal_graph() {
        let graph = create_minimal_graph(10);
        assert_eq!(graph.nodes.len(), 10);
        assert_eq!(graph.edges.len(), 0);

        for node in &graph.nodes {
            assert_eq!(node.weight, 1.0);
            assert_eq!(node.neighbors.len(), 0);
            assert_eq!(node.neighbors_set.len(), 0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let points1 = generate_points::<4>(50, &PointDistribution::standard_normal(), 123);
        let points2 = generate_points::<4>(50, &PointDistribution::standard_normal(), 123);

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            assert_eq!(
                p1.components, p2.components,
                "Same seed should produce same points"
            );
        }
    }
}
