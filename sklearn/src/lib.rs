//! Low-level bindings for scikit-learn's spatial data structures.
//!
//! This crate provides efficient Rust bindings to scikit-learn's KDTree and BallTree
//! implementations using PyO3 and NumPy for minimal data transfer overhead.

use std::sync::Once;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

static PYTHON_INIT: Once = Once::new();

/// Initialize Python without installing signal handlers.
/// This preserves Ctrl+C behavior in the host application.
fn ensure_python_initialized() {
    PYTHON_INIT.call_once(|| {
        unsafe {
            // Pre-initialize with config to disable signal handlers
            let mut preconfig: pyo3::ffi::PyPreConfig = std::mem::zeroed();
            pyo3::ffi::PyPreConfig_InitIsolatedConfig(&mut preconfig);
            pyo3::ffi::Py_PreInitialize(&preconfig);

            let mut config: pyo3::ffi::PyConfig = std::mem::zeroed();
            pyo3::ffi::PyConfig_InitIsolatedConfig(&mut config);
            config.install_signal_handlers = 0;

            pyo3::ffi::Py_InitializeFromConfig(&config);
            pyo3::ffi::PyConfig_Clear(&mut config);
        }
    });
}

/// Opaque handle to a scikit-learn KDTree
pub struct SklearnKDTreeIndex {
    tree: PyObject,
    num_points: usize,
    dimensions: usize,
}

/// Opaque handle to a scikit-learn BallTree
pub struct SklearnBallTreeIndex {
    tree: PyObject,
    num_points: usize,
    dimensions: usize,
}

/// Result from a radius query
pub struct SklearnResult {
    pub indices: Vec<usize>,
}

impl SklearnKDTreeIndex {
    /// Create a new KDTree index from flat position data.
    ///
    /// # Arguments
    /// * `points` - Flat array of f32 coordinates (row-major: [x0, y0, z0, x1, y1, z1, ...])
    /// * `num_points` - Number of points
    /// * `dimensions` - Dimensionality of each point
    /// * `leaf_size` - Maximum number of points in a leaf node (sklearn default: 40)
    pub fn create(
        points: &[f32],
        num_points: usize,
        dimensions: usize,
        leaf_size: usize,
    ) -> PyResult<Self> {
        ensure_python_initialized();
        Python::with_gil(|py| {
            let sklearn_neighbors = py.import("sklearn.neighbors")?;
            let kdtree_class = sklearn_neighbors.getattr("KDTree")?;

            // Convert to f64 for sklearn (it uses float64 internally)
            let points_f64: Vec<f64> = points.iter().map(|&x| x as f64).collect();

            // Create numpy array directly from slice and reshape
            let array = PyArray1::from_slice(py, &points_f64);
            let reshaped = array.reshape([num_points, dimensions])?;

            // Create KDTree
            let kwargs = PyDict::new(py);
            kwargs.set_item("leaf_size", leaf_size)?;
            let tree = kdtree_class.call((reshaped,), Some(&kwargs))?;

            Ok(Self {
                tree: tree.unbind(),
                num_points,
                dimensions,
            })
        })
    }

    /// Perform a radius search around a query point.
    ///
    /// # Arguments
    /// * `query_point` - The query point coordinates (length must equal dimensions)
    /// * `radius` - Search radius (Euclidean distance)
    ///
    /// # Returns
    /// Indices of all points within the radius
    pub fn radius_search(&self, query_point: &[f32], radius: f64) -> PyResult<SklearnResult> {
        Python::with_gil(|py| {
            let query_f64: Vec<f64> = query_point.iter().map(|&x| x as f64).collect();
            let dimensions = query_point.len();

            let query_array = PyArray1::from_slice(py, &query_f64);
            let query_reshaped = query_array.reshape([1, dimensions])?;

            let tree = self.tree.bind(py);
            let result = tree.call_method1("query_radius", (query_reshaped, radius))?;

            // Result is an array of arrays; get first element
            let indices_array = result.get_item(0)?;
            let indices: Vec<usize> = indices_array.extract()?;

            Ok(SklearnResult { indices })
        })
    }

    /// Get the number of points in the index
    pub fn point_count(&self) -> usize {
        self.num_points
    }

    /// Get the dimensionality of the index
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl SklearnBallTreeIndex {
    /// Create a new BallTree index from flat position data.
    ///
    /// # Arguments
    /// * `points` - Flat array of f32 coordinates (row-major: [x0, y0, z0, x1, y1, z1, ...])
    /// * `num_points` - Number of points
    /// * `dimensions` - Dimensionality of each point
    /// * `leaf_size` - Maximum number of points in a leaf node (sklearn default: 40)
    pub fn create(
        points: &[f32],
        num_points: usize,
        dimensions: usize,
        leaf_size: usize,
    ) -> PyResult<Self> {
        ensure_python_initialized();
        Python::with_gil(|py| {
            let sklearn_neighbors = py.import("sklearn.neighbors")?;
            let balltree_class = sklearn_neighbors.getattr("BallTree")?;

            // Convert to f64 for sklearn
            let points_f64: Vec<f64> = points.iter().map(|&x| x as f64).collect();

            // Create numpy array directly from slice and reshape
            let array = PyArray1::from_slice(py, &points_f64);
            let reshaped = array.reshape([num_points, dimensions])?;

            // Create BallTree
            let kwargs = PyDict::new(py);
            kwargs.set_item("leaf_size", leaf_size)?;
            let tree = balltree_class.call((reshaped,), Some(&kwargs))?;

            Ok(Self {
                tree: tree.unbind(),
                num_points,
                dimensions,
            })
        })
    }

    /// Perform a radius search around a query point.
    ///
    /// # Arguments
    /// * `query_point` - The query point coordinates (length must equal dimensions)
    /// * `radius` - Search radius (Euclidean distance)
    ///
    /// # Returns
    /// Indices of all points within the radius
    pub fn radius_search(&self, query_point: &[f32], radius: f64) -> PyResult<SklearnResult> {
        Python::with_gil(|py| {
            let query_f64: Vec<f64> = query_point.iter().map(|&x| x as f64).collect();
            let dimensions = query_point.len();

            let query_array = PyArray1::from_slice(py, &query_f64);
            let query_reshaped = query_array.reshape([1, dimensions])?;

            let tree = self.tree.bind(py);
            let result = tree.call_method1("query_radius", (query_reshaped, radius))?;

            // Result is an array of arrays; get first element
            let indices_array = result.get_item(0)?;
            let indices: Vec<usize> = indices_array.extract()?;

            Ok(SklearnResult { indices })
        })
    }

    /// Get the number of points in the index
    pub fn point_count(&self) -> usize {
        self.num_points
    }

    /// Get the dimensionality of the index
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// PyObject is Send when GIL is properly acquired
unsafe impl Send for SklearnKDTreeIndex {}
unsafe impl Sync for SklearnKDTreeIndex {}
unsafe impl Send for SklearnBallTreeIndex {}
unsafe impl Sync for SklearnBallTreeIndex {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_creation() {
        // 4 points in 2D
        let points: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = SklearnKDTreeIndex::create(&points, 4, 2, 40).expect("Failed to create KDTree");

        assert_eq!(tree.point_count(), 4);
        assert_eq!(tree.dimensions(), 2);
    }

    #[test]
    fn test_kdtree_radius_search() {
        // 4 points in 2D: (0,0), (1,0), (0,1), (1,1)
        let points: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = SklearnKDTreeIndex::create(&points, 4, 2, 40).expect("Failed to create KDTree");

        // Search from (0,0) with radius 1.1 should find (0,0), (1,0), (0,1)
        let query = vec![0.0f32, 0.0];
        let result = tree.radius_search(&query, 1.1).expect("Search failed");

        assert!(result.indices.contains(&0)); // (0,0)
        assert!(result.indices.contains(&1)); // (1,0)
        assert!(result.indices.contains(&2)); // (0,1)
        assert!(!result.indices.contains(&3)); // (1,1) is sqrt(2) away
    }

    #[test]
    fn test_balltree_creation() {
        let points: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree =
            SklearnBallTreeIndex::create(&points, 4, 2, 40).expect("Failed to create BallTree");

        assert_eq!(tree.point_count(), 4);
        assert_eq!(tree.dimensions(), 2);
    }

    #[test]
    fn test_balltree_radius_search() {
        let points: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree =
            SklearnBallTreeIndex::create(&points, 4, 2, 40).expect("Failed to create BallTree");

        let query = vec![0.0f32, 0.0];
        let result = tree.radius_search(&query, 1.1).expect("Search failed");

        assert!(result.indices.contains(&0));
        assert!(result.indices.contains(&1));
        assert!(result.indices.contains(&2));
        assert!(!result.indices.contains(&3));
    }
}
