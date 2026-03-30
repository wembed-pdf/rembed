//! Low-level bindings for scikit-learn's spatial data structures.
//!
//! This crate provides efficient Rust bindings to git@github.com:nla-group/snn.git
//! This requires the snnpy repository to be cloned into the same directory as src e.g. py-snn/snn
//! implementations using PyO3 and NumPy for minimal data transfer overhead.

use std::{os::unix::thread, sync::Once};

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

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

pub struct SnnIndex {
    tree: PyObject,
    num_points: usize,
    dimensions: usize,
    threadpool_limits: PyObject,
}

impl SnnIndex {
    /// Create a new SNN index from flat position data.
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
            let sys = py.import("sys")?;
            let snn_python_path = "/home/tobias/Uni/pdf/pdf/rembed/py-snn/snn/python";
            // println!("Adding {} to Python path", snn_python_path);
            sys.getattr("path")?
                .call_method1("append", (snn_python_path,))?;
            let snn_lib = py.import("snnpy")?;

            // snn = build_snn_model(fmn_train)
            let build_snn_model = snn_lib.getattr("build_snn_model")?;
            let points_f64: Vec<f32> = points.iter().map(|&x| x as f32).collect();
            let points_array = PyArray1::from_slice(py, &points_f64);
            let points_reshaped = points_array.reshape([num_points, dimensions])?;
            let snn = build_snn_model.call1((points_reshaped, leaf_size))?;
            let threadpool_limits = py.import("threadpoolctl")?.getattr("threadpool_limits")?;
            // // with threadpool_limits(limits=1):
            // let threadpool_limits = self.threadpool_limits.bind(py);
            let _threadpool_guard = threadpool_limits.call1((1,))?;

            Ok(Self {
                tree: snn.into(),
                threadpool_limits: threadpool_limits.into(),
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
    pub fn radius_search(&self, query_point: &[f32], radius: f64) -> PyResult<Vec<usize>> {
        Python::with_gil(|py| {
            let query_array: Bound<'_, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>> =
                PyArray1::from_slice(py, &query_point);

            let tree = self.tree.bind(py);
            let result = tree.call_method1("query_radius", (query_array, radius + 1e-2))?;

            Ok(result.extract::<Vec<usize>>()?)
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
unsafe impl Send for SnnIndex {}
unsafe impl Sync for SnnIndex {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_creation() {
        // 4 points in 2D
        let points: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = SnnIndex::create(&points, 4, 2, 40).expect("Failed to create python SNN");

        assert_eq!(tree.point_count(), 4);
        assert_eq!(tree.dimensions(), 2);
    }

    #[test]
    fn test_snn_radius_search() {
        // 4 points in 2D: (0,0), (1,0), (0,1), (1,1)
        let points: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let tree = SnnIndex::create(&points, 4, 2, 40).expect("Failed to create python SNN");

        // Search from (0,0) with radius 1.1 should find (0,0), (1,0), (0,1)
        let query = vec![0.0f32, 0.0];
        let result = tree.radius_search(&query, 1.1).expect("Search failed");

        assert!(result.contains(&0)); // (0,0)
        assert!(result.contains(&1)); // (1,0)
        assert!(result.contains(&2)); // (0,1)
        assert!(!result.contains(&3)); // (1,1) is sqrt(2) away
    }
}
