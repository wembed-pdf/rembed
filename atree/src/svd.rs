use crate::scalar::Scalar;
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct SVD<const D: usize> {
    mean: [f32; D],
    vt: DMatrix<f32>,
}

impl<const D: usize> SVD<D> {
    pub fn new() -> Self {
        Self {
            mean: [f32::ZERO; D],
            vt: DMatrix::<f32>::zeros(D, D),
        }
    }

    pub fn compute_svd(&mut self, data: &[[f32; D]]) {
        let n = data.len();
        if n == 0 {
            eprintln!("Error: Input data is empty.");
            return;
        }

        // Compute mean
        for i in 0..D {
            self.mean[i] = data.iter().map(|v| v[i]).sum::<f32>()
                / <f32 as crate::scalar::Scalar>::from_usize(n);
        }

        // Center data
        let mut centered_data = DMatrix::<f32>::zeros(n, D);
        for (i, v) in data.iter().enumerate() {
            for j in 0..D {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Compute SVD
        let svd = centered_data.svd(false, true);
        self.vt = svd.v_t.unwrap();
    }

    pub fn project(&self, point: &[f32; D]) -> [f32; D] {
        let mut centered = DVector::<f32>::zeros(D);
        for i in 0..D {
            centered[i] = point[i] - self.mean[i];
        }

        let projection = &self.vt * centered;
        projection
            .data
            .as_slice()
            .try_into()
            .unwrap_or([f32::ZERO; D])
    }
}

#[derive(Clone, Debug)]
pub struct DynamicSVD {
    mean: Vec<f32>,
    vt: Option<DMatrix<f32>>,
    normalization_factor: f32,
}

impl DynamicSVD {
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            vt: None,
            normalization_factor: 1.0,
        }
    }

    pub fn compute_svd(&mut self, data: &[Vec<f32>]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();

        // Compute mean
        self.mean = vec![0.0; d];
        for i in 0..d {
            self.mean[i] = data.iter().map(|v| v[i]).sum::<f32>()
                / <f32 as crate::scalar::Scalar>::from_usize(n);
        }

        // Center data
        let mut centered_data = DMatrix::<f32>::zeros(n, d);
        for (i, v) in data.iter().enumerate() {
            for j in 0..d {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Normalize data to improve numerical stability
        let mut max_abs_value = 0.0;
        for i in 0..n {
            for j in 0..d {
                let abs_value = centered_data[(i, j)].abs();
                if abs_value > max_abs_value {
                    max_abs_value = abs_value;
                }
            }
        }
        if max_abs_value > 0.0 {
            self.normalization_factor = max_abs_value;
            for i in 0..n {
                for j in 0..d {
                    centered_data[(i, j)] /= self.normalization_factor;
                }
            }
        }

        // Compute SVD
        let centered_data_for_svd = centered_data.clone();
        let svd = centered_data_for_svd.svd(false, true);
        self.vt = svd.v_t;
    }

    pub fn project(&self, point: &[f32]) -> Vec<f32> {
        let d = self.mean.len();
        let mut centered = DVector::<f32>::zeros(d);
        for i in 0..d {
            centered[i] = point[i] - self.mean[i];
        }
        centered /= self.normalization_factor; // Apply the same normalization factor used during SVD computation
        let projection = self.vt.as_ref().unwrap() * centered;
        projection.data.as_vec().clone()
    }

    pub fn normalize_radius(&self, radius: f32) -> f32 {
        radius / self.normalization_factor
    }
}
