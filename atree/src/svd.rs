use crate::scalar::Scalar;
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Debug)]
pub struct SVD<const D: usize, F: Scalar> {
    mean: [F; D],
    vt: DMatrix<F>,
}

impl<
    const D: usize,
    F: Scalar + num_traits::identities::Zero + nalgebra::ComplexField + nalgebra::RealField,
> SVD<D, F>
{
    pub fn new() -> Self {
        Self {
            mean: [F::ZERO; D],
            vt: DMatrix::<F>::zeros(D, D),
        }
    }

    pub fn compute_svd(&mut self, data: &[[F; D]]) {
        let n = data.len();
        if n == 0 {
            return;
        }

        // Compute mean
        for i in 0..D {
            self.mean[i] =
                data.iter().map(|v| v[i]).sum::<F>() / <F as crate::scalar::Scalar>::from_usize(n);
        }

        // Center data
        let mut centered_data = DMatrix::<F>::zeros(n, D);
        for (i, v) in data.iter().enumerate() {
            for j in 0..D {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Compute SVD
        let svd = centered_data.svd(false, true);
        self.vt = svd.v_t.unwrap();
    }

    pub fn project(&self, point: &[F; D]) -> [F; D] {
        let mut centered = DVector::<F>::zeros(D);
        for i in 0..D {
            centered[i] = point[i] - self.mean[i];
        }

        let projection = &self.vt * centered;
        projection
            .data
            .as_slice()
            .try_into()
            .unwrap_or([F::ZERO; D])
    }
}

#[derive(Clone, Debug)]
pub struct DynamicSVD<F: Scalar> {
    mean: Vec<F>,
    vt: DMatrix<F>,
}

impl<F: Scalar + num_traits::identities::Zero + nalgebra::ComplexField> DynamicSVD<F> {
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            vt: DMatrix::<F>::zeros(0, 0),
        }
    }

    pub fn compute_svd_old(&mut self, data: &[Vec<F>]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();

        // Compute mean
        self.mean = vec![F::ZERO; d];
        for i in 0..d {
            self.mean[i] =
                data.iter().map(|v| v[i]).sum::<F>() / <F as crate::scalar::Scalar>::from_usize(n);
        }

        // Center data
        let mut centered_data = DMatrix::<F>::zeros(n, d);
        for (i, v) in data.iter().enumerate() {
            for j in 0..d {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Compute SVD
        let svd = centered_data.svd(false, true);
        self.vt = svd.v_t.unwrap();
    }

    pub fn project(&self, point: &[F]) -> Vec<F> {
        let d = self.mean.len();
        let mut centered = DVector::<F>::zeros(d);
        for i in 0..d {
            centered[i] = point[i] - self.mean[i];
        }
        let projection = &self.vt * centered;
        projection.data.as_vec().clone()
    }

    pub fn compute_svd(&mut self, data: &[Vec<F>]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();

        // Compute mean
        self.mean = vec![F::ZERO; d];
        for i in 0..d {
            self.mean[i] =
                data.iter().map(|v| v[i]).sum::<F>() / <F as crate::scalar::Scalar>::from_usize(n);
        }

        // Center data
        let mut centered_data = DMatrix::<F>::zeros(n, d);
        for (i, v) in data.iter().enumerate() {
            for j in 0..d {
                centered_data[(i, j)] = v[j] - self.mean[j];
            }
        }

        // Compute covariance matrix
        let covariance = &centered_data.transpose() * &centered_data
            / <F as crate::scalar::Scalar>::from_usize(n - 1);

        // Compute SVD of covariance matrix
        let svd = covariance.svd(true, true);
        self.vt = svd.v_t.unwrap();
    }
}
