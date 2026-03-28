use crate::scalar::Scalar;
use nalgebra::{ComplexField, DMatrix, DVector, RealField};

#[derive(Clone, Debug)]
pub struct SVD_Rembed<const D: usize, F: Scalar> {
    mean: [F; D],
    vt: DMatrix<F>,
}

impl<const D: usize, F> SVD_Rembed<D, F>
where
    F: Scalar + num_traits::identities::Zero + ComplexField + RealField,
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

        // print variance of each component
        // let s = svd.singular_values;
        // let total_variance: F = s.iter().map(|&x| x * x).sum();
        // for i in 0..20 {
        //     let variance = s[i] * s[i];
        //     let explained = variance / total_variance;
        //     println!(
        //         "Component {}: variance = {:?}, explained = {:.2}%",
        //         i,
        //         variance,
        //         explained * <F as crate::scalar::Scalar>::from_usize(100)
        //     );
        // }
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
pub struct DynamicSVD {
    mean: Vec<f32>,
    vt: DMatrix<f32>,
}

impl DynamicSVD {
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            vt: DMatrix::<f32>::zeros(0, 0),
        }
    }

    pub fn compute_svd_old(&mut self, data: &[Vec<f32>]) {
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

        // Compute SVD using nalgebra_lapack
        // let svd = centered_data.svd(false, true);

        // Compute SVD using nalgebra_lapack
        use nalgebra_lapack::SVD as LapackSVD;
        let svd = LapackSVD::new(centered_data);
        self.vt = svd.unwrap().vt;
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

        // Compute covariance matrix
        let covariance = &centered_data.transpose() * &centered_data
            / <f32 as crate::scalar::Scalar>::from_usize(n - 1);

        // Compute SVD of covariance matrix
        // let svd = covariance.svd(true, true);
        // self.vt = svd.v_t.unwrap();

        // Compute SVD using nalgebra_lapack
        use nalgebra_lapack::SVD as LapackSVD;
        let svd = LapackSVD::new(covariance);
        self.vt = svd.unwrap().vt;
    }

    pub fn project(&self, point: &[f32]) -> Vec<f32> {
        let d = self.mean.len();
        let mut centered = DVector::<f32>::zeros(d);
        for i in 0..d {
            centered[i] = point[i] - self.mean[i];
        }
        let projection = &self.vt * centered;
        projection.data.as_vec().clone()
    }
}
