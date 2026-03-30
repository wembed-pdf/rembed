use crate::scalar::Scalar;
#[cfg(feature = "svd")]
use faer_core;
#[cfg(feature = "svd")]
use faer_core::Mat;
// use nalgebra::{DMatrix, DVector};
#[cfg(feature = "svd")]
use faer_svd;

#[derive(Clone, Debug)]
pub struct SVD<const D: usize, F: Scalar> {
    #[cfg(feature = "svd")]
    mean: [F; D],
    #[cfg(feature = "svd")]
    vt: Mat<F>,
    _phantom: std::marker::PhantomData<[F; D]>,
}

impl<const D: usize, F: Scalar + Default> Default for SVD<D, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize, F: Scalar> SVD<D, F> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "svd")]
            mean: [F::ZERO; D],
            #[cfg(feature = "svd")]
            vt: Mat::<F>::zeros(D, D),
            _phantom: std::marker::PhantomData,
        }
    }
    #[cfg(not(feature = "svd"))]
    pub fn compute_svd(&mut self, _: &[[F; D]]) {}

    #[cfg(feature = "svd")]
    pub fn compute_svd(&mut self, data: &[[F; D]]) {
        let n = data.len();
        if n == 0 {
            eprintln!("Error: Input data is empty.");
            return;
        }

        // Compute mean
        for i in 0..D {
            self.mean[i] = data.iter().map(|v| v[i]).sum::<F>() / F::from_usize(n).unwrap();
        }

        // Center data
        // let mut centered_data = Mat::<F>::zeros(n, D);
        // for (i, v) in data.iter().enumerate() {
        //     for j in 0..D {
        //         centered_data.write(i, j, v[j] - self.mean[j]);
        //     }
        // }

        let centered_data = Mat::<F>::from_fn(n, D, |i, j| data[i][j] - self.mean[j]);

        // Compute covariance matrix
        // let covariance =
        //     &centered_data.transpose() * &centered_data / F::from_usize(n - 1).unwrap();

        // Compute SVD
        let mut s = Mat::<F>::zeros(D, 1);

        let parallelism = faer_core::Parallelism::None; // You can choose Parallelism::Rayon for multi-threading

        let stack_req: faer_core::dyn_stack::StackReq = faer_svd::compute_svd_req::<F>(
            data.len(),
            D,
            faer_svd::ComputeVectors::Thin,
            faer_svd::ComputeVectors::Thin,
            parallelism,
            faer_svd::SvdParams::default(),
        )
        .unwrap();

        let mut buffer = vec![0u8; stack_req.size_bytes()];
        let stack = faer_core::dyn_stack::PodStack::new(&mut buffer);

        faer_svd::compute_svd(
            centered_data.as_ref(),
            s.as_mut(),
            None,
            Some(self.vt.as_mut()),
            parallelism,
            stack,
            faer_svd::SvdParams::default(),
        );

        // Alternative SVD using nalgebra-lapack for better performance on large datasets
        // use nalgebra_lapack::SVD as LapackSVD;
        // let svd = LapackSVD::new(centered_data);
        // self.vt = svd.unwrap().vt;
    }

    #[cfg(not(feature = "svd"))]
    pub fn project(&self, point: &[F; D]) -> [F; D] {
        *point
    }

    #[cfg(feature = "svd")]
    pub fn project(&self, point: &[F; D]) -> [F; D] {
        let centered = faer_core::Col::<F>::from_fn(D, |j| point[j] - self.mean[j]);

        let projected = self.vt.clone() * centered;
        let mut output = [F::ZERO; D];
        for j in 0..D {
            output[j] = projected.read(j);
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct DynamicSVD<F: Scalar> {
    #[cfg(feature = "svd")]
    mean: Vec<F>,
    #[cfg(feature = "svd")]
    vt: Mat<F>,
    normalization_factor: F,
}

impl<F: Scalar + Default> Default for DynamicSVD<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Scalar> DynamicSVD<F> {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "svd")]
            mean: Vec::new(),
            #[cfg(feature = "svd")]
            vt: Mat::<F>::zeros(0, 0),
            normalization_factor: F::ONE,
        }
    }

    #[cfg(not(feature = "svd"))]
    pub fn compute_svd(&mut self, _data: &[&[F]]) {}

    #[cfg(feature = "svd")]
    pub fn compute_svd(&mut self, data: &[&[F]]) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let d = data[0].len();
        self.vt = Mat::<F>::zeros(d, d);

        // Compute mean
        self.mean = vec![F::ZERO; d];
        for i in 0..d {
            self.mean[i] = data.iter().map(|v| v[i]).sum::<F>() / F::from_usize(n).unwrap();
        }

        // Center data
        let mut centered_data = Mat::from_fn(n, d, |i, j| data[i][j] - self.mean[j]);

        // Normalize data to improve numerical stability
        let mut max_abs_value = F::zero();
        for i in 0..n {
            for j in 0..d {
                let abs_value = num_traits::Float::abs(centered_data.read(i, j));
                if abs_value > max_abs_value {
                    max_abs_value = abs_value;
                }
            }
        }
        if max_abs_value > F::ZERO {
            self.normalization_factor = max_abs_value;
            for i in 0..n {
                for j in 0..d {
                    centered_data.write(i, j, centered_data.read(i, j) / self.normalization_factor);
                }
            }
        }

        // Compute covariance matrix
        // let covariance =
        //     &centered_data.transpose() * &centered_data / F::from_usize(n - 1).unwrap();

        // Compute SVD
        // let centered_data_for_svd = centered_data.clone();
        // let svd = centered_data_for_svd.svd(false, true);
        // self.vt = svd.v_t;
        let mut s = Mat::<F>::zeros(d, 1);

        let parallelism = faer_core::Parallelism::None; // You can choose Parallelism::Rayon for multi-threading

        let stack_req: faer_core::dyn_stack::StackReq = faer_svd::compute_svd_req::<F>(
            d,
            data.len(),
            faer_svd::ComputeVectors::Thin,
            faer_svd::ComputeVectors::Thin,
            parallelism,
            faer_svd::SvdParams::default(),
        )
        .unwrap();

        let mut buffer = vec![0u8; stack_req.size_bytes()];
        let stack = faer_core::dyn_stack::PodStack::new(&mut buffer);

        faer_svd::compute_svd(
            centered_data.as_ref(),
            s.as_mut(),
            None,
            Some(self.vt.as_mut()),
            parallelism,
            stack,
            faer_svd::SvdParams::default(),
        );

        // Alternative SVD using nalgebra-lapack for better performance on large datasets
        // use nalgebra_lapack::SVD as LapackSVD;
        // let svd = LapackSVD::new(centered_data);
        // self.vt = Some(svd.unwrap().vt);
    }

    #[cfg(not(feature = "svd"))]
    pub fn project(&self, point: &[F]) -> Vec<F> {
        point.to_vec()
    }

    #[cfg(feature = "svd")]
    pub fn project(&self, point: &[F]) -> Vec<F> {
        let d = self.mean.len();
        let centered = faer_core::Col::<F>::from_fn(d, |j| {
            (point[j] - self.mean[j]) / self.normalization_factor
        });
        let projected = self.vt.as_ref() * centered;
        let mut output = vec![F::ZERO; d];
        for j in 0..d {
            output[j] = projected.read(j);
        }
        output
    }

    pub fn normalize_radius(&self, radius: F) -> F {
        radius / self.normalization_factor
    }
}

// Test SVD implementation with a simple dataset
#[cfg(test)]
mod tests {
    use super::*;

    fn setup_data() -> Vec<[f32; 2]> {
        vec![[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]
    }

    #[test]
    fn test_svd() {
        let data = setup_data();
        let mut svd = SVD::<2, f32>::new();
        svd.compute_svd(&data);
        dbg!(&svd);

        let projected = svd.project(&[1.0, 1.0]);
        println!("Projected point: {:?}", projected);
        panic!("SVD test not fully implemented yet");
    }
}
