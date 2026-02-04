use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
use linreg;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;

use crate::{dvec::DVec, graph::Graph};

// Using the method from Estimating the intrinsic dimension  of datasets by a minimal  neighborhood information
// https://www.nature.com/articles/s41598-017-11873-y
// 1. Compute the pairwise distances for each point in the dataset i = 1, …, N
// 2. For each point i find the two shortest distances r 1 and r 2
// 3. For each point i compute mu_i = r 2 / r 1
// 4. Compute the empirical cumulate F emp(μ) by sorting the values of μ in an ascending order through a permutation σ, then define F emp(μ_σ(i)) = i / N
// 5. Fit the points of the plane given by coordinates {(log(μ i ), −log(1 − F emp(μ i )))|i = 1, …, N} with a straight line passing through the origin

pub fn intrinsic_dimension<const D: usize>(positions: &[DVec<D>]) -> f64 {
    // Use kiddo to build a kd-tree for efficient nearest neighbor search
    let kiddo_positions: Vec<[f32; D]> = positions
        .iter()
        .map(|p| {
            let mut arr = [0.0f32; D];
            for i in 0..D {
                arr[i] = p.components[i] as f32;
            }
            arr
        })
        .collect();
    let kdtree = ImmutableKdTree::new_from_slice(&kiddo_positions);

    let mut mus = kiddo_positions
        .par_iter()
        .map(|pos| {
            let results = kdtree.nearest_n::<SquaredEuclidean>(&pos, 3.try_into().unwrap());
            let r1 = (results[1].distance as f64).sqrt(); // first nearest neighbor (skip self)
            let r2 = (results[2].distance as f64).sqrt(); // second nearest neighbor
            r2 / r1
        })
        .collect::<Vec<f64>>();

    // Step 4: Compute empirical cumulative distribution
    mus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mus.truncate((mus.len() as f64 * 0.9) as usize); // remove any NaNs
    let n = mus.len() as f64;
    let mut log_mu = Vec::with_capacity(mus.len());
    let mut log_one_minus_f_emp = Vec::with_capacity(mus.len());
    for (i, &mu) in mus.iter().enumerate() {
        let f_emp: f64 = (i as f64) / n;
        log_mu.push(mu.ln());
        log_one_minus_f_emp.push(-1.0 * (1.0 - f_emp).ln());
    }

    // Step 5: Fit a line through the origin to the points (log_mu, log_one_minus_f_emp)

    // use linreg crate for linear regression

    // Assure that the fit goes through the origin by mirroring the data at the origin
    let mut x_vals = Vec::with_capacity(log_mu.len() * 2);
    let mut y_vals = Vec::with_capacity(log_one_minus_f_emp.len() * 2);
    for i in 0..log_mu.len() {
        x_vals.push(log_mu[i]);
        y_vals.push(log_one_minus_f_emp[i]);
        x_vals.push(-log_mu[i]);
        y_vals.push(-log_one_minus_f_emp[i]);
    }

    let slope = linreg::linear_regression(&x_vals, &y_vals)
        .expect("Linear regression failed")
        .0;

    slope
}
