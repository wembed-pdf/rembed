use rembed::dvec::DVec;
use std::ops::Mul;

#[derive(Clone, Debug)]
pub enum SplitStrategy {
    Element {
        num_buckets: usize,
    },
    Spatial {
        resolution: f32,
        bucket_offset: isize,
    },
}

pub struct DimReduction<const D: usize> {
    positions: Vec<DVec<D>>,
    root: Layer<D>,
}

#[derive(Debug)]
pub enum Layer<const D: usize> {
    BruteForce {
        positions: Vec<DVec<D>>,
        dim: usize,
    },
    Split {
        strategy: SplitStrategy,
        dim: usize,
        bucket_starts: Vec<(f32, f32)>,
        children: Vec<Layer<D>>,
    },
}

const LEAFSIZE: usize = 150;

impl<const D: usize> Layer<D> {
    fn new(mut positions: Vec<DVec<D>>, depth: usize) -> Self {
        let dim = depth % D;
        if positions.len() <= LEAFSIZE {
            return Self::BruteForce { positions, dim };
        }
        // dbg!(positions.len(), dim);

        // Sort by the specified dimension
        positions.sort_unstable_by(|a, b| a[dim].partial_cmp(&b[dim]).unwrap());

        const NUM_BUCKETS: usize = 2;
        const RESOLUTION: f32 = 2.;

        let element_split = Self::element_split(positions.clone(), dim, NUM_BUCKETS, depth);

        let _spatial_split = Self::spatial_split(positions, dim, RESOLUTION, depth);

        element_split
        // spatial_split
    }

    fn element_split(
        positions: Vec<DVec<D>>,
        dim: usize,
        num_buckets: usize,
        depth: usize,
    ) -> Self {
        let mut children = Vec::new();
        let total_len = positions.len();
        let mut bucket_starts = Vec::new();

        for i in 0..num_buckets {
            let start = i * total_len / num_buckets;
            let end = if i == num_buckets - 1 {
                total_len
            } else {
                (i + 1) * total_len / num_buckets
            };

            if start < end {
                let bucket_positions = positions[start..end].to_vec();

                let child = Layer::new(bucket_positions, depth + 1);
                bucket_starts.push((positions[start][dim], positions[end - 1][dim]));
                children.push(child);
            }
        }

        Self::Split {
            strategy: SplitStrategy::Element { num_buckets },
            dim,
            bucket_starts,
            children,
        }
    }

    fn spatial_split(positions: Vec<DVec<D>>, dim: usize, resolution: f32, depth: usize) -> Self {
        let min = positions[0][dim].mul(resolution).floor();
        let max = positions.last().unwrap()[dim].mul(resolution).floor();
        let num_buckets = (max - min) as usize + 1;

        if num_buckets <= 50 {
            return Self::BruteForce {
                positions,
                dim: depth,
            };
        }

        let mut buckets = vec![Vec::new(); num_buckets];
        let bucket_offset = -min as isize;

        for pos in positions {
            let bucket_key = (pos[dim] * resolution).floor() as isize + bucket_offset;
            buckets[bucket_key as usize].push(pos);
        }
        let bucket_starts: Vec<_> = buckets
            .iter()
            .map(|children| {
                (
                    children.first().unwrap_or(&DVec::from_fn(|_| -100.))[dim],
                    children.last().unwrap_or(&DVec::from_fn(|_| 100.))[dim],
                )
            })
            .collect();

        // Convert buckets to children layers
        let mut children = Vec::new();

        for bucket in buckets.into_iter() {
            let child = Layer::new(bucket, depth + 1);
            children.push(child);
        }

        Self::Split {
            strategy: SplitStrategy::Spatial {
                resolution,
                bucket_offset,
            },
            bucket_starts,
            dim,
            children,
        }
    }
}

pub struct QueryParams {
    use_radius_reduction: bool,
    use_snn: bool,
    best_snn_dim: bool,
    _approx_snn_dim: bool,
    use_snn_with_radius_reduction: bool,
}

impl QueryParams {
    pub fn new(
        use_radius_reduction: bool,
        use_snn: bool,
        best_snn_dim: bool,
        approx_snn_dim: bool,
        use_snn_with_radius_reduction: bool,
    ) -> Self {
        Self {
            use_radius_reduction,
            use_snn,
            best_snn_dim,
            _approx_snn_dim: approx_snn_dim,
            use_snn_with_radius_reduction,
        }
    }
}

#[derive(Default, Debug)]
pub struct Statistics {
    pub num_comparionsons: usize,
    pub num_splits: usize,
    pub num_reductions: usize,
    pub ground_truth_comparisons: usize,
}

impl<const D: usize> DimReduction<D> {
    pub fn new(positions: Vec<DVec<D>>) -> Self {
        let root = Layer::new(positions.clone(), 0);

        Self { positions, root }
    }
    pub fn query(
        &self,
        id: usize,
        radius: f32,
        _results: &mut [usize],
        stats: &mut Statistics,
        params: &QueryParams,
    ) {
        self.query_impl(
            self.positions[id],
            radius,
            DVec::zero(),
            &self.root,
            stats,
            params,
        );
    }

    fn query_impl(
        &self,
        pos: DVec<D>,
        radius: f32,
        spatial_offset: DVec<D>,
        layer: &Layer<D>,
        statistics: &mut Statistics,
        params: &QueryParams,
    ) {
        // println!("final_radius: {}", 1. - spatial_offset.magnitude());
        if let Layer::BruteForce { positions, dim } = layer {
            if params.use_snn {
                let parameter_dim = *dim;
                let mut min_checks = usize::MAX;
                let range = if params.best_snn_dim {
                    0..D
                } else {
                    parameter_dim..(parameter_dim + 1)
                };
                for dim in range {
                    let mut new_spatial_offset = spatial_offset;
                    new_spatial_offset[dim] = 0.;
                    let mut num_comparisons = 0;
                    //project onto next dimension and check if in radius there
                    for p in positions.iter() {
                        let projected_distance = (pos[dim] - p[dim]).abs();
                        let snn_radius = if params.use_snn_with_radius_reduction {
                            (radius.powi(2) - new_spatial_offset.magnitude_squared()).sqrt()
                        } else {
                            radius
                        };
                        if projected_distance <= snn_radius {
                            num_comparisons += 1;
                            // if dim == parameter_dim
                            //     && (pos - *p).magnitude_squared() < radius.powi(2)
                            // {
                            //     statistics.ground_truth_comparisons += 1;
                            // }
                        }
                    }
                    if min_checks > num_comparisons {
                        min_checks = num_comparisons;
                    }
                }
                statistics.num_comparionsons += min_checks;
            } else {
                statistics.num_comparionsons += positions.len();
            }
            // return;
        }

        if let Layer::Split {
            dim,
            bucket_starts,
            children,
            ..
        } = layer
        {
            let p = pos[*dim];
            // dbg!(p);
            for ((start, end), child) in bucket_starts.iter().zip(children.iter())
            {
                // dbg!(start, end);
                let mut new_spatial_offset = spatial_offset;
                if *end <= p {
                    new_spatial_offset[*dim] = p - *end;
                }
                if *start >= p {
                    new_spatial_offset[*dim] = *start - p;
                }

                statistics.num_splits += 1;
                if new_spatial_offset != spatial_offset {
                    statistics.num_reductions += 1;
                }
                // let before_first_or_after_last =
                //     (p < *start && i == 0) || (p > *end && i == bucket_starts.len() - 1);
                // // We know that we can't be in this set
                // if before_first_or_after_last {
                //     return;
                // }
                // assert!(!before_first_or_after_last);

                let should_recurse = (p + radius >= *start && p - radius <= *start)
                    || (p + radius >= *end && p - radius <= *end)
                    || (p >= *start && p <= *end);
                let should_recurse_red = new_spatial_offset.magnitude_squared() < radius.powi(2);
                if should_recurse_red && !should_recurse {
                    // println!("skipped check at depth {depth}");
                    dbg!(
                        pos,
                        dim,
                        new_spatial_offset,
                        pos - new_spatial_offset,
                        start,
                        end
                    );
                }
                if should_recurse_red && params.use_radius_reduction
                    || should_recurse && !params.use_radius_reduction
                {
                    // if should_recurse {
                    self.query_impl(
                        pos,
                        radius,
                        new_spatial_offset,
                        child,
                        statistics,
                        params,
                    );
                }
            }
        }
    }
}
