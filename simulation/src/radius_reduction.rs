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
    BruteForce(Vec<DVec<D>>),
    Split {
        strategy: SplitStrategy,
        dim: usize,
        bucket_starts: Vec<(f32, f32)>,
        children: Vec<Layer<D>>,
    },
}

impl<const D: usize> Layer<D> {
    fn new(mut positions: Vec<DVec<D>>, depth: usize) -> Self {
        if positions.len() <= 150 {
            return Self::BruteForce(positions);
        }
        let dim = depth % D;
        // dbg!(positions.len(), dim);

        // Sort by the specified dimension
        positions.sort_unstable_by(|a, b| a[dim].partial_cmp(&b[dim]).unwrap());

        const NUM_BUCKETS: usize = 2;
        const RESOLUTION: f32 = 2.;

        let element_split = Self::element_split(positions.clone(), dim, NUM_BUCKETS, depth);

        let spatial_split = Self::spatial_split(positions, dim, RESOLUTION, depth);

        // element_split
        spatial_split
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

        if num_buckets <= 1 {
            return Self::BruteForce(positions);
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

#[derive(Default, Debug)]
pub struct Statistics {
    pub num_comparionsons: usize,
    pub actual_matches: usize,
    pub num_splits: usize,
    pub num_reductions: usize,
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
        results: &mut Vec<usize>,
        stats: &mut Statistics,
        use_radius_reduction: bool,
    ) {
        self.query_impl(
            self.positions[id],
            radius,
            DVec::zero(),
            results,
            0,
            &self.root,
            stats,
            use_radius_reduction,
        );
    }

    fn query_impl(
        &self,
        pos: DVec<D>,
        radius: f32,
        spatial_offset: DVec<D>,
        results: &mut Vec<usize>,
        depth: usize,
        layer: &Layer<D>,
        statistics: &mut Statistics,
        use_radius_reduction: bool,
    ) {
        // println!("final_radius: {}", 1. - spatial_offset.magnitude());
        if let Layer::BruteForce(positions) = layer {
            statistics.num_comparionsons += positions.len();
            statistics.actual_matches += positions
                .iter()
                .filter(|x| x.distance_squared(&pos) < radius.powi(2))
                .count();

            if positions.len() > 60 {
                for dp in positions {
                    let close_enough = dp.distance_squared(&pos) < radius.powi(2);
                    // "{} {} {} {} {} {} {} {} {}",
                    println!(
                        "{} {} {} ",
                        dp[0],
                        dp[1],
                        // dp[2],
                        // dp[3],
                        // dp[4],
                        // dp[5],
                        // dp[6],
                        // dp[7],
                        if close_enough { 0 } else { 1 }
                    );
                }
                panic!();
            }
            eprintln!("len: {}", positions.len());
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
            for (i, ((start, end), child)) in bucket_starts.iter().zip(children.iter()).enumerate()
            {
                // dbg!(i, start, end);
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
                if should_recurse_red && use_radius_reduction
                    || should_recurse && !use_radius_reduction
                {
                    // if should_recurse {
                    self.query_impl(
                        pos,
                        radius,
                        new_spatial_offset,
                        results,
                        depth + 1,
                        child,
                        statistics,
                        use_radius_reduction,
                    );
                }
            }
        }
    }
}
