use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::tree::{ATree, LeafRange, Point, W};

impl<const D: usize, F: Scalar, I: IdStorage> ATree<D, F, I>
where
    usize: QueryOutput<I, F>,
{
    /// Returns a streaming iterator over all point indices within `radius` of `pos`.
    ///
    /// Unlike `query_radius`, this avoids allocating a results Vec. Internally it
    /// processes one SIMD batch (W elements) at a time and yields IDs one by one.
    ///
    /// Call `.with_distances()` on the result to get `(index, squared_distance)` pairs instead.
    pub fn query_radius_streaming(&self, pos: &[F; D], radius: F) -> RadiusIter<'_, D, F, I> {
        RadiusIter(self.make_iter_core(pos, radius))
    }

    /// Returns a streaming iterator yielding `(index, squared_distance)` pairs.
    pub fn query_radius_streaming_with_distances(
        &self,
        pos: &[F; D],
        radius: F,
    ) -> RadiusDistIter<'_, D, F, I> {
        RadiusDistIter(self.make_iter_core(pos, radius))
    }

    fn make_iter_core(&self, pos: &[F; D], radius: F) -> RadiusIterCore<'_, D, F, I> {
        let pos = Point::new(*pos);
        let radius_sq = radius * radius;
        let mut ranges = Vec::new();
        let total_pdvecs =
            self.collect_ranges(&pos, 0, 0, radius_sq, &mut [F::ZERO; D], &mut ranges);
        RadiusIterCore::new(self, pos, radius_sq, ranges, total_pdvecs)
    }
}

/// Core state shared between `RadiusIter` and `RadiusDistIter`.
///
/// Processes one SIMD batch (W elements) at a time and buffers the
/// compressed IDs and distances for element-by-element consumption.
pub struct RadiusIterCore<'a, const D: usize, F: Scalar, I: IdStorage> {
    tree: &'a ATree<D, F, I>,
    pos: Point<D, F>,
    radius_sq: F,
    half_radius_threshold: F,
    ranges: Vec<LeafRange>,
    range_idx: usize,
    pdvec_idx: usize,
    range_end: usize,
    // Buffers from last compress
    buf_ids: [I; W],
    buf_dists: [F; W],
    buf_count: usize,
    buf_pos: usize,
    // Upper-bound tracking for size_hint
    remaining_pdvecs: usize,
}

impl<'a, const D: usize, F: Scalar, I: IdStorage> RadiusIterCore<'a, D, F, I> {
    fn new(
        tree: &'a ATree<D, F, I>,
        pos: Point<D, F>,
        radius_sq: F,
        ranges: Vec<LeafRange>,
        total_pdvecs: usize,
    ) -> Self {
        let half_radius_threshold = radius_sq * F::HALF + F::from_f32(1e-4);
        let (pdvec_idx, range_end) = if let Some(r) = ranges.first() {
            (r.min_i, r.max_i)
        } else {
            (0, 0)
        };
        let remaining_pdvecs = total_pdvecs;

        RadiusIterCore {
            tree,
            pos,
            radius_sq,
            half_radius_threshold,
            ranges,
            range_idx: 0,
            pdvec_idx,
            range_end,
            buf_ids: [I::SENTINEL; W],
            buf_dists: [F::ZERO; W],
            buf_count: 0,
            buf_pos: 0,
            remaining_pdvecs,
        }
    }

    /// Fill buffer from the next PDVec. Returns false if exhausted.
    #[inline(always)]
    fn fill_buf(&mut self) -> bool {
        loop {
            if self.pdvec_idx < self.range_end {
                let pdvec = &self.tree.positions_sorted[self.pdvec_idx];
                self.pdvec_idx += 1;
                self.remaining_pdvecs -= 1;

                let (count, ids, dists) = if D < 6 {
                    let distances = pdvec.dist_squared(self.pos.pos);
                    pdvec.compress(distances, self.radius_sq)
                } else {
                    let distances =
                        pdvec.dist_half_squared(self.pos.pos, self.pos.squared_half);
                    pdvec.compress(distances, self.half_radius_threshold)
                };

                if count > 0 {
                    self.buf_ids = ids;
                    self.buf_dists = dists;
                    self.buf_count = count;
                    self.buf_pos = 0;
                    return true;
                }
                continue;
            }

            // Advance to next range
            self.range_idx += 1;
            if self.range_idx >= self.ranges.len() {
                return false;
            }
            let range = self.ranges[self.range_idx];
            self.pdvec_idx = range.min_i;
            self.range_end = range.max_i;
        }
    }

    /// Number of buffered items remaining.
    #[inline(always)]
    fn buffered(&self) -> usize {
        self.buf_count - self.buf_pos
    }

    /// Upper bound: buffered + remaining PDVecs × W.
    #[inline(always)]
    fn upper_bound(&self) -> usize {
        self.buffered() + self.remaining_pdvecs * W
    }
}

/// Streaming iterator yielding position indices.
pub struct RadiusIter<'a, const D: usize, F: Scalar, I: IdStorage>(
    RadiusIterCore<'a, D, F, I>,
);

impl<'a, const D: usize, F: Scalar, I: IdStorage> RadiusIter<'a, D, F, I> {
    /// Convert into a distance-yielding iterator.
    ///
    /// Each element becomes `(index, squared_distance)`.
    /// For D < 6, distances come directly from the SIMD path.
    /// For D >= 6, distances are recomputed from positions.
    pub fn with_distances(self) -> RadiusDistIter<'a, D, F, I> {
        RadiusDistIter(self.0)
    }
}

impl<'a, const D: usize, F: Scalar, I: IdStorage> Iterator for RadiusIter<'a, D, F, I> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        let core = &mut self.0;
        if core.buf_pos < core.buf_count {
            let id = core.buf_ids[core.buf_pos];
            core.buf_pos += 1;
            return Some(id.to_usize());
        }
        if core.fill_buf() {
            let id = core.buf_ids[core.buf_pos];
            core.buf_pos += 1;
            Some(id.to_usize())
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.upper_bound()))
    }

    fn fold<B, G>(mut self, init: B, mut f: G) -> B
    where
        G: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        let core = &mut self.0;
        loop {
            // Drain current buffer
            while core.buf_pos < core.buf_count {
                let id = core.buf_ids[core.buf_pos].to_usize();
                core.buf_pos += 1;
                acc = f(acc, id);
            }
            if !core.fill_buf() {
                return acc;
            }
        }
    }
}

/// Streaming iterator yielding `(index, squared_distance)` pairs.
///
/// For low dimensions (D < 6), squared distances come directly from the
/// SIMD distance computation. For higher dimensions, actual squared
/// distances are recomputed from stored positions.
pub struct RadiusDistIter<'a, const D: usize, F: Scalar, I: IdStorage>(
    RadiusIterCore<'a, D, F, I>,
);

impl<'a, const D: usize, F: Scalar, I: IdStorage> Iterator for RadiusDistIter<'a, D, F, I> {
    type Item = (usize, F);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, F)> {
        let core = &mut self.0;
        if core.buf_pos >= core.buf_count && !core.fill_buf() {
            return None;
        }
        let idx = core.buf_pos;
        core.buf_pos += 1;
        let id = core.buf_ids[idx].to_usize();
        let dist_sq = if D < 6 {
            // dist_squared gives actual squared distances
            core.buf_dists[idx]
        } else {
            // dist_half_squared is approximate; recompute
            let other = &core.tree.positions[id];
            (0..D)
                .map(|j| {
                    let d = core.pos.pos[j] - other[j];
                    d * d
                })
                .sum()
        };
        Some((id, dist_sq))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.0.upper_bound()))
    }

    fn fold<B, G>(mut self, init: B, mut f: G) -> B
    where
        G: FnMut(B, Self::Item) -> B,
    {
        let mut acc = init;
        let core = &mut self.0;
        loop {
            while core.buf_pos < core.buf_count {
                let idx = core.buf_pos;
                core.buf_pos += 1;
                let id = core.buf_ids[idx].to_usize();
                let dist_sq = if D < 6 {
                    core.buf_dists[idx]
                } else {
                    let other = &core.tree.positions[id];
                    (0..D)
                        .map(|j| {
                            let d = core.pos.pos[j] - other[j];
                            d * d
                        })
                        .sum()
                };
                acc = f(acc, (id, dist_sq));
            }
            if !core.fill_buf() {
                return acc;
            }
        }
    }
}
