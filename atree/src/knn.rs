use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount};
use crate::tree::{ATree, Point, children, lut_size};

/// A nearest-neighbour query result.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Neighbour<F> {
    /// Index of the item in the original dataset.
    pub item: usize,
    /// Squared Euclidean distance from the query point.
    pub distance: F,
}

impl<F: Scalar> PartialEq for Neighbour<F> {
    fn eq(&self, other: &Self) -> bool {
        F::total_cmp(&self.distance, &other.distance) == Ordering::Equal
    }
}

impl<F: Scalar> Eq for Neighbour<F> {}

impl<F: Scalar> PartialOrd for Neighbour<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Scalar> Ord for Neighbour<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        F::total_cmp(&self.distance, &other.distance)
    }
}

// ── Bounded max-heap ────────────────────────────────────────────────

pub(crate) struct KnnHeap<F: Scalar> {
    heap: BinaryHeap<Neighbour<F>>,
    k: usize,
    max_dist: F,
}

impl<F: Scalar> KnnHeap<F> {
    fn new(k: usize) -> Self {
        KnnHeap {
            heap: BinaryHeap::with_capacity(k),
            k,
            max_dist: F::INFINITY,
        }
    }

    fn with_radius(k: usize, radius_sq: F) -> Self {
        KnnHeap {
            heap: BinaryHeap::with_capacity(k),
            k,
            max_dist: radius_sq,
        }
    }

    #[inline(always)]
    fn push(&mut self, item: usize, distance: F) {
        if self.heap.len() < self.k {
            self.heap.push(Neighbour { distance, item });
        } else if F::total_cmp(&distance, &self.heap.peek().unwrap().distance) == Ordering::Less {
            let mut top = self.heap.peek_mut().unwrap();
            *top = Neighbour { distance, item };
        }
    }

    #[inline(always)]
    fn worst_distance(&self) -> F {
        if self.heap.len() < self.k {
            self.max_dist
        } else {
            let top = self.heap.peek().unwrap().distance;
            if F::total_cmp(&top, &self.max_dist) == Ordering::Less {
                top
            } else {
                self.max_dist
            }
        }
    }

    fn into_sorted_vec(self) -> Vec<Neighbour<F>> {
        self.heap.into_sorted_vec()
    }
    fn into_vec(self) -> Vec<Neighbour<F>> {
        self.heap.into_vec()
    }

    fn from_vec(vec: Vec<Neighbour<F>>, k: usize, max_dist: F) -> Self {
        let heap = BinaryHeap::from(vec);

        Self { heap, k, max_dist }
    }
}

// ── KNN queries on ATree ────────────────────────────────────────────

impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> ATree<D, W, F, I>
where
    LaneCount<W>: SupportedLaneCount,
    PDVec<D, W, F, I>: CompressDispatch<W, F, I>,
{
    /// Returns the single nearest neighbour, or `None` if the tree is empty.
    pub fn nearest_one(&self, pos: &[F; D]) -> Option<Neighbour<F>>
    where
        (usize, F): QueryOutput<I, F>,
    {
        if self.is_empty() {
            return None;
        }
        let mut results = self.nearest_n(pos, 1, false);
        results.pop()
    }

    /// Returns up to `k` nearest neighbours sorted by ascending squared
    /// distance.
    ///
    /// First scans the leaf bucket containing the query point to obtain a
    /// tight initial radius, then performs a tree traversal that prunes
    /// subtrees whose minimum distance exceeds the current k-th best.
    pub fn nearest_n(&self, pos: &[F; D], k: usize, sorted: bool) -> Vec<Neighbour<F>>
    where
        (usize, F): QueryOutput<I, F>,
    {
        self.nearest_n_core(pos, k, KnnHeap::new(k), sorted)
    }

    /// Returns up to `k` nearest neighbours within `radius` (Euclidean),
    /// sorted by ascending squared distance.
    ///
    /// Combines a hard radius bound with the k-nearest search: the
    /// traversal prunes by whichever is tighter — the k-th best
    /// distance or `radius²`.
    pub fn nearest_n_within(
        &self,
        pos: &[F; D],
        k: usize,
        radius: F,
        sorted: bool,
    ) -> Vec<Neighbour<F>>
    where
        (usize, F): QueryOutput<I, F>,
    {
        self.nearest_n_core(pos, k, KnnHeap::with_radius(k, radius * radius), sorted)
    }

    fn nearest_n_core(
        &self,
        pos: &[F; D],
        k: usize,
        mut heap: KnnHeap<F>,
        sorted: bool,
    ) -> Vec<Neighbour<F>>
    where
        (usize, F): QueryOutput<I, F>,
    {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let pos = Point::new(*pos);

        // Phase 1: scan the home leaf for a tight initial radius.
        let home_leaf = self.descend_to_leaf(&pos);
        self.scan_leaf_full(&pos, home_leaf, k, &mut heap);

        // Phase 2: tree traversal with ever-refining radius.
        let mut distances = [F::ZERO; D];
        self.knn_traverse(&pos, 0, 0, &mut heap, &mut distances, F::ZERO, home_leaf);

        if sorted {
            heap.into_sorted_vec()
        } else {
            heap.into_vec()
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Navigate from root to the leaf containing `pos`.
    /// Returns the leaf index (0-based into `self.leaves`).
    fn descend_to_leaf(&self, pos: &Point<D, F>) -> usize {
        let mut heap_idx = 0;
        for depth in 0..self.total_depth {
            let dim = depth % D;
            let split = self.nodes[heap_idx];
            let (left, right) = children(heap_idx);
            heap_idx = if pos.pos[dim] < split { left } else { right };
        }
        heap_idx - ((1 << self.total_depth) - 1)
    }

    /// Scan every PDVec in a leaf without LUT narrowing.
    /// Used for the home leaf where we have no radius yet.
    fn scan_leaf_full(&self, pos: &Point<D, F>, leaf_idx: usize, k: usize, heap: &mut KnnHeap<F>)
    where
        (usize, F): QueryOutput<I, F>,
    {
        let snn = &self.leaves[leaf_idx];

        let half_len = snn.lut.len() / 2;
        let start = snn.lut[0];
        let end = snn.lut[half_len + half_len - 1];
        let vec_end = end.min(start + (k.min(end) / W));
        let mut vec: Vec<Neighbour<F>> = Vec::with_capacity(k);

        for (i, &pdvec) in self.positions_sorted[start..vec_end].iter().enumerate() {
            let results_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    vec.as_mut_ptr().wrapping_add(i * W) as *mut std::mem::MaybeUninit<(usize, F)>,
                    W,
                )
            };
            let dists = pdvec.dist_squared(pos.pos);
            pdvec.store_into(dists, results_slice.try_into().unwrap());
        }
        unsafe {
            vec.set_len((vec_end - start) * W);
        }
        *heap = KnnHeap::from_vec(vec, k, heap.max_dist);

        for pdvec in &self.positions_sorted[vec_end..end] {
            let dists = pdvec.dist_squared(pos.pos);
            let threshold = heap.worst_distance();
            let (count, ids, ds) = pdvec.compress(dists, threshold);
            for i in 0..count {
                heap.push(ids[i].to_usize(), ds[i]);
            }
        }
    }

    /// Scan a leaf using the LUT to narrow the PDVec range.
    fn scan_leaf_lut(
        &self,
        pos: &Point<D, F>,
        leaf_idx: usize,
        depth: usize,
        distances: &[F; D],
        sum_dist: F,
        heap: &mut KnnHeap<F>,
    ) {
        let snn = &self.leaves[leaf_idx];
        if snn.lut.is_empty() {
            return;
        }

        let dim = depth % D;

        // Budget available in the split dimension:
        //   heap.worst - (sum of other dimensional costs)
        let dim_budget = heap.worst_distance() - sum_dist + distances[dim];
        if dim_budget <= F::ZERO {
            return;
        }
        let reduced_radius = dim_budget.sqrt();

        let own_pos = pos.pos[dim] - snn.min;
        let min = own_pos - reduced_radius;
        let max = own_pos + reduced_radius;
        let max_lut = lut_size::<D>() - 1;

        let min_scaled = min * snn.resolution;
        let idx = if min_scaled >= F::ZERO {
            F::to_f32(min_scaled) as usize
        } else {
            0
        }
        .min(max_lut);

        let max_scaled = max * snn.resolution;
        let end_idx = if max_scaled >= F::ZERO {
            F::to_f32(max_scaled) as usize
        } else {
            0
        }
        .min(max_lut);

        let half_len = snn.lut.len() / 2;
        let min_i = snn.lut[idx];
        let max_i = snn.lut[end_idx + half_len];

        for pdvec in &self.positions_sorted[min_i..max_i] {
            let dists = pdvec.dist_squared(pos.pos);
            let threshold = heap.worst_distance();
            let (count, ids, ds) = pdvec.compress(dists, threshold);
            for i in 0..count {
                heap.push(ids[i].to_usize(), ds[i]);
            }
        }
    }

    /// Recursive traversal visiting the near side first. Prunes subtrees
    /// whose minimum distance to the query exceeds the current k-th best.
    fn knn_traverse(
        &self,
        pos: &Point<D, F>,
        depth: usize,
        heap_idx: usize,
        heap: &mut KnnHeap<F>,
        distances: &mut [F; D],
        sum_dist: F,
        home_leaf: usize,
    ) {
        if sum_dist >= heap.worst_distance() {
            return;
        }

        if depth == self.total_depth {
            let leaf_idx = heap_idx - ((1 << self.total_depth) - 1);
            if leaf_idx != home_leaf {
                self.scan_leaf_lut(pos, leaf_idx, depth, distances, sum_dist, heap);
            }
            return;
        }

        let dim = depth % D;
        let split = self.nodes[heap_idx];
        let (left, right) = children(heap_idx);
        let own_pos = pos.pos[dim];
        let old_dist = distances[dim];
        let split_dist = {
            let d = own_pos - split;
            d * d
        };
        let far_sum = sum_dist - old_dist + split_dist;

        if own_pos < split {
            // Near side (left)
            self.knn_traverse(pos, depth + 1, left, heap, distances, sum_dist, home_leaf);
            // Far side (right) — only if reachable after heap may have tightened
            distances[dim] = split_dist;
            if far_sum < heap.worst_distance() {
                self.knn_traverse(pos, depth + 1, right, heap, distances, far_sum, home_leaf);
            }
            distances[dim] = old_dist;
        } else {
            // Near side (right)
            self.knn_traverse(pos, depth + 1, right, heap, distances, sum_dist, home_leaf);
            // Far side (left)
            distances[dim] = split_dist;
            if far_sum < heap.worst_distance() {
                self.knn_traverse(pos, depth + 1, left, heap, distances, far_sum, home_leaf);
            }
            distances[dim] = old_dist;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ATree;

    fn brute_force_knn<const D: usize>(
        positions: &[[f32; D]],
        query: &[f32; D],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d: f32 = (0..D).map(|j| (p[j] - query[j]).powi(2)).sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(k);
        dists
    }

    #[test]
    fn nearest_n_matches_brute_force_2d() {
        let mut rng = 42u64;
        let n = 5000;
        let mut positions = Vec::with_capacity(n);
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (rng >> 33) as f32 / (1u64 << 31) as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (rng >> 33) as f32 / (1u64 << 31) as f32;
            positions.push([x, y]);
        }

        let tree: ATree<2> = ATree::new(&positions);

        for k in [1, 5, 10, 50] {
            // Query from a few known positions
            for &qi in &[0usize, 100, 999, 4999] {
                let query = positions[qi];
                let knn = tree.nearest_n(&query, k, true);
                let brute = brute_force_knn(&positions, &query, k);

                assert_eq!(knn.len(), brute.len(), "k={k}, qi={qi}");
                for (got, expected) in knn.iter().zip(brute.iter()) {
                    assert_eq!(got.item, expected.0, "k={k}, qi={qi}: wrong item");
                    assert!(
                        (got.distance - expected.1).abs() < 1e-4,
                        "k={k}, qi={qi}: distance mismatch {:.6} vs {:.6}",
                        got.distance,
                        expected.1
                    );
                }
            }
        }
    }

    #[test]
    fn nearest_n_matches_brute_force_8d() {
        let mut rng = 123u64;
        let n = 3000;
        let mut positions = Vec::with_capacity(n);
        for _ in 0..n {
            let mut p = [0.0f32; 8];
            for v in &mut p {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *v = (rng >> 33) as f32 / (1u64 << 31) as f32;
            }
            positions.push(p);
        }

        let tree: ATree<8> = ATree::new(&positions);

        for k in [1, 10, 25] {
            for &qi in &[0usize, 500, 2999] {
                let query = positions[qi];
                let knn = tree.nearest_n(&query, k, true);
                let brute = brute_force_knn(&positions, &query, k);

                assert_eq!(knn.len(), brute.len(), "k={k}, qi={qi}");
                for (got, expected) in knn.iter().zip(brute.iter()) {
                    assert_eq!(got.item, expected.0, "k={k}, qi={qi}: wrong item");
                }
            }
        }
    }

    #[test]
    fn nearest_one_returns_closest() {
        let positions = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tree: ATree<2> = ATree::new(&positions);

        let result = tree.nearest_one(&[0.1, 0.1]).unwrap();
        assert_eq!(result.item, 0);

        let result = tree.nearest_one(&[0.9, 0.9]).unwrap();
        assert_eq!(result.item, 3);
    }

    #[test]
    fn empty_tree_returns_none() {
        let tree: ATree<2> = ATree::new(&[]);
        assert!(tree.nearest_one(&[0.0, 0.0]).is_none());
        assert!(tree.nearest_n(&[0.0, 0.0], 5, true).is_empty());
    }

    #[test]
    fn k_larger_than_n() {
        let positions = [[0.0f32, 0.0], [1.0, 1.0]];
        let tree: ATree<2> = ATree::new(&positions);
        let results = tree.nearest_n(&[0.0, 0.0], 100, true);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn nearest_n_within_respects_radius() {
        let mut rng = 55u64;
        let n = 5000;
        let mut positions = Vec::with_capacity(n);
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (rng >> 33) as f32 / (1u64 << 31) as f32;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (rng >> 33) as f32 / (1u64 << 31) as f32;
            positions.push([x, y]);
        }

        let tree: ATree<2> = ATree::new(&positions);

        for radius in [0.05f32, 0.1, 0.2] {
            let radius_sq = radius * radius;
            for &qi in &[0usize, 100, 2500] {
                let query = positions[qi];

                // Brute-force: k-nearest within radius
                let k = 10;
                let mut brute: Vec<(usize, f32)> = positions
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let d: f32 = (0..2).map(|j| (p[j] - query[j]).powi(2)).sum();
                        (i, d)
                    })
                    .filter(|&(_, d)| d <= radius_sq)
                    .collect();
                brute.sort_by(|a, b| a.1.total_cmp(&b.1));
                brute.truncate(k);

                let knn = tree.nearest_n_within(&query, k, radius, true);

                assert_eq!(knn.len(), brute.len(), "r={radius}, qi={qi}");
                for (got, expected) in knn.iter().zip(brute.iter()) {
                    assert_eq!(got.item, expected.0, "r={radius}, qi={qi}: wrong item");
                    assert!(
                        got.distance <= radius_sq + 1e-5,
                        "r={radius}, qi={qi}: distance {:.6} exceeds radius²",
                        got.distance,
                    );
                }
            }
        }
    }
}
