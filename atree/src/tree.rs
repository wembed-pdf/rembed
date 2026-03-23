use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};
use crate::simd::PDVec;

pub(crate) const LEAFSIZE: usize = 150;
pub(crate) const W: usize = 8;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Point<const D: usize, F: Scalar> {
    pub pos: [F; D],
    pub squared_half: F,
}

impl<const D: usize, F: Scalar> Point<D, F> {
    pub fn new(pos: [F; D]) -> Self {
        Self {
            pos,
            squared_half: pos.iter().copied().map(|x| x * x).sum::<F>() * F::HALF,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LeafRange {
    pub min_i: usize,
    pub max_i: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Snn<F: Scalar> {
    pub lut: Box<[usize]>,
    pub min: F,
    pub resolution: F,
}

#[derive(Clone)]
pub struct ATree<const D: usize, F: Scalar = f32, I: IdStorage = u32> {
    pub(crate) positions: Vec<[F; D]>,
    pub(crate) positions_sorted: Vec<PDVec<D, W, F, I>>,
    pub(crate) node_ids: Vec<usize>,
    pub(crate) d_pos: Vec<F>,
    pub(crate) nodes: Vec<F>,
    pub(crate) leaves: Vec<Snn<F>>,
    pub(crate) total_depth: usize,
}

impl<const D: usize, F: Scalar, I: IdStorage> ATree<D, F, I>
where
    usize: QueryOutput<I, F>,
{
    /// Build a new ATree from a slice of positions.
    /// Each position is identified by its index in the slice.
    pub fn new(positions: &[[F; D]]) -> Self {
        let n = positions.len();
        let td = compute_total_depth(n);
        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut tree = ATree {
            positions: Vec::new(),
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            nodes: vec![F::ZERO; num_internal],
            leaves: vec![Snn::default(); num_leaves],
            total_depth: td,
        };
        if !positions.is_empty() {
            tree.update(positions);
        }
        tree
    }

    /// Rebuild the tree with new positions. Reuses allocations where possible.
    pub fn update(&mut self, positions: &[[F; D]]) {
        if self.positions.len() != positions.len() {
            self.positions = positions.to_vec();
        } else {
            self.positions.copy_from_slice(positions);
        }

        let n = positions.len();
        let td = compute_total_depth(n);
        self.total_depth = td;

        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut nodes = std::mem::take(&mut self.nodes);
        let mut leaves = std::mem::take(&mut self.leaves);
        if nodes.len() != num_internal {
            nodes = vec![F::ZERO; num_internal];
        }
        if leaves.len() != num_leaves {
            leaves = vec![Snn::default(); num_leaves];
        }

        let mut node_ids: Vec<_> = (0..n).collect();
        let mut d_pos = vec![F::ZERO; node_ids.len()];

        build_tree(
            &mut nodes,
            &mut leaves,
            node_ids.as_mut_slice(),
            d_pos.as_mut_slice(),
            0,
            td,
            0,
            &self.positions,
            0,
        );

        self.nodes = nodes;
        self.leaves = leaves;
        self.node_ids = node_ids;
        self.positions_sorted.clear();

        for snn in self.leaves.iter_mut() {
            let offset = snn.lut[0];
            let last = snn.lut.last().expect("empty lut");
            let node_ids = &self.node_ids[offset..*last];
            let new_offset = self.positions_sorted.len();

            for chunk in node_ids.chunks(W) {
                let pdvec =
                    PDVec::new(chunk.iter().map(|id| (self.positions[*id], *id)));
                self.positions_sorted.push(pdvec)
            }
            let half_len = snn.lut.len() / 2;
            for lut_entry in &mut snn.lut[0..half_len] {
                *lut_entry = (*lut_entry - offset) / W + new_offset;
            }
            for lut_entry in &mut snn.lut[half_len..] {
                *lut_entry = (*lut_entry - offset).div_ceil(W) + new_offset;
            }
        }
        self.d_pos = d_pos;
    }

    /// Number of indexed positions.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns the stored position for a given index.
    pub fn position(&self, index: usize) -> &[F; D] {
        &self.positions[index]
    }
}

pub(crate) fn build_tree<const D: usize, F: Scalar>(
    nodes: &mut [F],
    leaves: &mut [Snn<F>],
    node_ids: &mut [usize],
    d_pos: &mut [F],
    depth: usize,
    total_depth: usize,
    heap_idx: usize,
    positions: &[[F; D]],
    offset: usize,
) {
    if depth == total_depth {
        node_ids.sort_unstable_by(|a, b| {
            F::total_cmp(&positions[*a][depth % D], &positions[*b][depth % D])
        });

        for (d_pos, pos) in d_pos
            .iter_mut()
            .zip(node_ids.iter().map(|id| &positions[*id]))
        {
            *d_pos = pos[depth % D];
        }
        let mut lut = vec![];
        let mut end_lut = vec![];
        let min = d_pos[0].floor();
        let slack = d_pos.len() - node_ids.len();
        let max = d_pos.iter().rev().nth(slack).unwrap().ceil();
        let num_buckets = lut_size::<D>();
        let resolution = F::from_usize(num_buckets) / (max - min);
        for i in 0..num_buckets {
            let boundary = F::from_usize(i) / resolution + min;
            let start_idx = d_pos.iter().take_while(|&&x| x < boundary).count();
            lut.push(start_idx + offset);
            let next_boundary = F::from_usize(i + 1) / resolution + min;
            let end_idx = d_pos.iter().take_while(|&&x| x < next_boundary).count();
            end_lut.push(end_idx + offset);
        }
        lut.extend_from_slice(&end_lut);

        let leaf_idx = heap_idx - ((1 << total_depth) - 1);
        leaves[leaf_idx] = Snn {
            lut: lut.into(),
            min: d_pos[0].floor(),
            resolution,
        };
        return;
    }

    let median_idx = node_ids.len() / 2;
    node_ids.select_nth_unstable_by(median_idx, |a, b| {
        F::total_cmp(&positions[*a][depth % D], &positions[*b][depth % D])
    });

    let split = positions[node_ids[median_idx]][depth % D];
    let mut split_pos = median_idx;

    let mut i = 0;
    while i < split_pos {
        if positions[node_ids[i]][depth % D] == split {
            split_pos -= 1;
            node_ids.swap(i, split_pos);
        } else {
            i += 1;
        }
    }

    let slack = d_pos.len() - node_ids.len();
    let (a_ids, b_ids) = node_ids.split_at_mut(split_pos);
    let (a_dpos, b_dpos) = d_pos.split_at_mut(split_pos + slack / 2);

    let (a_id, b_id) = children(heap_idx);
    let depth = depth + 1;

    build_tree(
        nodes,
        leaves,
        a_ids,
        a_dpos,
        depth,
        total_depth,
        a_id,
        positions,
        offset,
    );
    build_tree(
        nodes,
        leaves,
        b_ids,
        b_dpos,
        depth,
        total_depth,
        b_id,
        positions,
        offset + split_pos,
    );

    nodes[heap_idx] = split;
}

pub(crate) fn compute_total_depth(n: usize) -> usize {
    if n <= LEAFSIZE {
        0
    } else {
        (n / LEAFSIZE).ilog2() as usize + 1
    }
}

pub(crate) const fn dim_lut_multiplier<const D: usize>() -> f32 {
    match D {
        x if x <= 2 => 0.1,
        x if x <= 8 => 0.5,
        x if x <= 12 => 0.8,
        x if x > 12 => 1.5,
        _ => unreachable!(),
    }
}

pub(crate) const fn lut_size<const D: usize>() -> usize {
    let multiplier = dim_lut_multiplier::<D>();
    (multiplier * (LEAFSIZE as f32)) as usize
}

#[inline(always)]
pub(crate) fn children(index: usize) -> (usize, usize) {
    (index * 2 + 1, index * 2 + 2)
}
