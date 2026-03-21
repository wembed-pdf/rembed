use crate::{
    Embedding, NodeId, Query,
    atree::simd::PDVec,
    dvec::DVec,
    query::{self, SpatialIndex, Update},
};

#[derive(Clone, Copy, Debug)]
pub struct Point<const D: usize> {
    pos: DVec<D>,
    squared_half: f32,
}

use std::cell::RefCell;

#[derive(Clone, Copy)]
struct LeafRange {
    min_i: usize,
    max_i: usize,
}

thread_local! {
    static SCRATCH: RefCell<Vec<LeafRange>> = RefCell::new(Vec::with_capacity(128));
}

pub mod simd;

impl<const D: usize> Point<D> {
    fn new(pos: DVec<D>) -> Self {
        Self {
            pos,
            squared_half: pos.magnitude_squared() / 2.,
        }
    }
}
impl<const D: usize> std::ops::Index<usize> for Point<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.pos.components[index]
    }
}

const LEAFSIZE: usize = 150;
const W: usize = 8;

#[derive(Clone)]
pub struct ATree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub positions_sorted: Vec<PDVec<D, W>>,
    pub node_ids: Vec<usize>,
    pub d_pos: Vec<f32>,
    pub graph: &'a crate::graph::Graph,
    nodes: Vec<f32>,
    leaves: Vec<Snn>,
    total_depth: usize,
}

impl<const D: usize> crate::query::Graph for ATree<'_, D> {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.graph.is_connected(first, second)
    }

    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.graph.neighbors(index)
    }

    fn weight(&self, index: NodeId) -> f64 {
        self.graph.weight(index)
    }
}

impl<const D: usize> query::Position<D> for ATree<'_, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}
impl<const D: usize> query::Update<D> for ATree<'_, D> {
    fn update_positions(&mut self, postions: &[DVec<D>], _: Option<f64>) {
        if self.positions.len() != postions.len() {
            self.positions = postions.to_vec();
        } else {
            for (old_pos, pos) in self.positions.iter_mut().zip(postions.iter()) {
                *old_pos = *pos;
            }
        }

        let n = postions.len();
        let td = compute_total_depth(n);
        self.total_depth = td;

        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut nodes = std::mem::take(&mut self.nodes);
        let mut leaves = std::mem::take(&mut self.leaves);
        if nodes.len() != num_internal {
            nodes = vec![0.0; num_internal];
        }
        if leaves.len() != num_leaves {
            leaves = vec![Snn::default(); num_leaves];
        }

        let mut node_ids: Vec<_> = (0..n).collect();
        let mut d_pos = vec![0.; node_ids.len()];

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
                let pdvec = PDVec::new(chunk.iter().map(|id| (self.positions[*id], *id)));
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
}

#[derive(Clone, Debug, Default)]
struct Snn {
    lut: Box<[usize]>,
    min: f32,
    resolution: f32,
}

fn build_tree<const D: usize>(
    nodes: &mut [f32],
    leaves: &mut [Snn],
    node_ids: &mut [NodeId],
    d_pos: &mut [f32],
    depth: usize,
    total_depth: usize,
    heap_idx: usize,
    positions: &[DVec<D>],
    offset: usize,
) {
    if depth == total_depth {
        node_ids
            .sort_unstable_by_key(|i| i32::from_ne_bytes(positions[*i][depth % D].to_ne_bytes()));

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
        let resolution = num_buckets as f32 / (max - min);
        for i in 0..num_buckets {
            let boundary = (i as f32 / resolution) + min;
            let start_idx = d_pos.iter().take_while(|&&x| x < boundary).count();
            lut.push(start_idx + offset);
            let next_boundary = ((i + 1) as f32 / resolution) + min;
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
    node_ids.select_nth_unstable_by_key(median_idx, |i| {
        i32::from_ne_bytes(positions[*i][depth % D].to_ne_bytes())
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

fn compute_total_depth(n: usize) -> usize {
    if n <= LEAFSIZE {
        0
    } else {
        (n / LEAFSIZE).ilog2() as usize + 1
    }
}

const fn dim_lut_multiplier<const D: usize>() -> f32 {
    match D {
        x if x <= 2 => 0.1,
        x if x <= 8 => 0.5,
        x if x <= 12 => 0.8,
        x if x > 12 => 1.5,
        _ => unreachable!(),
    }
}
const fn lut_size<const D: usize>() -> usize {
    let multiplier = dim_lut_multiplier::<D>();
    (multiplier * (LEAFSIZE as f32)) as usize
}

#[inline(always)]
fn children(index: usize) -> (usize, usize) {
    (index * 2 + 1, index * 2 + 2)
}

impl<'a, const D: usize> ATree<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let n = embedding.positions.len();
        let td = compute_total_depth(n);
        let num_internal = (1usize << td) - 1;
        let num_leaves = 1usize << td;

        let mut tree = ATree {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            nodes: vec![0.0; num_internal],
            leaves: vec![Snn::default(); num_leaves],
            total_depth: td,
        };
        if !tree.positions.is_empty() {
            tree.update_positions(&embedding.positions, None);
        }
        tree
    }
    pub fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let pos = Point::new(pos);
        let radius_sq = radius.powi(2);

        SCRATCH.with(|scratch| {
            let mut ranges = scratch.borrow_mut();
            ranges.clear();

            // Phase 1: collect leaf ranges — only touches nodes[] + snn
            self.collect_ranges(&pos, 0, 0, radius_sq as f32, &mut DVec::zero(), &mut ranges);

            self.snn(results, pos, radius_sq, ranges);
        });
    }

    fn collect_ranges(
        &self,
        pos: &Point<D>,
        depth: usize,
        heap_idx: usize,
        dim_radius_squared: f32,
        distances: &mut DVec<D>,
        out: &mut Vec<LeafRange>,
    ) {
        let dim = depth % D;

        if depth == self.total_depth {
            let leaf_idx = heap_idx - ((1 << self.total_depth) - 1);
            let snn = &self.leaves[leaf_idx];
            if snn.lut.is_empty() {
                return;
            }

            let own_pos = pos[dim] - snn.min;
            let reduced_radius = (dim_radius_squared + distances[dim]).sqrt();
            let min = own_pos - reduced_radius;
            let max = own_pos + reduced_radius;
            let max_lut = lut_size::<D>() - 1;

            let idx = if min * snn.resolution >= 0.0 {
                (min * snn.resolution) as usize
            } else {
                0
            }
            .min(max_lut);
            let end_idx = if max * snn.resolution >= 0.0 {
                (max * snn.resolution) as usize
            } else {
                0
            }
            .min(max_lut);

            let min_i = snn.lut[idx];
            let max_i = snn.lut[end_idx + snn.lut.len() / 2];

            out.push(LeafRange { min_i, max_i });
            return;
        }

        let split = self.nodes[heap_idx];
        let (left, right) = children(heap_idx);
        let own_pos = pos[dim];
        let current_delta = distances[dim];
        let dist = (own_pos - split).powi(2);
        let other_radius = dim_radius_squared + current_delta - dist;

        // Always left-first for forward-sequential positions_sorted access
        if own_pos < split {
            self.collect_ranges(pos, depth + 1, left, dim_radius_squared, distances, out);
            distances[dim] = dist;
            if other_radius > 0.0 {
                self.collect_ranges(pos, depth + 1, right, other_radius, distances, out);
            }
            distances[dim] = current_delta;
        } else {
            distances[dim] = dist;
            if other_radius > 0.0 {
                self.collect_ranges(pos, depth + 1, left, other_radius, distances, out);
            }
            distances[dim] = current_delta;
            self.collect_ranges(pos, depth + 1, right, dim_radius_squared, distances, out);
        }
    }

    fn snn(
        &self,
        results: &mut Vec<usize>,
        pos: Point<D>,
        radius_sq: f64,
        ranges: std::cell::RefMut<'_, Vec<LeafRange>>,
    ) {
        // Single reserve for everything
        let mut capacity = results.capacity();

        // Phase 2: forward sweep through positions_sorted
        let initial_len = results.len();
        let mut len = results.len();
        let half_radius_threshold = radius_sq as f32 / 2. + 1e-4;
        let radius_sq_f32 = radius_sq as f32;

        for range in ranges.iter() {
            // SAFETY: We need to allocate enough space upfront to allow us to write to the vector without checking if the size is valid
            if len + (range.max_i - range.min_i) * W + W - 1 > capacity {
                results.reserve((len - initial_len) + (range.max_i - range.min_i) * W + W - 1);
                capacity = results.capacity();
            }

            for other_pos in &self.positions_sorted[range.min_i..range.max_i] {
                let results_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        results.as_mut_ptr().wrapping_add(len) as *mut std::mem::MaybeUninit<usize>,
                        W,
                    )
                };
                let new_elements = if D < 6 {
                    let distances = other_pos.dist_squared(pos.pos);
                    other_pos.compare(distances, radius_sq_f32, results_slice.try_into().unwrap())
                } else {
                    let distances = other_pos.dist_half_squared(pos.pos, pos.squared_half);
                    other_pos.compare(
                        distances,
                        half_radius_threshold,
                        results_slice.try_into().unwrap(),
                    )
                };
                len += new_elements;
            }
        }
        unsafe { results.set_len(len) };
    }
}

impl<const D: usize> Query<D> for ATree<'_, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        // let radius = radius.powi(2);
        assert_eq!(self.positions.len(), self.node_ids.len());
        self.query_radius(pos, radius, results);
    }
}
impl<const D: usize> SpatialIndex<D> for ATree<'_, D> {
    fn name(&self) -> String {
        String::from("atree")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("atree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for ATree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}
