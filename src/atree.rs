use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Position, SpatialIndex, Update},
};

#[derive(Clone, Copy, Debug)]
pub struct Point<const D: usize> {
    pos: DVec<D>,
    squared_half: f32,
    node_id: u32,
}

impl<const D: usize> Point<D> {
    fn new(pos: DVec<D>, id: usize) -> Self {
        Self {
            pos,
            squared_half: pos.magnitude_squared() / 2.,
            node_id: id as u32,
        }
    }
    // #[inline(never)]
    fn closer_than(&self, other: &Self, half_radius_threshold: f32) -> bool {
        let a = &self.pos.components;
        let b = &other.pos.components;
        let dot: f32 = if D % 4 == 0 {
            let mut acc = [0.0f32; 4];
            let chunks = D / 4;
            for i in 0..chunks {
                let base = i * 4;
                acc[0] += a[base] * b[base];
                acc[1] += a[base + 1] * b[base + 1];
                acc[2] += a[base + 2] * b[base + 2];
                acc[3] += a[base + 3] * b[base + 3];
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else if D % 2 == 0 {
            let mut acc = [0.0f32; 2];
            let chunks = D / 2;
            for i in 0..chunks {
                let base = i * 2;
                acc[0] += a[base] * b[base];
                acc[1] += a[base + 1] * b[base + 1];
            }
            acc[0] + acc[1]
        } else {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
        };
        (self.squared_half + other.squared_half) - dot <= half_radius_threshold
    }
}
impl<const D: usize> std::ops::Index<usize> for Point<D> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.pos.components[index]
    }
}

const LEAFSIZE: usize = 150;

#[derive(Clone)]
pub struct ATree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub positions_sorted: Vec<Point<D>>,
    pub node_ids: Vec<usize>,
    pub d_pos: Vec<f32>,
    pub graph: &'a crate::graph::Graph,
    layers: Vec<Layer>,
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

        let mut node_ids: Vec<_> = (0..postions.len()).collect();

        let mut d_pos = vec![0.; node_ids.len()];
        let mut layers = std::mem::take(&mut self.layers);
        if layers.len() < node_ids.len() {
            layers = vec![Layer::Leaf(Snn::default()); node_ids.len()];
        }
        Layer::init(
            node_ids.as_mut_slice(),
            d_pos.as_mut_slice(),
            &mut layers,
            0,
            0,
            self,
            0,
        );
        self.layers = layers;
        self.node_ids = node_ids;
        self.positions_sorted = self
            .node_ids
            .iter()
            .map(|id| Point::new(*self.position(*id), *id))
            .collect();
        self.d_pos = d_pos;
    }
}

#[derive(Clone, Debug)]
struct Node {
    split: f32,
}

#[derive(Clone, Debug, Default)]
struct Snn {
    lut: Box<[usize]>,
    min: f32,
    resolution: f32,
}

#[derive(Clone, Debug)]
enum Layer {
    Node(Node),
    Leaf(Snn),
}

impl Layer {
    fn init<const D: usize>(
        nodes: &mut [NodeId],
        d_pos: &mut [f32],
        layers: &mut [Layer],
        depth: usize,
        layer_id: usize,
        atree: &ATree<D>,
        offset: usize,
    ) {
        // if nodes.len()
        //     <= ((atree.positions.len() as f64).powf((D as f64).recip()) as usize).max(LEAFSIZE)
        // {
        // if nodes.len() <= (atree.positions.len().isqrt()).max(LEAFSIZE) {
        if nodes.len() <= LEAFSIZE {
            // if nodes.len() <= { 900 } {
            // if nodes.len() <= { 156 } {
            // let depth = D - 1;
            // let depth = 0;
            // For leaf nodes, we need full sorting for the lookup table
            nodes.sort_unstable_by_key(|i| {
                i32::from_ne_bytes(atree.position(*i)[depth].to_ne_bytes())
            });

            for (d_pos, pos) in d_pos
                .iter_mut()
                .zip(nodes.iter().map(|id| atree.position(*id)))
            {
                *d_pos = pos[depth];
            }
            let mut lut = vec![];
            let mut end_lut = vec![];
            let min = d_pos[0].floor();
            let slack = d_pos.len() - nodes.len();
            let max = d_pos.iter().rev().nth(slack).unwrap().ceil();
            let multiplier = match D {
                x if x <= 2 => 0.13,
                x if x <= 8 => 0.5,
                x if x <= 12 => 0.8,
                // x if x <= 12 => 1.5,
                x if x > 12 => 2.,
                _ => unreachable!(),
            };
            // let multiplier = 2.;
            let resolution = (nodes.len().max(10) as f32 * multiplier) / (max - min);
            // let resolution = 1.;
            let num_buckets = ((max - min) * resolution).ceil() as i32;
            // dbg!(num_buckets, max - min, resolution, nodes.len(), multiplier);
            for i in 0..num_buckets {
                let boundary = (i as f32 / resolution) + min;
                let start_idx = d_pos.iter().take_while(|&&x| x < boundary).count();
                lut.push(start_idx + offset);
                // Use the next bucket's boundary so end_lut[i] is an upper bound
                // for any value that truncates to bucket i
                let next_boundary = ((i + 1) as f32 / resolution) + min;
                let end_idx = d_pos.iter().take_while(|&&x| x < next_boundary).count();
                end_lut.push(end_idx + offset);
            }
            lut.extend_from_slice(&end_lut);

            layers[layer_id] = Self::Leaf(Snn {
                lut: lut.into(),
                // end_lut: end_lut.into(),
                min: d_pos[0].floor(),
                resolution,
            });
            return;
        }

        // For internal nodes, use select_nth_unstable to partition around median
        let median_idx = nodes.len() / 2;
        nodes.select_nth_unstable_by_key(median_idx, |i| {
            i32::from_ne_bytes(atree.position(*i)[depth].to_ne_bytes())
        });

        // After select_nth_unstable, all elements left of median_idx have values <= pivot
        // We need to move elements equal to pivot to the right side for strict partitioning
        let split = atree.position(nodes[median_idx])[depth];
        let mut split_pos = median_idx;

        let mut i = 0;
        while i < split_pos {
            if atree.position(nodes[i])[depth] == split {
                // Move equal value to the end of left half and shrink left half
                split_pos -= 1;
                nodes.swap(i, split_pos);
                // Don't increment i, check the swapped element
            } else {
                i += 1;
            }
        }

        let slack = d_pos.len() - nodes.len();
        let (a_ids, b_ids) = nodes.split_at_mut(split_pos);
        let (a_dpos, b_dpos) = d_pos.split_at_mut(split_pos + slack / 2);

        let (a_id, b_id) = children(layer_id);

        // let depth = (depth + 1) % (D - 1);
        let depth = (depth + 1) % D;

        Layer::init(a_ids, a_dpos, layers, depth, a_id, atree, offset);
        Layer::init(
            b_ids,
            b_dpos,
            layers,
            depth,
            b_id,
            atree,
            offset + split_pos,
        );

        let node = Node { split };
        layers[layer_id] = Self::Node(node);
    }
}

#[inline(always)]
fn children(index: usize) -> (usize, usize) {
    (index * 2 + 1, index * 2 + 2)
}

impl<'a, const D: usize> ATree<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut line_lsh = ATree {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
            layers: vec![Layer::Node(Node { split: 0. }); embedding.positions.len()],
        };
        if !line_lsh.positions.is_empty() {
            line_lsh.update_positions(&embedding.positions, None);
        }
        line_lsh
    }
    pub fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        self.query_recursive(
            Point::new(pos, 0),
            0,
            0,
            radius.powi(2) as f32,
            radius,
            DVec::zero(),
            results,
        );
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn query_recursive(
        &self,
        pos: Point<D>,
        depth: usize,
        layer_id: usize,
        dim_radius_squared: f32,
        original_radius_squared: f64,
        mut distances: DVec<D>,
        results: &mut Vec<NodeId>,
    ) {
        let layer = &self.layers[layer_id];
        let own_pos = pos[depth];
        // let new_depth = (depth + 1) % (D - 1);
        let new_depth = (depth + 1) % D;
        match layer {
            Layer::Node(node) => {
                let (left, right) = children(layer_id);
                let (own, other) = if own_pos < node.split {
                    (left, right)
                } else {
                    (right, left)
                };
                let current_delta = distances[depth];
                let dist = (own_pos - node.split).powi(2);
                self.query_recursive(
                    pos,
                    new_depth,
                    own,
                    dim_radius_squared,
                    original_radius_squared,
                    distances,
                    results,
                );
                let reduced_radius = dim_radius_squared + current_delta - dist;
                distances[depth] = dist;
                if reduced_radius <= 0. {
                    return;
                }

                self.query_recursive(
                    pos,
                    new_depth,
                    other,
                    reduced_radius,
                    original_radius_squared,
                    distances,
                    results,
                );
            }
            Layer::Leaf(snn) => {
                // let depth = D - 1;
                let dim_diff_squared = distances[depth];
                let reduced_radius = (dim_radius_squared + dim_diff_squared).sqrt();
                // let reduced_radius = (dim_radius_squared).sqrt();
                self.snn(
                    pos,
                    depth,
                    reduced_radius,
                    original_radius_squared,
                    results,
                    snn,
                );
            }
        }
    }

    // #[inline(never)]
    fn snn(
        &self,
        pos: Point<D>,
        depth: usize,
        reduced_radius: f32,
        original_radius_squared: f64,
        results: &mut Vec<usize>,
        snn: &Snn,
    ) {
        let own_pos = pos[depth] - snn.min;
        let min = own_pos - reduced_radius;
        let max = own_pos + reduced_radius;
        if snn.lut.is_empty() {
            return;
        }
        let max_idx = snn.lut.len() / 2 - 1;
        let idx = ((min * snn.resolution) as usize).min(max_idx);
        let end_idx = ((max * snn.resolution) as usize).min(max_idx);
        let min_i = snn.lut[idx];
        let max_i = snn.lut[end_idx + snn.lut.len() / 2];

        // SAFETY: We need to allocate enough space upfront to allow us to write to the vector without checking if the size is valid
        results.reserve(max_i - min_i);
        let mut len = results.len();
        let half_radius_threshold = original_radius_squared as f32 / 2. + 1e-4;
        let radius_sq_f32 = original_radius_squared as f32;
        // dbg!(max_i - min_i);
        for i in min_i..max_i {
            let other_pos = self.positions_sorted[i];
            let is_in_radius = if D < 8 {
                pos.pos.distance_squared(&other_pos.pos) <= radius_sq_f32
            } else {
                pos.closer_than(&other_pos, half_radius_threshold)
            };
            unsafe { *results.get_unchecked_mut(len) = other_pos.node_id as usize };
            len += is_in_radius as usize;
        }
        unsafe { results.set_len(len) };
    }
}

impl<const D: usize> Query<D> for ATree<'_, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let radius = radius.powi(2);
        assert_eq!(self.positions.len(), self.node_ids.len());
        self.query_recursive(
            Point::new(pos, 0),
            0,
            0,
            radius as f32,
            radius,
            DVec::zero(),
            results,
        );
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
