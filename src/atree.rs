use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Position, SpatialIndex, Update},
};

const LEAFSIZE: usize = 150;

#[derive(Clone)]
pub struct ATree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub positions_sorted: Vec<DVec<D>>,
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

        let num_leafs = node_ids.len().max(LEAFSIZE).ilog2() - LEAFSIZE.ilog2();
        // let mut d_pos = vec![0.; node_ids.len() + num_leafs * 4];
        let mut d_pos =
            vec![f32::INFINITY; node_ids.len() + (1usize << num_leafs as usize) * 2 * 4];
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
            0,
        );
        self.layers = layers;
        self.node_ids = node_ids;
        self.positions_sorted = self.node_ids.iter().map(|id| *self.position(*id)).collect();
        for _ in 0..4 {
            d_pos.push(f32::INFINITY);
            self.positions_sorted.push(DVec::splat(f32::INFINITY));
        }
        self.d_pos = d_pos;
        // println!("dpos: {:?}", self.d_pos);
    }
}

#[derive(Clone, Debug)]
struct Node {
    split: f32,
}

#[derive(Clone, Debug, Default)]
struct Snn {
    len: usize,
    dpos_offset: usize,
    end: usize,
    lut: Vec<usize>,
    end_lut: Vec<usize>,
    min: f32,
    resolution: f32,
}

#[derive(Clone, Debug)]
enum Layer {
    Node(Node),
    Leaf(Snn),
}
// IDEAS: use end_lut
// approximate square root
// stop reducing radius after first couple of tries

impl Layer {
    fn init<const D: usize>(
        nodes: &mut [NodeId],
        d_pos: &mut [f32],
        layers: &mut [Layer],
        depth: usize,
        layer_id: usize,
        atree: &ATree<D>,
        offset: usize,
        dpos_offset: usize,
    ) {
        // For leaf nodes, we need full sorting for the lookup table
        if nodes.len() <= { LEAFSIZE } {
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
            let resolution = 100. / (max - min);
            let num_buckets = ((max - min) * resolution) as i32;
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

            layers[layer_id] = Self::Leaf(Snn {
                len: nodes.len(),
                dpos_offset: dpos_offset - offset,
                end: nodes.len() + offset,
                lut,
                end_lut,
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

        let depth = (depth + 1) % D;

        Layer::init(
            a_ids,
            a_dpos,
            layers,
            depth,
            a_id,
            atree,
            offset,
            dpos_offset,
        );
        Layer::init(
            b_ids,
            b_dpos,
            layers,
            depth,
            b_id,
            atree,
            offset + split_pos,
            dpos_offset + split_pos + slack / 2,
        );

        let node = Node { split };
        layers[layer_id] = Self::Node(node);
    }
}
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
            pos,
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
        pos: DVec<D>,
        depth: usize,
        layer_id: usize,
        dim_radius_squared: f32,
        original_radius_squared: f64,
        mut distances: DVec<D>,
        results: &mut Vec<NodeId>,
    ) {
        let layer = &self.layers[layer_id];
        let own_pos = pos[depth];
        let new_depth = (depth + 1) % D;
        match layer {
            Layer::Node(node) => {
                let (left, right) = children(layer_id);
                let (own, other) = if own_pos < node.split {
                    (left, right)
                } else {
                    (right, left)
                };
                self.query_recursive(
                    pos,
                    new_depth,
                    own,
                    dim_radius_squared,
                    original_radius_squared,
                    distances,
                    results,
                );
                let current_delta = distances[depth];
                let dist = (own_pos - node.split).powi(2);
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
                self.snn(
                    pos,
                    depth,
                    dim_radius_squared,
                    original_radius_squared,
                    distances,
                    results,
                    own_pos,
                    snn,
                );
            }
        }
    }

    #[inline(never)]
    fn snn(
        &self,
        pos: DVec<D>,
        depth: usize,
        dim_radius_squared: f32,
        original_radius_squared: f64,
        distances: DVec<D>,
        results: &mut Vec<usize>,
        own_pos: f32,
        snn: &Snn,
    ) {
        let dim_diff_squared = distances[depth];
        let radius_sqrt = (dim_radius_squared + dim_diff_squared).sqrt();
        let min = own_pos - radius_sqrt;
        let max = own_pos + radius_sqrt;
        let idx = (((min - snn.min) * snn.resolution) as usize).min(snn.lut.len().max(1) - 1);
        let end_idx = (((max - snn.min) * snn.resolution) as usize).min(snn.end_lut.len() - 1);
        if snn.lut.is_empty() {
            return;
        }
        let min_i = snn.lut[idx];
        let max_i = snn.end_lut[end_idx];

        // let mut i = min_i;
        // loop {
        // results.reserve(4);
        // for i in min_i.. {
        for i in min_i..max_i {
            // let p = self.d_pos[i + snn.dpos_offset];
            // dbg!(p);
            // let b = p > max;
            // if p > max {
            //     if i > max_i {
            //         panic!(
            //             "min i: {min_i}, {i} > max_i {max_i}, own_pos: {own_pos}, max: {max} d_pos: {:?}",
            //             &self.d_pos[max_i..i]
            //         );
            //     }
            //     break;
            // }
            // let unroll = 2;
            // for j in 0..unroll {
            //     // let i = (i + j).min(self.node_ids.len() - 1);
            //     let i = i + j;
            let other_pos = self.positions_sorted[i];
            if pos.distance_squared(&other_pos) <= original_radius_squared as f32 {
                results.push(self.node_ids[i]);
            }
            // }
            // // i += unroll;
            // if b {
            //     break;
            // }
        }
    }
}

impl<const D: usize> Query<D> for ATree<'_, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let radius = radius.powi(2);
        self.query_recursive(pos, 0, 0, radius as f32, radius, DVec::zero(), results);
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

#[cfg(test)]
mod test {
    #[test]
    fn simple() {}
}
