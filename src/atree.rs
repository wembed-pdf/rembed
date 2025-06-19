use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

const LEAFSIZE: usize = 500;

#[derive(Clone)]
pub struct ATree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub positions_sorted: Vec<DVec<D>>,
    pub node_ids: Vec<usize>,
    pub d_pos: Vec<f32>,
    pub graph: &'a crate::graph::Graph,
    layer: Layer,
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
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        self.positions = postions.to_vec();
        let mut node_ids: Vec<_> = (0..postions.len()).collect();
        self.layer = Layer::new(node_ids.as_mut_slice(), 0, self, 0);
        self.node_ids = node_ids;
        let d = (self.positions.len() / LEAFSIZE).ilog2() as usize % D;
        self.positions_sorted = self.node_ids.iter().map(|id| *self.position(*id)).collect();
        println!("creating d_pos array with depth: {d}");
        self.d_pos = self
            .node_ids
            .iter()
            .map(|id| self.position(*id)[d])
            .collect();
    }
}

#[derive(Clone, Debug)]
struct Node {
    split: f32,
    a: Box<Layer>,
    b: Box<Layer>,
}

#[derive(Clone, Debug)]
struct Snn {
    offset: usize,
    len: usize,
}

#[derive(Clone, Debug)]
enum Layer {
    Node(Node),
    Leaf(Snn),
}

impl Layer {
    fn new<const D: usize>(
        nodes: &mut [NodeId],
        depth: usize,
        atree: &ATree<D>,
        offset: usize,
    ) -> Self {
        nodes.sort_unstable_by_key(|i| i32::from_ne_bytes(atree.position(*i)[depth].to_ne_bytes()));

        if nodes.len() <= LEAFSIZE {
            return Self::Leaf(Snn {
                offset,
                len: nodes.len(),
            });
        }

        let mut split_pos = nodes.len() / 2;

        let split = atree.position(nodes[split_pos])[depth];
        while split_pos != 0 && atree.position(nodes[split_pos - 1])[depth] == split {
            split_pos -= 1;
        }

        let (a_ids, b_ids) = nodes.split_at_mut(split_pos);

        let a = Layer::new(a_ids, (depth + 1) % D, atree, offset);
        let b = Layer::new(b_ids, (depth + 1) % D, atree, offset + split_pos);

        let node = Node {
            split,
            a: Box::new(a),
            b: Box::new(b),
        };
        Self::Node(node)
    }
}

impl<'a, const D: usize> ATree<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut line_lsh = ATree {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            layer: Layer::Leaf(Snn { offset: 0, len: 0 }),
            positions_sorted: Vec::new(),
            node_ids: Vec::new(),
            d_pos: Vec::new(),
        };
        line_lsh.update_positions(&embedding.positions);
        line_lsh
    }
    fn light_nn(&self, index: usize, radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        self.query_recursive(
            index,
            0,
            &self.layer,
            radius as f32,
            radius,
            DVec::zero(),
            &mut results,
        );
        results
    }
    fn query_recursive(
        &self,
        index: usize,
        depth: usize,
        layer: &Layer,
        dim_radius_squared: f32,
        original_radius_squared: f64,
        mut distances: DVec<D>,
        results: &mut Vec<NodeId>,
    ) {
        let own_pos = self.position(index)[depth];
        match layer {
            Layer::Node(node) => {
                let (own, other) = if own_pos < node.split {
                    (&node.a, &node.b)
                } else {
                    (&node.b, &node.a)
                };
                self.query_recursive(
                    index,
                    (depth + 1) % D,
                    own,
                    dim_radius_squared,
                    original_radius_squared,
                    distances,
                    results,
                );
                let dist = own_pos - node.split;
                let dist_squared = dist.powi(2);
                if dist_squared < original_radius_squared as f32 {
                    let d_2 = dist - distances[depth];
                    let x = 2. * distances[depth] * d_2 + d_2.powi(2);
                    let mut reduced_radius = dim_radius_squared;
                    distances[depth] += dist;
                    reduced_radius -= x;
                    if reduced_radius <= 0. {
                        return;
                    }

                    self.query_recursive(
                        index,
                        (depth + 1) % D,
                        other,
                        reduced_radius,
                        original_radius_squared,
                        distances,
                        results,
                    );
                }
            }
            Layer::Leaf(snn) => {
                let dim_diff_squared = distances[depth].powi(2);
                let radius_sqrt = (dim_radius_squared as f32 + dim_diff_squared).sqrt();
                // dbg!(dim_diff_squared, radius_sqrt);
                let min = own_pos - radius_sqrt;
                let max = own_pos + radius_sqrt;
                let min_i = self.d_pos[snn.offset..(snn.offset + snn.len)]
                    .binary_search_by(|a| min.partial_cmp(a).unwrap())
                    .unwrap_or(0);

                // for i in 0..(snn.ids.len()) {
                for i in (snn.offset)..(snn.offset + snn.len) {
                    // for i in (min_i + snn.offset)..(snn.offset + snn.len) {
                    let p = self.d_pos[i];
                    // if p > max {
                    //     break;
                    // }
                    if self.node_ids[i] == index {
                        continue;
                    }
                    let other_pos = self.positions_sorted[i];
                    // if self.weight(snn.ids[i]) > self.weight(index) {
                    //     continue;
                    // }
                    if self.position(index).distance_squared(&other_pos)
                        <= original_radius_squared as f32
                    {
                        if p < min || p > max {
                            println!("found {p} which is not in bounds {min}..{max}");
                            println!("own_pos: {:?}", self.position(index));
                            println!("other_pos: {other_pos:?}");
                            println!("depth: {depth}");
                            panic!();
                        }
                        results.push(self.node_ids[i]);
                    }
                }
                // for i in snn.offset..(snn.offset + snn.len) {
                //     let other_pos = self.positions_sorted[i];
                //     let distance_squared = other_pos.distance_squared(self.position(index)) as f64;
                //     if distance_squared < original_radius_squared {
                //         results.push(self.node_ids[i]);
                //     }
                // }
            }
        }
    }
}

impl<const D: usize> Query for ATree<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        self.light_nn(index, (radius * self.weight(index).powi(2)).powi(2))
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
