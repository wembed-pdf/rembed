use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct ATree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
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
        let node_ids: Vec<_> = (0..postions.len()).collect();
        self.layer = Layer::new(node_ids.as_slice(), 0, self);
    }
}

#[derive(Clone)]
struct Node {
    split: f32,
    a: Box<Layer>,
    b: Box<Layer>,
}

#[derive(Clone)]
enum Layer {
    Node(Node),
    Leaf(Vec<NodeId>),
}

impl Layer {
    fn new<const D: usize>(nodes: &[NodeId], depth: usize, atree: &ATree<D>) -> Self {
        if nodes.len() <= 100 || depth == D - 1 {
            return Self::Leaf(nodes.to_vec());
        }

        let mut sorted: Vec<_> = nodes
            .iter()
            .map(|&i| (i, atree.positions[i][depth]))
            .collect();
        sorted.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        let mut split_pos = sorted.len() / 2;

        let split = sorted[split_pos].1;
        while split_pos != 0 && sorted[split_pos - 1].1 == split {
            split_pos -= 1;
        }

        let a_ids: Vec<_> = sorted[..split_pos].iter().map(|(id, _)| *id).collect();
        let b_ids: Vec<_> = sorted[split_pos..].iter().map(|(id, _)| *id).collect();

        let a = Layer::new(a_ids.as_slice(), depth + 1, atree);
        let b = Layer::new(b_ids.as_slice(), depth + 1, atree);

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
            layer: Layer::Leaf(Vec::new()),
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
            (radius * self.weight(index).powi(2)).powi(2) as f32,
            radius,
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
        original_radius: f64,
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
                let dist_squared = (own_pos - node.split).powi(2);
                if dist_squared < dim_radius_squared {
                    // let reduced_radius = dim_radius_squared;
                    let reduced_radius = dim_radius_squared - dist_squared;
                    self.query_recursive(
                        index,
                        depth + 1,
                        other,
                        reduced_radius,
                        original_radius,
                        results,
                    );
                }
                self.query_recursive(
                    index,
                    depth + 1,
                    own,
                    dim_radius_squared,
                    original_radius,
                    results,
                );
            }
            Layer::Leaf(items) => {
                for &i in items {
                    let other_pos = self.position(i);
                    let other_weight = self.weight(i);
                    let distance_squared = other_pos.distance_squared(self.position(index)) as f64;
                    if distance_squared
                        < (self.weight(index) * other_weight * original_radius).powi(2)
                    {
                        results.push(i);
                    }
                }
            }
        }
    }
}

impl<const D: usize> Query for ATree<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        self.light_nn(index, radius)
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
