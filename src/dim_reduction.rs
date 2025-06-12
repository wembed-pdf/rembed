use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct LayeredLsh<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    layer: Layer,
}

#[derive(Clone)]
struct LineLsh {
    offset: i32,
    buckets: Vec<Layer>,
}

struct Snn {
    offset: i32,
    lut: Vec<usize>,
}

#[derive(Clone)]
enum Layer {
    Lsh(LineLsh),
    Snn(Vec<(NodeId, f32)>),
}

impl<'a, const D: usize> crate::query::Graph for LayeredLsh<'a, D> {
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
impl<'a, const D: usize> query::Position<D> for LayeredLsh<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
}
impl<'a, const D: usize> query::Update<D> for LayeredLsh<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        self.positions = postions.to_vec();
        let node_ids: Vec<_> = (0..postions.len()).collect();
        self.layer = Layer::new(0, &node_ids, &self.positions)
    }
}

const RESOLUTION: usize = 2;

impl Layer {
    fn new<const D: usize>(depth: usize, nodes: &[NodeId], positions: &[DVec<D>]) -> Self {
        if nodes.len() < 100 || depth == D - 1 {
            let mut nodes: Vec<_> = nodes.iter().map(|x| (*x, positions[*x][depth])).collect();
            nodes.sort_unstable_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap());
            return Self::Snn(nodes);
        }

        let node_index = || {
            nodes
                .iter()
                .map(|x| (positions[*x][depth] * RESOLUTION as f32).floor() as i32)
        };
        let min = node_index().min().unwrap_or(0);
        let max = node_index().max().unwrap_or(0);

        let mut temp_buckets = vec![vec![]; (-min) as usize + max as usize + 1];
        let mut buckets = Vec::with_capacity(temp_buckets.len());
        for (&node_id, bucket) in nodes.iter().zip(node_index()) {
            temp_buckets[(bucket - min) as usize].push(node_id);
        }

        for nodes in temp_buckets {
            buckets.push(Self::new(depth + 1, &nodes, positions));
        }
        Layer::Lsh(LineLsh {
            offset: min,
            buckets,
        })
    }
}
impl<'a, const D: usize> LayeredLsh<'a, D> {
    pub fn new(embedding: &Embedding<'a, D>) -> Self {
        let mut line_lsh = LayeredLsh {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            layer: Layer::Snn(Vec::new()),
        };
        line_lsh.update_positions(&embedding.positions);
        line_lsh
    }
    fn light_nn(&self, index: usize, radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        self.query_recursive(index, 0, &self.layer, radius, radius, &mut results);
        results
    }
    fn query_recursive(
        &self,
        index: usize,
        depth: usize,
        layer: &Layer,
        dim_radius: f64,
        original_radius: f64,
        results: &mut Vec<NodeId>,
    ) {
        let full_pos = self.position(index);
        let pos = self.position(index)[depth];
        match layer {
            Layer::Lsh(line_lsh) => {
                let bucket_index = (pos - line_lsh.offset as f32) * RESOLUTION as f32;
                let min_bucket =
                    (bucket_index - dim_radius as f32 * RESOLUTION as f32).max(0.) as usize;
                let max_bucket = ((bucket_index + dim_radius as f32 * RESOLUTION as f32) as usize)
                    .min(line_lsh.buckets.len() - 1);
                for i in min_bucket..=max_bucket {
                    let diff = if (i as f32) < bucket_index {
                        (bucket_index - i as f32 - 1.).min(0.)
                    } else {
                        i as f32 - bucket_index
                    };
                    let diff = (diff * (RESOLUTION as f32).recip()) as f64;
                    if diff < dim_radius {
                        let layer = &line_lsh.buckets[i];
                        self.query_recursive(
                            index,
                            depth + 1,
                            layer,
                            dim_radius - diff.powi(2),
                            // dim_radius - diff.powi(2),
                            original_radius,
                            results,
                        );
                    }
                }
            }
            Layer::Snn(items) => {
                let min = pos - dim_radius as f32;
                let max = pos + dim_radius as f32;
                let mid = items
                    .binary_search_by(|(_, x)| x.partial_cmp(&pos).unwrap())
                    .unwrap_or_else(|x| x);
                // let start = items
                //     .binary_search_by(|(_, x)| x.partial_cmp(&min).unwrap())
                //     .unwrap_or_else(|x| x);
                // let end = items
                //     .binary_search_by(|(_, x)| x.partial_cmp(&max).unwrap())
                //     .unwrap_or_else(|x| x);
                let query_radius = original_radius.powi(2) as f32;
                // if start >= end {
                //     return;
                // }
                let mut checked = 0;
                let mut found = 0;
                for (i, p) in &items[mid..] {
                    if p > &max {
                        break;
                    }
                    checked += 1;
                    if i != &index && full_pos.distance_squared(self.position(*i)) <= query_radius {
                        results.push(*i);
                        found += 1;
                    }
                }
                for (i, p) in items[..mid].iter().rev() {
                    if p < &min {
                        break;
                    }
                    checked += 1;
                    if i != &index && full_pos.distance_squared(self.position(*i)) <= query_radius {
                        results.push(*i);
                        found += 1;
                    }
                }
                // if dim_radius < 0.8 {
                //     dbg!(dim_radius, query_radius);
                // }
                // if found != 0 {
                //     dbg!(found, checked);
                // }
            }
        }
    }
}

impl<const D: usize> Query for LayeredLsh<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        if self.weight(index) < 1. {
            return self.light_nn(index, radius);
        }
        let mut output = Vec::new();
        let graph = self.graph;
        let own_weight = self.weight(index);
        let own_position = self.position(index);

        for (i, (node, position)) in graph.nodes.iter().zip(self.positions.iter()).enumerate() {
            let weight = own_weight * node.weight;
            let distance = own_position.distance_squared(position);
            if (distance as f64) < weight.powi(2) * radius {
                output.push(i);
            }
        }
        output
    }
}
impl<const D: usize> SpatialIndex<D> for LayeredLsh<'_, D> {
    fn name(&self) -> String {
        String::from("line-lsh")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("embedding.rs")
    }
}

impl<'a, const D: usize> query::Embedder<D> for LayeredLsh<'a, D> {}

#[cfg(test)]
mod test {
    #[test]
    fn simple() {}
}
