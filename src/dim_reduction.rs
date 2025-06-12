use ordered_float::NotNan;

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

#[derive(Clone)]
enum Layer {
    Lsh(LineLsh),
    Snn(Vec<NodeId>),
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

const RESOLUTION: usize = 10;

impl Layer {
    fn snn_mut(&mut self) -> &mut Vec<NodeId> {
        let Self::Snn(vec) = self else { unreachable!() };
        vec
    }

    fn new<const D: usize>(depth: usize, nodes: &[NodeId], positions: &[DVec<D>]) -> Self {
        if nodes.len() < 10 {
            let mut nodes = nodes.to_vec();
            nodes.sort_unstable_by(|x, y| {
                positions[*x][depth]
                    .partial_cmp(&positions[*y][depth])
                    .unwrap()
            });
            return Self::Snn(nodes);
        }

        let node_index = || {
            nodes
                .iter()
                .map(|x| (positions[*x][depth] * RESOLUTION as f32).floor() as i32)
        };
        let min = node_index().min().unwrap_or(0);
        let max = node_index().max().unwrap_or(0);

        let mut buckets = vec![Layer::Snn(Vec::new()); (-min) as usize + max as usize + 1];
        for (&node_id, bucket) in nodes.iter().zip(node_index()) {
            buckets[(bucket - min) as usize].snn_mut().push(node_id);
        }

        for bucket in &mut buckets {
            let nodes = std::mem::take(bucket.snn_mut());
            *bucket = Self::new(depth + 1, &nodes, positions);
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
                for i in 0..(line_lsh.buckets.len()) {
                    let diff = if (i as f32) < bucket_index {
                        (bucket_index - i as f32 - 1.).min(0.)
                    } else {
                        i as f32 - bucket_index
                    };
                    let diff = (diff / RESOLUTION as f32) as f64;
                    if diff < dim_radius {
                        let layer = &line_lsh.buckets[i];
                        self.query_recursive(
                            index,
                            depth + 1,
                            layer,
                            dim_radius - diff.powi(2),
                            original_radius,
                            results,
                        );
                    }
                }
            }
            Layer::Snn(items) => {
                let min = NotNan::new(pos - dim_radius as f32).unwrap();
                let max = NotNan::new(pos + dim_radius as f32).unwrap();
                let start = items
                    .binary_search_by_key(&min, |id| {
                        NotNan::new(self.position(*id)[depth]).unwrap()
                    })
                    .unwrap_or_else(|x| x);
                let end = items
                    .binary_search_by_key(&max, |id| {
                        NotNan::new(self.position(*id)[depth]).unwrap()
                    })
                    .unwrap_or_else(|x| x);
                let query_radius = original_radius.powi(2) as f32;
                if start >= end {
                    return;
                }
                for i in &items[start..end] {
                    if i != &index && full_pos.distance_squared(self.position(*i)) <= query_radius {
                        results.push(*i);
                    }
                }
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
