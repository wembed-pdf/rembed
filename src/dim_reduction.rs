use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone)]
pub struct LayeredLsh<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    layer: Layer<D>,
}

#[derive(Clone)]
struct LineLsh<const D: usize> {
    offset: i32,
    buckets: Vec<Layer<D>>,
}

#[derive(Clone, Default)]
struct Snn<const D: usize> {
    offset: f32,
    resolution: f32,
    lut: Vec<usize>,
    ids: Vec<NodeId>,
    d_pos: Vec<f32>,
    pos: Vec<DVec<D>>,
}

#[derive(Clone)]
enum Layer<const D: usize> {
    Lsh(LineLsh<D>),
    Snn(Snn<D>),
    Empty,
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

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}
impl<'a, const D: usize> query::Update<D> for LayeredLsh<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        self.positions = postions.to_vec();
        let node_ids: Vec<_> = (0..postions.len()).collect();
        self.layer = Layer::new(0, &node_ids, &self.positions)
    }
}

impl<const D: usize> Layer<D> {
    fn new(depth: usize, nodes: &[NodeId], positions: &[DVec<D>]) -> Self {
        if nodes.is_empty() {
            return Self::Empty;
        }
        if nodes.len() < 100 || depth == D - 1 {
            let mut nodes: Vec<_> = nodes.iter().map(|x| (*x, positions[*x][depth])).collect();
            nodes.sort_unstable_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap());
            let ids: Vec<_> = nodes.iter().map(|(x, _)| *x).collect();
            let d_pos: Vec<_> = nodes.iter().map(|(_, p)| *p).collect();
            let pos = ids.iter().map(|i| positions[*i]).collect();

            let mut lut = vec![];
            // let mut pos_idx = 0;
            let min = d_pos[0].floor() as i32;
            let max = d_pos.last().unwrap().ceil() as i32;
            let resolution = 50. / (max - min) as f64;
            for i in 0..(((max - min) as f64 * resolution) as i32) {
                let pos_idx = d_pos
                    .iter()
                    .take_while(|&&x| x < ((i as f64 / resolution) + min as f64) as f32)
                    .count();
                lut.push(pos_idx);
            }

            let snn = Snn {
                offset: min as f32,
                resolution: resolution as f32,
                lut,
                ids,
                d_pos,
                pos,
            };
            return Self::Snn(snn);
        }

        let node_index = || nodes.iter().map(|x| (positions[*x][depth]).floor() as i32);
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
            layer: Layer::Snn(Default::default()),
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
        layer: &Layer<D>,
        dim_radius_squared: f64,
        original_radius_squared: f64,
        results: &mut Vec<NodeId>,
    ) {
        let full_pos = self.position(index);
        let pos = self.position(index)[depth];
        // TODO: Increase resolution for subsequent dimensions based on estimated radius reduction / switch to different metric
        match layer {
            Layer::Lsh(line_lsh) => {
                let offset_position = pos as f64 - line_lsh.offset as f64;
                let min_bucket = (offset_position + (-(dim_radius_squared)).floor()) as usize;
                let max_bucket = ((offset_position + dim_radius_squared) as usize + 1)
                    .min(line_lsh.buckets.len() - 1);
                for i in min_bucket..=max_bucket {
                    let diff = if (i as f64) < offset_position {
                        (offset_position - i as f64 - 1.).max(0.)
                    } else {
                        i as f64 - offset_position
                    };
                    let new_dim_radius = dim_radius_squared - diff.powi(2);
                    if new_dim_radius > 0. {
                        let layer = &line_lsh.buckets[i];
                        self.query_recursive(
                            index,
                            depth + 1,
                            layer,
                            new_dim_radius,
                            original_radius_squared,
                            results,
                        );
                    }
                }
            }
            Layer::Snn(snn) => {
                let radius_sqrt = (dim_radius_squared as f32).sqrt();
                let min = pos - radius_sqrt;
                let max = pos + radius_sqrt;
                let idx = (((min - snn.offset) * snn.resolution) as usize).min(snn.lut.len() - 1);
                let min_i = snn.lut[idx];

                // for i in 0..(snn.ids.len()) {
                for i in min_i..(snn.d_pos.len()) {
                    let p = snn.d_pos[i];
                    if p > max {
                        break;
                    }
                    if snn.ids[i] == index {
                        continue;
                    }
                    let other_pos = snn.pos[i];
                    // if self.weight(snn.ids[i]) > self.weight(index) {
                    //     continue;
                    // }
                    if full_pos.distance_squared(&other_pos) <= original_radius_squared as f32 {
                        results.push(snn.ids[i]);
                    }
                }
            }
            Layer::Empty => (),
        }
    }
}

impl<const D: usize> Query for LayeredLsh<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        self.light_nn(index, radius * self.weight(index).powi(4))
    }
}
impl<const D: usize> SpatialIndex<D> for LayeredLsh<'_, D> {
    fn name(&self) -> String {
        String::from("line-lsh")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("dim_reduction.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for LayeredLsh<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn simple() {}
}
