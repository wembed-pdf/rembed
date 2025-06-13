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
    offset: i32,
    resolution: f64,
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

const RESOLUTION: usize = 1;

impl<const D: usize> Layer<D> {
    fn new(depth: usize, nodes: &[NodeId], positions: &[DVec<D>]) -> Self {
        if nodes.is_empty() {
            return Self::Empty;
        }
        if nodes.len() < 200 || depth == D - 1 {
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
                offset: min,
                resolution,
                lut,
                ids,
                d_pos,
                pos,
            };
            return Self::Snn(snn);
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
        // TODO: avoid numerical anihilation from adding small values to bigger number
        dim_radius_squared: f64,
        original_radius_squared: f64,
        results: &mut Vec<NodeId>,
    ) {
        let full_pos = self.position(index);
        let pos = self.position(index)[depth];
        match layer {
            Layer::Lsh(line_lsh) => {
                let bucket_index = (pos as f64 - line_lsh.offset as f64) * RESOLUTION as f64;
                let min_bucket =
                    (bucket_index - dim_radius_squared * RESOLUTION as f64).max(0.) as usize;
                let max_bucket = ((bucket_index + dim_radius_squared * RESOLUTION as f64).ceil()
                    as usize)
                    .min(line_lsh.buckets.len() - 1);
                for i in min_bucket..=max_bucket {
                    let diff = if (i as f64) < bucket_index {
                        (bucket_index - i as f64 - 1.).max(0.)
                    } else {
                        i as f64 - bucket_index
                    };
                    let diff = (diff * (RESOLUTION as f64).recip());
                    let new_dim_radius = dim_radius_squared - diff.powi(2);
                    if new_dim_radius > 0. {
                        let layer = &line_lsh.buckets[i];
                        self.query_recursive(
                            index,
                            depth + 1,
                            layer,
                            // (dim_radius.powi(2) - diff.powi(2))
                            //     .sqrt()
                            //     .min(dim_radius - diff.powi(2)),
                            new_dim_radius,
                            // dim_radius,
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
                let idx = (((pos - snn.offset as f32) * snn.resolution as f32).floor() as i32)
                    .min(snn.lut.len() as i32 - 1)
                    .max(0);
                let vec_idx = snn.lut[idx as usize];
                let query_radius = original_radius_squared as f32;
                // let mut checked = 0;
                // let mut found = 0;
                let mut min_i = vec_idx;
                let mut max_i = (vec_idx + 1).min(snn.pos.len() - 1);
                for i in vec_idx..(snn.d_pos.len()) {
                    let p = snn.d_pos[i];
                    if p > max {
                        break;
                    }
                    max_i = i;
                }
                for i in (0..vec_idx).rev() {
                    let p = snn.d_pos[i];
                    if p < min {
                        // if p < pos && (p - pos).powi(2) > dim_radius as f32 {
                        break;
                    }
                    min_i = i;
                }
                for i in min_i..=max_i {
                    // for i in 0..(snn.ids.len()) {
                    // checked += 1;

                    let other_pos = snn.pos[i];
                    if snn.ids[i] == index {
                        continue;
                    }
                    if full_pos.distance_squared(&other_pos) <= query_radius {
                        results.push(snn.ids[i]);
                        // if i < min_i || i > max_i {
                        //     println!("min: {}, max: {}", min_i, max_i);
                        //     println!("found at i {i}");
                        //     println!(
                        //         "pos:{}, dim_radius:{}, vec_idx: {vec_idx} idx:{idx}, offset: {}",
                        //         pos, dim_radius, snn.offset
                        //     );
                        //     println!("lut{:?}", &snn.lut);
                        //     println!("dpos: {:?}", &snn.d_pos);
                        // }
                        // found += 1;
                    }
                }
                // if dim_radius < 0.8 {
                //     dbg!(dim_radius, query_radius);
                // }
                // if found != 0 {
                //     dbg!(found, checked);
                // }
            }
            Layer::Empty => (),
        }
    }
}

impl<const D: usize> Query for LayeredLsh<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        // if self.weight(index) < 1. {
        return self.light_nn(index, radius * self.weight(index).powi(4));
        // return self.light_nn(index, radius);
        // }
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
