use std::{collections::HashMap, ops::Mul};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
// User smaller number to store in hashmap
type SpatialMap<const D: usize> = HashMap<[i32; D], Node<D>>;

const DIM_CHUNK_SIZE: usize = 2;

#[derive(Clone)]
pub enum Node<const D: usize> {
    Map(SpatialMap<D>, usize),
    Leaf(Vec<NodeId>),
}

impl<const D: usize> Node<D> {
    fn map(&self) -> Option<&SpatialMap<D>> {
        let Node::Map(spatial_map, _) = self else {
            return None;
        };
        Some(spatial_map)
    }
    fn map_mut(&mut self) -> &mut SpatialMap<D> {
        let Node::Map(spatial_map, _) = self else {
            panic!();
        };
        spatial_map
    }
    fn leaf(&self) -> Option<&Vec<NodeId>> {
        let Node::Leaf(leaf) = self else {
            return None;
        };
        Some(leaf)
    }
    fn leaf_mut(&mut self) -> &mut Vec<NodeId> {
        let Node::Leaf(leaf) = self else {
            panic!();
        };
        leaf
    }
}

#[derive(Clone)]
pub struct Lsh<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub weight_threshold: f64,
    pub map: SpatialMap<D>,
}

impl<'a, const D: usize> crate::query::Graph for Lsh<'a, D> {
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
impl<'a, const D: usize> query::Position<D> for Lsh<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}
impl<'a, const D: usize> query::Update<D> for Lsh<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>], _: Option<f64>) {
        self.positions = postions.to_vec();
        for (id, pos) in self.positions.iter().enumerate() {
            for rounded_pos in point(pos, 0, DIM_CHUNK_SIZE) {
                // dbg!(&rounded_pos);
                let map = self
                    .map
                    .entry(rounded_pos)
                    .or_insert(Node::Map(SpatialMap::default(), 1))
                    .map_mut();
                for rounded_pos in point(pos, DIM_CHUNK_SIZE, DIM_CHUNK_SIZE) {
                    // dbg!(&rounded_pos);
                    let inner_slot = map.entry(rounded_pos).or_insert(Node::Leaf(Vec::new()));
                    inner_slot.leaf_mut().push(id);
                }
            }
        }
    }
}

impl<const D: usize> Query for Lsh<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<NodeId>) {
        if self.weight(index) >= self.weight_threshold {
            self.heavy_nn(index, radius, results)
        } else {
            self.light_nn(index, results)
        }
    }
}
impl<const D: usize> SpatialIndex<D> for Lsh<'_, D> {
    fn name(&self) -> String {
        format!("lsh-{}", DIM_CHUNK_SIZE)
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("lsh.rs")
    }
}

impl<'a, const D: usize> Lsh<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let Embedding { positions, graph } = embedding;
        let mut new = Self {
            positions: positions.clone(),
            graph,
            weight_threshold: 1.,
            // TODO: use different hash function
            map: HashMap::with_capacity(positions.len()),
        };
        new.update_positions(&positions, None);
        new
    }

    fn heavy_nn(&self, index: usize, radius: f64, output: &mut Vec<NodeId>) {
        let graph = self.graph;
        let positions = &self.positions;
        let own_weight = graph.nodes[index].weight;
        let own_position = positions[index];

        for (i, (node, position)) in graph.nodes.iter().zip(positions.iter()).enumerate() {
            if own_weight < node.weight {
                continue;
            }
            let weight = own_weight * node.weight;
            let distance = own_position.distance_squared(position);
            if (distance as f64) < weight.powi(2) * radius {
                output.push(i);
            }
        }
    }

    fn light_nn(&self, index: usize, neighbors: &mut Vec<NodeId>) {
        let spatial_maps =
            nbig_box(self.position(index), 0, DIM_CHUNK_SIZE).flat_map(|x| self.map.get(&x));
        let own_pos = self.position(index);

        for spatial_map in spatial_maps {
            let spatial_map = spatial_map.map().unwrap();
            if spatial_map.is_empty() {
                continue;
            }

            if nbig_box(self.position(index), DIM_CHUNK_SIZE, DIM_CHUNK_SIZE).count()
                > spatial_map.len() * 20
            {
                for list in spatial_map.values() {
                    for node in list.leaf().unwrap() {
                        if own_pos.distance_squared(self.position(*node)) <= 1. {
                            neighbors.push(*node);
                        }
                    }
                }
            }
            let lists = nbig_box(self.position(index), DIM_CHUNK_SIZE, DIM_CHUNK_SIZE)
                .flat_map(|x| spatial_map.get(&x));
            for list in lists {
                for node in list.leaf().unwrap() {
                    if own_pos.distance_squared(self.position(*node)) <= 1. {
                        neighbors.push(*node);
                    }
                }
            }
        }
    }
}

fn point<const D: usize>(
    pos: &DVec<D>,
    dim_offset: usize,
    dim_count: usize,
) -> impl Iterator<Item = [i32; D]> {
    let vec = DVec::units(((1 << dim_count) - 1) << dim_offset);
    std::iter::once(pos.mul(vec).map(|x| x.round()).to_int_array())
}

fn nbig_box<const D: usize>(
    pos: &DVec<D>,
    dim_offset: usize,
    dim_count: usize,
) -> impl Iterator<Item = [i32; D]> {
    let vec = DVec::units(((1 << dim_count) - 1) << dim_offset);
    let pos = *pos * vec;
    let rounded = pos.map(|x| x.round()).to_int_array();
    let total = 3usize.pow(dim_count as u32);
    (0..total).map(move |i| {
        let mut result = rounded;
        let mut n = i;
        // let mut skip = false;
        for d in 0..dim_count {
            let offset = (n % 3) as i32 - 1;
            n /= 3;
            // if d == 0 && offset == -1 {
            //     skip = true;
            // }
            if dim_offset + d < D {
                result[dim_offset + d] += offset;
            }
        }
        result
        // (!skip).then(|| result)
    })
}

impl<'a, const D: usize> query::Embedder<'a, D> for Lsh<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
