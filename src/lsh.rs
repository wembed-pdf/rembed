use std::{
    collections::{HashMap, HashSet},
    ops::Mul,
};

use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, Update},
};
// User smaller number to store in hashmap
type SpatialMap<const D: usize> = HashMap<[i8; D], Node<D>>;

const DIM_CHUNK_SIZE: usize = 2;

#[derive(Clone)]
pub enum Node<const D: usize> {
    Map(SpatialMap<D>, usize),
    Leaf(Vec<NodeId>, [Vec<NodeId>; D]),
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
    fn level(&self) -> usize {
        match self {
            Node::Map(_hash_map, level) => *level,
            Node::Leaf(_items, _) => D,
        }
    }
    fn leaf(&self) -> Option<&Vec<NodeId>> {
        let Node::Leaf(leaf, _) = self else {
            return None;
        };
        Some(leaf)
    }
    fn leaf_mut(&mut self) -> &mut Vec<NodeId> {
        let Node::Leaf(leaf, _) = self else {
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
    pub map: SpatialMap<DIM_CHUNK_SIZE>,
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
}
impl<'a, const D: usize> query::Update<D> for Lsh<'a, D> {
    fn update_positions(&mut self, postions: &[DVec<D>]) {
        self.positions = postions.to_vec();
        self.map.clear();
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
                    let inner_slot = map
                        .entry(rounded_pos)
                        .or_insert(Node::Leaf(Vec::new(), std::array::from_fn(|_| Vec::new())));
                    inner_slot.leaf_mut().push(id);
                }
            }
        }
        for map in self.map.values_mut() {
            for list in map.map_mut().values_mut() {
                let Node::Leaf(list, sorted) = list else {
                    panic!()
                };
                let mut sort_list = list.clone();
                for d in 0..sorted.len() {
                    sort_list.sort_by_key(|&x| (self.positions[x][d] * 100000000.) as i64);
                    sorted[d] = sort_list.clone();
                }
            }
        }
    }
}

impl<const D: usize> Query for Lsh<'_, D> {
    fn nearest_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        if self.weight(index) >= self.weight_threshold {
            self.heavy_nn(index, radius)
        } else {
            self.light_nn(index)
        }
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
        new.update_positions(&positions);
        new
    }

    fn heavy_nn(&self, index: usize, radius: f64) -> Vec<usize> {
        let mut output = Vec::new();
        let graph = self.graph;
        let positions = &self.positions;
        let own_weight = graph.nodes[index].weight;
        let own_position = positions[index];

        for (i, (node, position)) in graph.nodes.iter().zip(positions.iter()).enumerate() {
            let weight = own_weight * node.weight;
            let distance = own_position.distance_squared(position);
            if (distance as f64) < weight.powi(2) * radius {
                output.push(i);
            }
        }
        output
    }

    fn light_nn(&self, index: usize) -> Vec<usize> {
        // let mut neighbors = HashSet::with_capacity(1000);
        let mut neighbors = Vec::with_capacity(100);
        let spatial_maps =
            nbig_box(self.position(index), 0, DIM_CHUNK_SIZE).flat_map(|x| self.map.get(&x));
        // let spatial_maps: Vec<_> = spatial_maps.collect();
        // dbg!(spatial_maps.len());
        let own_pos = self.position(index);
        // dbg!(own_pos);

        for spatial_map in spatial_maps {
            let spatial_map = spatial_map.map().unwrap();
            if spatial_map.is_empty() {
                continue;
            }

            // if nbig_box(self.position(index), DIM_CHUNK_SIZE, DIM_CHUNK_SIZE).count()
            //     > spatial_map.len() * 20
            // {
            //     // println!("short circuting second level");
            //     for list in spatial_map.values() {
            //         for node in list.leaf().unwrap() {
            //             if own_pos.distance_squared(self.position(*node)) <= 1. {
            //                 // neighbors.insert(*node);
            //                 neighbors.push(*node);
            //             }
            //         }
            //     }
            // }
            let lists = nbig_box(self.position(index), DIM_CHUNK_SIZE, DIM_CHUNK_SIZE)
                .flat_map(|x| spatial_map.get(&x));
            // println!("new_list");
            for list in lists {
                // dbg!(list.leaf());
                for node in list.leaf().unwrap() {
                    // if dbg!(dbg!(own_pos).distance_squared(dbg!(self.position(*node)))) <= 1. {
                    if node == &index {
                        continue;
                    }
                    if own_pos.distance_squared(self.position(*node)) <= 1. {
                        // neighbors.insert(*node);
                        neighbors.push(*node);
                    }
                }
            }
        }
        neighbors.into_iter().collect()
    }
}

//// TODO:
/// For each bucket, sort remaining elements on 1d line and use the distance from the center to narrow the query radius for the snn line. Potentially repeat with more dimension by using buckets.
/// Track the distance to all of the buckets (but the center bucket) and reduce the query radius on the sorted line accordingly. Do this for multiple dimension reductions

fn point<const D: usize, const N: usize>(
    pos: &DVec<D>,
    dim_offset: usize,
    dim_count: usize,
) -> impl Iterator<Item = [i8; N]> {
    let vec = DVec::units(((1 << dim_count) - 1) << dim_offset);
    let vec_offset = if N == D { 0 } else { dim_offset };
    std::iter::once(
        pos.mul(vec)
            .map(|x| x.floor())
            .slice(vec_offset)
            .to_int_array(),
    )
}
fn nbox<const D: usize>(
    pos: &DVec<D>,
    dim_offset: usize,
    dim_count: usize,
) -> impl Iterator<Item = [i8; D]> {
    let vec = DVec::units(((1 << dim_count) - 1) << dim_offset);
    (0..(1 << dim_count)).map(move |mask| round_to_dimensions(&(*pos * vec), mask << dim_offset))
}
fn round_to_dimensions<const D: usize>(pos: &DVec<D>, mask: usize) -> [i8; D] {
    let mut unit = DVec::zero();
    for i in 0..D {
        if mask & 1 << i != 0 {
            unit += DVec::unit(i);
        }
    }
    (*pos + unit).map(|x| x.floor()).to_int_array()
}

fn nbig_box<const D: usize, const N: usize>(
    pos: &DVec<D>,
    dim_offset: usize,
    dim_count: usize,
) -> impl Iterator<Item = [i8; N]> {
    let vec = DVec::units(((1 << dim_count) - 1) << dim_offset);
    let pos = *pos * vec;
    let rounded = pos.map(|x| x.floor());
    let diff = pos - rounded;
    let total = 3usize.pow(dim_count as u32);
    let vec_offset = if N == D { 0 } else { dim_offset };
    (0..total).flat_map(move |i| {
        let mut result = rounded.slice(vec_offset).to_int_array();
        let mut n = i;
        let mut dist = 0.;
        for d in dim_offset..(dim_offset + dim_count) {
            let offset = (n % 3) as i8 - 1;
            n /= 3;
            if offset == 1 {
                dist += (1. - diff[d]).powi(2);
            }
            result[d - vec_offset] += offset;
        }
        if dist < 1. { Some(result) } else { None }
    })
}

impl<'a, const D: usize> query::Embedder<D> for Lsh<'a, D> {}

#[cfg(test)]
mod test {
    use crate::graph::{Graph, Node};

    use super::*;

    #[test]
    fn should_intersect() {
        return;
        let p1 = point([
            1.04259, 1.55822, 4.6893, 3.99121, 2.93722, 4.016, 1.45376, 3.72547,
        ]);
        let p2 = point([
            1.33247, 1.38505, 4.1491, 4.04101, 3.17872, 3.95226, 1.77118, 3.40646,
        ]);
        let positions = [p1, p2];

        let graph = create_graph(2);
        let embedding = Embedding {
            positions: positions.to_vec(),
            graph: &graph,
        };

        let lsh = Lsh::new(embedding);
        dbg!(lsh.nearest_neighbors(0, 1.));
        panic!();
    }
    #[test]
    fn should_intersect_v2() {
        return;
        let p1 = point([
            -0.774175, 1.06153, 4.19799, 3.894, 3.41074, 3.78547, 1.98516, 1.36116,
        ]);
        let p2 = point([
            -0.837306, 1.53845, 3.98596, 3.96884, 3.17616, 3.79302, 2.28606, 1.26695,
        ]);
        let positions = [p1, p2];

        let graph = create_graph(2);
        let embedding = Embedding {
            positions: positions.to_vec(),
            graph: &graph,
        };

        let lsh = Lsh::new(embedding);
        dbg!(lsh.nearest_neighbors(0, 1.));
        panic!();
    }
    #[test]
    fn test_nbig_box() {
        let p1 = point([
            3.89074, 1.63223, 3.89705, 3.22904, 2.2936, 2.76524, 2.58811, 1.69013,
        ]);
        let boxes = nbig_box::<8, DIM_CHUNK_SIZE>(&p1, 0, DIM_CHUNK_SIZE);
        let p2 = point([
            3.99191, 1.21484, 3.99002, 2.95835, 2.89702, 2.70426, 2.83737, 1.67377,
        ]);
        eprintln!(
            "Testing\n{:?} vs:",
            super::point::<8, DIM_CHUNK_SIZE>(&p2, 0, DIM_CHUNK_SIZE)
                .next()
                .unwrap()
        );
        for b in boxes {
            eprintln!("{:?}", b);
        }
        let boxes = nbig_box::<8, DIM_CHUNK_SIZE>(&p1, 2, DIM_CHUNK_SIZE);
        eprintln!(
            "Testing\n{:?} vs:",
            super::point::<8, DIM_CHUNK_SIZE>(&p2, 2, DIM_CHUNK_SIZE)
                .next()
                .unwrap()
        );
        for b in boxes {
            eprintln!("{:?}", b);
        }
        let positions = [p1, p2];

        let graph = create_graph(2);
        let embedding = Embedding {
            positions: positions.to_vec(),
            graph: &graph,
        };

        let lsh = Lsh::new(embedding);
        let list = dbg!(lsh.nearest_neighbors(0, 1.));
        assert!(list.contains(&1));
        // panic!();
    }

    fn point(arr: [f32; 8]) -> DVec<8> {
        DVec { components: arr }
    }
    fn create_graph(n: i8) -> Graph {
        Graph {
            nodes: (0..n)
                .map(|_| Node {
                    weight: 0.9173302940621955,
                    neighbors: vec![],
                })
                .collect(),
            edges: vec![],
        }
    }
}
