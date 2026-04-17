use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

const MAX_DEPTH: usize = 10;
const NODE_CAPACITY: usize = 32;

/// D-dimensional axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperRect<const D: usize> {
    pub min: DVec<D>,
    pub max: DVec<D>,
}

impl<const D: usize> HyperRect<D> {
    #[inline(always)]
    pub fn new(min: DVec<D>, max: DVec<D>) -> Self {
        Self { min, max }
    }

    #[inline(always)]
    pub fn center(&self) -> DVec<D> {
        (self.min + self.max) * 0.5
    }

    /// Returns the orthant index (0..2^D) of a point relative to the given center.
    #[inline(always)]
    pub fn orthant(point: &DVec<D>, center: &DVec<D>) -> usize {
        let mut idx = 0;
        for d in 0..D {
            if point[d] >= center[d] {
                idx |= 1 << d;
            }
        }
        idx
    }

    #[inline(always)]
    pub fn intersects(&self, other: &HyperRect<D>) -> bool {
        for d in 0..D {
            if self.min[d] > other.max[d] || self.max[d] < other.min[d] {
                return false;
            }
        }
        true
    }

    /// Returns true if the entire box is within the given sphere.
    #[inline(always)]
    pub fn contained_in_sphere(&self, center: &DVec<D>, radius_sq: f32) -> bool {
        let mut max_dist_sq = 0.0f32;
        for d in 0..D {
            let d_min = center[d] - self.min[d];
            let d_max = center[d] - self.max[d];
            let far = if d_min.abs() > d_max.abs() {
                d_min
            } else {
                d_max
            };
            max_dist_sq += far * far;
        }
        max_dist_sq <= radius_sq
    }

    /// Compute the child bound for a given orthant.
    #[inline(always)]
    pub fn child_bound(&self, orthant: usize, center: &DVec<D>) -> HyperRect<D> {
        let mut child_min = self.min;
        let mut child_max = self.max;
        for d in 0..D {
            if orthant & (1 << d) != 0 {
                child_min.components[d] = center[d];
            } else {
                child_max.components[d] = center[d];
            }
        }
        HyperRect::new(child_min, child_max)
    }
}

// ===========================================================================
// Arena-based orthtree with in-place partitioning.
//
// All node metadata is stored in parallel flat arrays. Point ids are
// partitioned in-place (like sprk's build_tree). Leaf nodes reference
// slices of the sorted id array by (offset, len).
// ===========================================================================

const SENTINEL: u32 = u32::MAX;

/// Flat arena storing the orthtree.
///  - `bounds[i]`:       bounding box of node i
///  - `first_child[i]`:  index of first child (SENTINEL for leaves)
///  - `leaf_start[i]`:   offset into `ids` for leaf data
///  - `leaf_len[i]`:     number of ids in the leaf
///  - `ids`:             flat array of point ids, partitioned in-place
pub struct OrthtreeArena<const D: usize> {
    bounds: Vec<HyperRect<D>>,
    first_child: Vec<u32>,
    leaf_start: Vec<u32>,
    leaf_len: Vec<u32>,
    ids: Vec<u32>,
}

impl<const D: usize> OrthtreeArena<D> {
    fn build(positions: &[DVec<D>], bound: HyperRect<D>) -> Self {
        let n = positions.len();
        let mut arena = OrthtreeArena {
            bounds: Vec::with_capacity(n / NODE_CAPACITY * 2),
            first_child: Vec::with_capacity(n / NODE_CAPACITY * 2),
            leaf_start: Vec::with_capacity(n / NODE_CAPACITY * 2),
            leaf_len: Vec::with_capacity(n / NODE_CAPACITY * 2),
            ids: (0..n as u32).collect(),
        };
        arena.build_node(bound, 0, n, positions, 0);
        arena
    }

    /// Build a node covering ids[start..start+len]. Returns the node's arena index.
    fn build_node(
        &mut self,
        bound: HyperRect<D>,
        start: usize,
        len: usize,
        positions: &[DVec<D>],
        depth: usize,
    ) -> u32 {
        let node_idx = self.bounds.len() as u32;

        // Leaf
        if len <= NODE_CAPACITY || depth >= MAX_DEPTH {
            self.bounds.push(bound);
            self.first_child.push(SENTINEL);
            self.leaf_start.push(start as u32);
            self.leaf_len.push(len as u32);
            return node_idx;
        }

        // Internal: partition ids[start..start+len] by orthant in-place
        let num_children = 1usize << D;
        let center = bound.center();

        // Count points per orthant
        let mut counts = vec![0u32; num_children];
        for i in start..start + len {
            let orth = HyperRect::orthant(&positions[self.ids[i] as usize], &center);
            counts[orth] += 1;
        }

        // Compute prefix sums (offsets within the start..start+len range)
        let mut offsets = vec![0u32; num_children + 1];
        for i in 0..num_children {
            offsets[i + 1] = offsets[i] + counts[i];
        }

        // In-place partition using a temp buffer
        // (cycling permutation is complex; a temp copy is simpler and fast for small slices)
        let slice = self.ids[start..start + len].to_vec();
        let mut cursors = offsets[..num_children].to_vec();
        for id in slice {
            let orth = HyperRect::orthant(&positions[id as usize], &center);
            self.ids[start + cursors[orth] as usize] = id;
            cursors[orth] += 1;
        }

        // Reserve this node's slot
        self.bounds.push(bound);
        self.first_child.push(SENTINEL); // patched below
        self.leaf_start.push(0);
        self.leaf_len.push(0);

        // Build children
        let fc = self.bounds.len() as u32;

        // Reserve child slots first so they are contiguous
        for _ in 0..num_children {
            self.bounds.push(HyperRect::new(DVec::zero(), DVec::zero()));
            self.first_child.push(SENTINEL);
            self.leaf_start.push(0);
            self.leaf_len.push(0);
        }

        for orth in 0..num_children {
            let child_start = start + offsets[orth] as usize;
            let child_len = counts[orth] as usize;
            let child_bound = bound.child_bound(orth, &center);
            let child_slot = fc as usize + orth;

            if child_len <= NODE_CAPACITY || depth + 1 >= MAX_DEPTH {
                // Inline leaf directly into the reserved slot
                self.bounds[child_slot] = child_bound;
                self.first_child[child_slot] = SENTINEL;
                self.leaf_start[child_slot] = child_start as u32;
                self.leaf_len[child_slot] = child_len as u32;
            } else {
                // Build subtree; the root of the subtree goes into a new slot,
                // but we want it in child_slot. Build, then copy.
                let sub_root =
                    self.build_node(child_bound, child_start, child_len, positions, depth + 1);
                self.bounds[child_slot] = self.bounds[sub_root as usize];
                self.first_child[child_slot] = self.first_child[sub_root as usize];
                self.leaf_start[child_slot] = self.leaf_start[sub_root as usize];
                self.leaf_len[child_slot] = self.leaf_len[sub_root as usize];
                // sub_root slot is now dead (unreachable), acceptable waste
            }
        }

        self.first_child[node_idx as usize] = fc;
        node_idx
    }

    #[inline]
    fn query_node(
        &self,
        idx: usize,
        center: &DVec<D>,
        radius_sq: f32,
        query_aabb: &HyperRect<D>,
        positions: &[DVec<D>],
        results: &mut Vec<NodeId>,
    ) {
        if self.first_child[idx] == SENTINEL {
            // Leaf
            let s = self.leaf_start[idx] as usize;
            let e = s + self.leaf_len[idx] as usize;
            if self.bounds[idx].contained_in_sphere(center, radius_sq) {
                for &id in &self.ids[s..e] {
                    results.push(id as usize);
                }
            } else {
                for &id in &self.ids[s..e] {
                    if center.distance_squared(&positions[id as usize]) <= radius_sq {
                        results.push(id as usize);
                    }
                }
            }
        } else {
            let fc = self.first_child[idx] as usize;
            let num_children = 1 << D;
            for i in 0..num_children {
                let ci = fc + i;
                if !self.bounds[ci].intersects(query_aabb) {
                    continue;
                }
                self.query_node(ci, center, radius_sq, query_aabb, positions, results);
                // if self.bounds[ci].contained_in_sphere(center, radius_sq) {
                //     self.collect_all(ci, results);
                // } else {
                //     self.query_node(ci, center, radius_sq, query_aabb, positions, results);
                // }
            }
        }
    }

    fn collect_all(&self, idx: usize, results: &mut Vec<NodeId>) {
        if self.first_child[idx] == SENTINEL {
            let s = self.leaf_start[idx] as usize;
            let e = s + self.leaf_len[idx] as usize;
            for &id in &self.ids[s..e] {
                results.push(id as usize);
            }
        } else {
            let fc = self.first_child[idx] as usize;
            let num_children = 1 << D;
            for i in 0..num_children {
                self.collect_all(fc + i, results);
            }
        }
    }
}

// ===========================================================================
// Public wrapper
// ===========================================================================

pub struct Orthtree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    arena: Option<OrthtreeArena<D>>,
}

impl<'a, const D: usize> Orthtree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            arena: None,
        };
        tree.update_positions(&embedding.positions, None);
        tree
    }
}

impl<'a, const D: usize> Clone for Orthtree<'a, D> {
    fn clone(&self) -> Self {
        let mut tree = Self {
            positions: self.positions.clone(),
            graph: self.graph,
            arena: None,
        };
        tree.update_positions(&self.positions, None);
        tree
    }
}

impl<'a, const D: usize> Graph for Orthtree<'a, D> {
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

impl<'a, const D: usize> Position<D> for Orthtree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }
    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for Orthtree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();

        if positions.is_empty() {
            self.arena = None;
            return;
        }

        let mut bb_min = positions[0];
        let mut bb_max = positions[0];
        for pos in positions {
            for d in 0..D {
                bb_min.components[d] = bb_min.components[d].min(pos[d]);
                bb_max.components[d] = bb_max.components[d].max(pos[d]);
            }
        }

        self.arena = Some(OrthtreeArena::build(
            positions,
            HyperRect::new(bb_min, bb_max),
        ));
    }
}

impl<'a, const D: usize> Query<D> for Orthtree<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let Some(arena) = &self.arena else { return };
        if arena.bounds.is_empty() {
            return;
        }

        let radius_sq = (radius * radius) as f32;
        let r = radius as f32;
        let mut aabb_min = pos;
        let mut aabb_max = pos;
        for d in 0..D {
            aabb_min.components[d] -= r;
            aabb_max.components[d] += r;
        }
        let query_aabb = HyperRect::new(aabb_min, aabb_max);

        arena.query_node(0, &pos, radius_sq, &query_aabb, &self.positions, results);
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Orthtree<'a, D> {
    fn name(&self) -> String {
        "orthtree".to_string()
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("orthtree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Orthtree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
