use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

const DEPTH: usize = 10;
const NODE_CAPACITY: usize = 50;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Rect {
    /// Start
    pub aa: DVec<2>,
    /// End
    pub bb: DVec<2>,
}

impl Rect {
    pub const fn new(a: DVec<2>, b: DVec<2>) -> Self {
        Self { aa: a, bb: b }
    }

    /// Returns true if the point is inside the rectangle.
    pub fn contains(&self, point: DVec<2>) -> bool {
        point[0] >= self.aa[0]
            && point[0] <= self.bb[0]
            && point[1] >= self.aa[1]
            && point[1] <= self.bb[1]
    }

    /// Returns the quadrant (0-3) of the point relative to this Rect, or None if not inside.
    pub fn quadrant(&self, point: DVec<2>) -> Option<usize> {
        let center = (self.aa + self.bb) * 0.5;
        let in_x = point[0] >= self.aa[0] && point[0] <= self.bb[0];
        let in_y = point[1] >= self.aa[1] && point[1] <= self.bb[1];
        if !in_x || !in_y {
            return None;
        }
        let quad = match (point[0] >= center[0], point[1] >= center[1]) {
            (false, false) => 0, // bottom-left
            (true, false) => 1,  // bottom-right
            (false, true) => 2,  // top-left
            (true, true) => 3,   // top-right
        };
        Some(quad)
    }

    /// Returns true if this rectangle intersects with another rectangle.
    /// This version avoids short-circuiting for branch prediction optimization.
    pub fn intersects(&self, other: &Rect) -> bool {
        let x1 = self.aa[0] <= other.bb[0];
        let x2 = self.bb[0] >= other.aa[0];
        let y1 = self.aa[1] <= other.bb[1];
        let y2 = self.bb[1] >= other.aa[1];
        x1 & x2 & y1 & y2
    }

    /// Quarter the rect to produce four smaller rects
    pub fn quarter(&self) -> [Self; 4] {
        let center = self.center();
        let diff = center - self.aa;
        let diff_x = DVec::<2>::new([diff[0], 0.]);
        let diff_y = DVec::<2>::new([0., diff[1]]);

        [
            Rect::new(self.aa, center),
            Rect::new(self.aa + diff_x, center + diff_x),
            Rect::new(self.aa + diff_y, center + diff_y),
            Rect::new(center, self.bb),
        ]
    }

    /// Get the midpoint of aa and bb
    pub fn center(&self) -> DVec<2> {
        (self.aa + self.bb) * 0.5
    }
}

/// Groups items by their quadrant within the given bounding rectangle.
/// Returns a Vec of 5 Vecs: one for each quadrant (0-3) and one for items that don't fit.
fn group_by_quadrant(bound: Rect, items: Vec<TreeItem>) -> [Vec<TreeItem>; 5] {
    let mut groups: [Vec<TreeItem>; 5] = Default::default();
    for item in items {
        match bound.quadrant(item.point) {
            Some(q) if q < 4 => groups[q].push(item),
            _ => groups[4].push(item), // Items that don't fit in any quadrant
        }
    }
    groups
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TreeItem {
    pub id: usize,
    pub point: DVec<2>,
}

pub struct TreeQuery {
    pub center: DVec<2>,
    pub radius: f32,
    pub aabb: Rect,
}

impl TreeQuery {
    pub fn contains_fast(&self, point: DVec<2>) -> bool {
        (point[0] - self.center[0]).powi(2) + (point[1] - self.center[1]).powi(2) <= self.radius
    }

    pub fn contains_rect(&self, rect: &Rect) -> bool {
        self.contains_fast(rect.aa)
            && self.contains_fast(rect.bb)
            && self.contains_fast(DVec::<2>::new([rect.aa[0], rect.bb[1]]))
            && self.contains_fast(DVec::<2>::new([rect.bb[0], rect.aa[1]]))
    }
}

/// A generic Quadtree implementation for spatial indexing of 2D points
#[derive(Debug)]
pub struct QuadtreeTree {
    root: Node,
    node_capacity: usize,
    max_depth: usize,
    count: usize,
}

impl QuadtreeTree {
    /// Create a new empty quadtree
    ///
    /// ## Arguments
    /// - `bound`: The bound of the quadtree
    /// - `node_capacity`: The maximum number of items a node can hold before subdividing
    /// - `max_depth`: The maximum depth of the tree at which nodes will ignore capacity
    pub const fn new(bound: Rect, node_capacity: usize, max_depth: usize) -> Self {
        Self {
            root: Node {
                bound: bound,
                children: None,
                only_ids: Vec::new(),
                data: Vec::new(),
            },
            node_capacity,
            max_depth,
            count: 0,
        }
    }

    /// Get current number of items stored
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Insert multiple items into the Quadtree
    ///
    /// **Returns** a vector of items that failed to insert, if any
    pub fn insert_many(&mut self, items: &[TreeItem]) -> Vec<usize> {
        let items = items.to_vec();
        let num_items = items.len();
        let mut failed = Vec::with_capacity(items.len());
        self.root
            .insert_many(items, self.node_capacity, 0, self.max_depth, &mut failed);
        self.count += num_items - failed.len();
        failed
    }

    /// Query for items within a specified shape area
    ///
    /// **Returns** a vector of immutable references to items
    pub fn query_ref_tree(&self, shape: &TreeQuery, results: &mut Vec<NodeId>) {
        self.root.query_ref(shape, results);
    }

    /// Return the point at the center of the boundary
    pub fn center(&self) -> DVec<2> {
        self.root.center()
    }

    /// Get the boundary rect of the quadtree
    pub fn bound(&self) -> Rect {
        self.root.bound
    }
}

#[derive(Debug)]
pub struct Node {
    bound: Rect,
    children: Option<[Box<Self>; 4]>,
    only_ids: Vec<usize>, // only store ids for faster queries
    data: Vec<TreeItem>,
}

impl Node {
    fn insert_many(
        &mut self,
        items: Vec<TreeItem>,
        capacity: usize,
        depth: usize,
        max_depth: usize,
        failed: &mut Vec<usize>,
    ) {
        // println!("Node has children: {}", self.children.is_some());
        // If this node has children, distribute items to children
        if let Some(children) = &mut self.children {
            // println!(
            //     "Inserting {} items into node at depth {}",
            //     items.len(),
            //     depth,
            // );
            let mut groups = group_by_quadrant(self.bound, items).into_iter();
            for c in children {
                let items = groups.next().unwrap();
                if items.len() > 0 {
                    c.insert_many(items, capacity, depth + 1, max_depth, failed)
                }
            }
            let cur_failed = groups.next().unwrap();
            if cur_failed.len() > 0 {
                failed.extend(cur_failed.into_iter().map(|item| item.id));
            }
            return;
        }

        // If node has enough capacity or reached max depth, store items here
        if self.data.len() + items.len() <= capacity || depth >= max_depth {
            self.only_ids.extend(items.iter().map(|item| item.id));
            self.data.extend(items);
            return;
        }

        // Otherwise, subdivide and re-insert all items (including existing)
        let mut all_items = std::mem::take(&mut self.data);
        all_items.extend(items);
        let children = self.subdivide();
        self.children = Some(children);
        self.only_ids.clear();
        self.insert_many(all_items, capacity, depth, max_depth, failed);
    }

    fn query_ref(&self, shape: &TreeQuery, results: &mut Vec<NodeId>) {
        if self.children.is_none() {
            if shape.contains_rect(&self.bound) {
                results.extend_from_slice(&self.only_ids);
                return;
            }

            for item in self.data.iter() {
                if shape.contains_fast(item.point) {
                    results.push(item.id);
                }
            }
        } else {
            for child in self.children.as_ref().unwrap() {
                if child.bound.intersects(&shape.aabb) {
                    child.query_ref(shape, results);
                }
            }
        }
    }

    fn center(&self) -> DVec<2> {
        self.bound.center()
    }

    fn subdivide(&self) -> [Box<Self>; 4] {
        let rects = self.bound.quarter();
        rects.map(|r| {
            Box::new(Self {
                bound: r,
                children: None,
                only_ids: Vec::new(),
                data: Vec::new(),
            })
        })
    }
}

pub struct Quadtree<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub quadtree: QuadtreeTree,
}

impl<'a, const D: usize> Quadtree<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        if D != 2 {
            return Self {
                positions: embedding.positions.to_vec(),
                graph: embedding.graph,
                quadtree: QuadtreeTree::new(
                    Rect {
                        aa: DVec::<2>::new([0.0, 0.0]),
                        bb: DVec::<2>::new([1.0, 1.0]),
                    },
                    NODE_CAPACITY,
                    DEPTH,
                ),
            };
        }
        let mut tree = Self {
            positions: embedding.positions.to_vec(),
            graph: embedding.graph,
            quadtree: QuadtreeTree::new(
                Rect {
                    aa: DVec::<2>::new([0.0, 0.0]),
                    bb: DVec::<2>::new([1.0, 1.0]),
                },
                NODE_CAPACITY,
                DEPTH,
            ),
        };
        tree.update_positions(&embedding.positions, None);
        tree
    }
}

impl<'a, const D: usize> Clone for Quadtree<'a, D> {
    fn clone(&self) -> Self {
        if D != 2 {
            return Self {
                positions: self.positions.clone(),
                graph: self.graph,
                quadtree: QuadtreeTree::new(
                    Rect {
                        aa: DVec::<2>::new([0.0, 0.0]),
                        bb: DVec::<2>::new([1.0, 1.0]),
                    },
                    NODE_CAPACITY,
                    DEPTH,
                ),
            };
        }
        let mut tree = Self {
            positions: self.positions.clone(),
            graph: self.graph,
            quadtree: QuadtreeTree::new(
                Rect {
                    aa: DVec::<2>::new([0.0, 0.0]),
                    bb: DVec::<2>::new([1.0, 1.0]),
                },
                NODE_CAPACITY,
                DEPTH,
            ),
        };
        tree.update_positions(&self.positions, None);
        tree
    }
}

impl<'a, const D: usize> Graph for Quadtree<'a, D> {
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

impl<'a, const D: usize> Position<D> for Quadtree<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for Quadtree<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        self.positions = positions.to_vec();
        if D != 2 {
            return;
        }
        let min_x = self
            .positions
            .iter()
            .map(|p| p[0])
            .fold(f32::INFINITY, |a, b| a.min(b));
        let max_x = self
            .positions
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let min_y = self
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::INFINITY, |a, b| a.min(b));
        let max_y = self
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let bound = Rect {
            aa: DVec::<2>::new([min_x, min_y]),
            bb: DVec::<2>::new([max_x, max_y]),
        };
        self.quadtree = QuadtreeTree::new(bound, NODE_CAPACITY, DEPTH);
        let items: Vec<TreeItem> = self
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| TreeItem {
                id: i,
                point: DVec::<2>::new([p[0], p[1]]),
            })
            .collect();
        self.quadtree.insert_many(&items);
    }
}

impl<'a, const D: usize> Query<D> for Quadtree<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        if D != 2 {
            return;
        }
        let scaled_radius = radius.powi(2);

        let query = TreeQuery {
            center: DVec::<2>::new([pos[0], pos[1]]),
            radius: scaled_radius as f32,
            aabb: Rect {
                aa: DVec::<2>::new([pos[0] - radius as f32, pos[1] - radius as f32]),
                bb: DVec::<2>::new([pos[0] + radius as f32, pos[1] + radius as f32]),
            },
        };
        self.quadtree.root.query_ref(&query, results);
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Quadtree<'a, D> {
    fn name(&self) -> String {
        "quadtree".to_string()
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("quadtree.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Quadtree<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
