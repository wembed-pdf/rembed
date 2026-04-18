use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};

pub struct Grid<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    cells: Vec<GridCellInner>,
    cell_positions: Vec<Vec<DVec<D>>>,
    grid_size: f64,
    min: [f32; D],
    extents: [usize; D],
}

#[derive(Clone)]
struct GridCellInner {
    ids: Vec<NodeId>,
}

impl<'a, const D: usize> Grid<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        let mut tree = Self {
            positions: Vec::new(),
            graph: embedding.graph,
            cells: Vec::new(),
            cell_positions: Vec::new(),
            grid_size: 1.0,
            min: [0.0; D],
            extents: [1; D],
        };
        tree.update_positions(&embedding.positions, None);
        tree
    }

    /// Convert D-dimensional grid coordinates to a flat index.
    #[inline(always)]
    fn flat_index(&self, coords: &[usize; D]) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        for d in (0..D).rev() {
            idx += coords[d] * stride;
            stride *= self.extents[d];
        }
        idx
    }

    /// Compute the grid coordinate for a position along dimension `d`.
    #[inline(always)]
    fn grid_coord(&self, pos: f32, d: usize) -> usize {
        let c = ((pos - self.min[d]) / self.grid_size as f32).floor() as usize;
        c.min(self.extents[d] - 1)
    }

    /// Check if a cell (given by its grid coordinates) is entirely within a sphere.
    #[inline(always)]
    fn cell_contained_in_sphere(
        &self,
        coords: &[usize; D],
        center: &DVec<D>,
        radius_sq: f32,
    ) -> bool {
        // The farthest corner from the center determines if the whole cell is inside.
        // For each dimension, pick whichever corner edge is farther from center.
        let mut max_dist_sq = 0.0f32;
        for d in 0..D {
            let cell_min = self.min[d] + (coords[d] as f32) * self.grid_size as f32;
            let cell_max = cell_min + self.grid_size as f32;
            let d_min = center[d] - cell_min;
            let d_max = center[d] - cell_max;
            let far = if d_min.abs() > d_max.abs() {
                d_min
            } else {
                d_max
            };
            max_dist_sq += far * far;
        }
        max_dist_sq <= radius_sq
    }
}

impl<'a, const D: usize> Clone for Grid<'a, D> {
    fn clone(&self) -> Self {
        let mut tree = Self {
            positions: self.positions.clone(),
            graph: self.graph,
            cells: self.cells.clone(),
            cell_positions: self.cell_positions.clone(),
            grid_size: self.grid_size,
            min: self.min,
            extents: self.extents,
        };
        tree.update_positions(&self.positions, None);
        tree
    }
}

impl<'a, const D: usize> Graph for Grid<'a, D> {
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

impl<'a, const D: usize> Position<D> for Grid<'a, D> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        &self.positions[index]
    }

    fn num_nodes(&self) -> usize {
        self.positions.len()
    }
}

impl<'a, const D: usize> Update<D> for Grid<'a, D> {
    fn update_positions(&mut self, positions: &[DVec<D>], _: Option<f64>) {
        // Recompute bounding box and extents
        let mut min = [f32::MAX; D];
        let mut max = [f32::MIN; D];
        for pos in positions {
            for d in 0..D {
                min[d] = min[d].min(pos[d]);
                max[d] = max[d].max(pos[d]);
            }
        }
        self.min = min;
        for d in 0..D {
            self.extents[d] = ((max[d] - min[d]) / self.grid_size as f32).ceil().max(1.0) as usize;
        }

        let total_cells: usize = self.extents.iter().product();
        let mut cells = vec![GridCellInner { ids: Vec::new() }; total_cells];
        let mut cell_positions: Vec<Vec<DVec<D>>> = vec![Vec::new(); total_cells];

        for (i, pos) in positions.iter().enumerate() {
            let mut coords = [0usize; D];
            for d in 0..D {
                coords[d] = self.grid_coord(pos[d], d);
            }
            let idx = self.flat_index(&coords);
            cells[idx].ids.push(i);
            cell_positions[idx].push(*pos);
        }

        self.cells = cells;
        self.cell_positions = cell_positions;
        self.positions = positions.to_vec();
    }
}

impl<'a, const D: usize> Query<D> for Grid<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let radius_sq = (radius * radius) as f32;
        let grid_size = self.grid_size as f32;

        // Compute the range of grid cells to check per dimension
        let mut min_grid = [0usize; D];
        let mut max_grid = [0usize; D];
        for d in 0..D {
            let lo = ((pos[d] - self.min[d] - radius as f32) / grid_size).floor() as i32;
            let hi = ((pos[d] - self.min[d] + radius as f32) / grid_size).floor() as i32;
            min_grid[d] = lo.max(0) as usize;
            max_grid[d] = (hi as usize).min(self.extents[d] - 1);
        }

        let check_containment = radius > 2.0 * self.grid_size;

        // Enumerate all cells in the D-dimensional box [min_grid, max_grid]
        // using an iterative counter.
        let mut coords = min_grid;
        loop {
            let flat = self.flat_index(&coords);

            // Whole-cell containment optimization
            if check_containment && self.cell_contained_in_sphere(&coords, &pos, radius_sq) {
                results.extend_from_slice(&self.cells[flat].ids);
            } else {
                // Branchless per-point distance check
                let cell_ids = &self.cells[flat].ids;
                let cell_pos = &self.cell_positions[flat];
                let mut len = results.len();
                results.reserve(cell_ids.len());
                for i in 0..cell_ids.len() {
                    unsafe {
                        let p = cell_pos.get_unchecked(i);
                        *results.get_unchecked_mut(len) = cell_ids[i];
                        len += (pos.distance_squared(p) <= radius_sq) as usize;
                    }
                }
                unsafe {
                    results.set_len(len);
                }
            }

            // Advance the multi-dimensional counter (rightmost dimension first)
            let mut carry = true;
            for d in (0..D).rev() {
                if carry {
                    coords[d] += 1;
                    if coords[d] > max_grid[d] {
                        coords[d] = min_grid[d];
                    } else {
                        carry = false;
                        break;
                    }
                }
            }
            if carry {
                break;
            }
        }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Grid<'a, D> {
    fn name(&self) -> String {
        "grid".to_string()
    }

    fn set_radius_hint(&mut self, radius: f64) {
        self.grid_size = radius;
        let positions = std::mem::take(&mut self.positions);
        self.update_positions(&positions, None);
    }

    fn implementation_string(&self) -> &'static str {
        include_str!("grid.rs")
    }
}

impl<'a, const D: usize> query::Embedder<'a, D> for Grid<'a, D> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        Self::new(embedding.clone())
    }
}
