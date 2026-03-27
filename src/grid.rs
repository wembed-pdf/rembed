use crate::{
    Embedding, NodeId, Query,
    dvec::DVec,
    query::{self, Graph, Position, SpatialIndex, Update},
};
use std::cmp::{max, min};

#[derive(Clone)]
pub struct PositionWithId {
    id: NodeId,
    position_0: f32,
    position_1: f32,
}

#[derive(Clone)]
pub struct GridCell {
    ids: Vec<NodeId>,
    positions: Vec<PositionWithId>,
}

pub struct Grid<'a, const D: usize> {
    pub positions: Vec<DVec<D>>,
    pub graph: &'a crate::graph::Graph,
    pub grid: Vec<Vec<GridCell>>,
    pub grid_size: f64,
    pub min_x: f64,
    pub min_y: f64,
}

impl<'a, const D: usize> Grid<'a, D> {
    pub fn new(embedding: Embedding<'a, D>) -> Self {
        if D != 2 {
            println!(
                "Warning: Grid is only implemented for 2D embeddings. The provided embedding has dimension {}.",
                D
            );
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for pos in &embedding.positions {
            min_x = min_x.min(pos[0]);
            min_y = min_y.min(pos[1]);
            max_x = max_x.max(pos[0]);
            max_y = max_y.max(pos[1]);
        }

        let grid_size = 1.;

        let grid_width = ((max_x - min_x) / grid_size as f32).ceil() as usize;
        let grid_height = ((max_y - min_y) / grid_size as f32).ceil() as usize;
        let grid = vec![
            vec![
                GridCell {
                    ids: Vec::new(),
                    positions: Vec::new()
                };
                grid_height
            ];
            grid_width
        ];
        let mut tree = Self {
            positions: embedding.positions.clone(),
            graph: embedding.graph,
            grid: grid,
            grid_size,
            min_x: min_x as f64,
            min_y: min_y as f64,
        };
        tree.update_positions(&embedding.positions, None);
        tree
    }
}

impl<'a, const D: usize> Clone for Grid<'a, D> {
    fn clone(&self) -> Self {
        if D != 2 {
            println!(
                "Warning: Grid is only implemented for 2D embeddings. The provided embedding has dimension {}.",
                D
            );
        }
        let mut tree = Self {
            positions: self.positions.clone(),
            graph: self.graph,
            grid: self.grid.clone(),
            grid_size: self.grid_size,
            min_x: self.min_x,
            min_y: self.min_y,
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
        if D != 2 {
            println!(
                "Warning: Grid is only implemented for 2D embeddings. The provided embedding has dimension {}.",
                D
            );
        }
        // Clear the grid before updating positions
        let mut grid = vec![
            vec![
                GridCell {
                    ids: Vec::new(),
                    positions: Vec::new()
                };
                self.grid[0].len()
            ];
            self.grid.len()
        ];
        for (i, pos) in positions.iter().enumerate() {
            let mut grid_x =
                ((pos[0] - self.min_x as f32) / self.grid_size as f32).floor() as usize;
            let mut grid_y =
                ((pos[1] - self.min_y as f32) / self.grid_size as f32).floor() as usize;

            if grid_x >= grid.len() {
                if grid_x >= grid.len() + 1 {
                    println!(
                        "Warning: Node {} with position {:?} is far out of grid bounds in x direction (grid_x: {}, grid width: {}). Clamping to grid boundary.",
                        i,
                        pos,
                        grid_x,
                        grid.len()
                    );
                }
                grid_x = grid.len() - 1;
            }
            if grid_y >= grid[0].len() {
                if grid_y >= grid[0].len() + 1 {
                    println!(
                        "Warning: Node {} with position {:?} is far out of grid bounds in y direction (grid_y: {}, grid height: {}). Clamping to grid boundary.",
                        i,
                        pos,
                        grid_y,
                        grid[0].len()
                    );
                }
                grid_y = grid[0].len() - 1;
            }

            grid[grid_x][grid_y].ids.push(i);
            grid[grid_x][grid_y].positions.push(PositionWithId {
                id: i,
                position_0: pos[0],
                position_1: pos[1],
            });
        }
        self.grid = grid;
        self.positions = positions.to_vec();
    }
}

impl<'a, const D: usize> Query<D> for Grid<'a, D> {
    fn query_radius(&self, pos: DVec<D>, radius: f64, results: &mut Vec<NodeId>) {
        let own_position = pos;

        if D != 2 {
            return;
        }

        let own_position_2d = DVec::from([own_position[0], own_position[1]]);
        let own_x = own_position[0];
        let own_y = own_position[1];
        let scaled_radius = (radius * radius) as f32;

        let min_grid_x = max(
            0,
            ((own_position[0] - self.min_x as f32 - radius as f32) / self.grid_size as f32).floor()
                as i32,
        ) as usize;
        let max_grid_x = min(
            self.grid.len() as usize - 1,
            ((own_position[0] - self.min_x as f32 + radius as f32) / self.grid_size as f32).floor()
                as usize,
        );
        let min_grid_y = max(
            0,
            ((own_position[1] - self.min_y as f32 - radius as f32) / self.grid_size as f32).floor()
                as usize,
        ) as usize;
        let max_grid_y = min(
            self.grid[0].len() as usize - 1,
            ((own_position[1] - self.min_y as f32 + radius as f32) / self.grid_size as f32).floor()
                as usize,
        ) as usize;

        // clamp grid coordinates to grid boundaries
        let min_grid_x = max(0, min_grid_x);
        let max_grid_x = min(self.grid.len() as usize - 1, max_grid_x);
        let min_grid_y = max(0, min_grid_y);
        let max_grid_y = min(self.grid[0].len() as usize - 1, max_grid_y);

        for x in min_grid_x..=max_grid_x {
            for y in min_grid_y..=max_grid_y {
                // if radius > 1 check if the complete grid cell is within the radius, if so add all nodes in the cell without checking their distance
                if radius > 2. * self.grid_size {
                    let cell_min_x = self.min_x as f32 + (x as f32) * self.grid_size as f32;
                    let cell_max_x = cell_min_x + self.grid_size as f32;
                    let cell_min_y = self.min_y as f32 + (y as f32) * self.grid_size as f32;
                    let cell_max_y = cell_min_y + self.grid_size as f32;

                    let top_left_is_within = (cell_min_x - own_position_2d[0])
                        * (cell_min_x - own_position_2d[0])
                        + (cell_min_y - own_position_2d[1]) * (cell_min_y - own_position_2d[1])
                        <= scaled_radius;
                    let top_right_is_within = (cell_max_x - own_position_2d[0])
                        * (cell_max_x - own_position_2d[0])
                        + (cell_min_y - own_position_2d[1]) * (cell_min_y - own_position_2d[1])
                        <= scaled_radius;
                    let bottom_left_is_within = (cell_min_x - own_position_2d[0])
                        * (cell_min_x - own_position_2d[0])
                        + (cell_max_y - own_position_2d[1]) * (cell_max_y - own_position_2d[1])
                        - 2.0 * (cell_min_x * own_position_2d[0] + cell_max_y * own_position_2d[1])
                        <= scaled_radius;
                    let bottom_right_is_within = (cell_max_x - own_position_2d[0])
                        * (cell_max_x - own_position_2d[0])
                        + (cell_max_y - own_position_2d[1]) * (cell_max_y - own_position_2d[1])
                        <= scaled_radius;

                    if top_left_is_within
                        && top_right_is_within
                        && bottom_left_is_within
                        && bottom_right_is_within
                    {
                        results.extend_from_slice(&self.grid[x][y].ids);
                        continue;
                    }
                }
                let mut len = results.len();
                results.reserve(self.grid[x][y].ids.len());
                for i in 0..self.grid[x][y].ids.len() {
                    unsafe {
                        let position = self.grid[x][y].positions.get_unchecked(i);

                        *results.get_unchecked_mut(len) = position.id;
                        len += (((own_x - position.position_0) * (own_x - position.position_0)
                            + (own_y - position.position_1) * (own_y - position.position_1))
                            <= scaled_radius) as usize;
                    }
                }
                unsafe {
                    results.set_len(len);
                }
            }
        }
    }
}

impl<'a, const D: usize> SpatialIndex<D> for Grid<'a, D> {
    fn name(&self) -> String {
        "grid".to_string()
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
