use core::panic;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::dvec::DVec;

#[derive(Debug)]
pub struct Position<const D: usize> {
    pub index: usize,
    pub weight: f64,
    pub coordinates: DVec<D>,
}

#[derive(Debug)]
pub struct Iteration<const D: usize> {
    pub number: usize,
    pub positions: Vec<Position<D>>,
}

pub fn parse_positions_file<P: AsRef<Path>, const D: usize>(
    path: P,
) -> io::Result<Vec<Iteration<D>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut iterations: Vec<Iteration<D>> = Vec::new();
    let mut current_iteration = None;
    let mut buffer = String::new();

    while let Ok(bytes_read) = reader.read_line(&mut buffer) {
        if bytes_read == 0 {
            break;
        }
        let line = &buffer;
        if line.starts_with("ITERATION") {
            if let Some(iter) = current_iteration.take() {
                iterations.push(iter);
            }

            let num = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            // Consume the line containing the number of nodes and the dimension
            _ = reader.read_line(&mut buffer);

            current_iteration = Some(Iteration {
                number: num,
                positions: Vec::with_capacity(
                    iterations
                        .last()
                        .map(|x| x.positions.len())
                        .unwrap_or_default(),
                ),
            });
        } else if line.starts_with("---") {
            // End of positions block
        } else if let Some(ref mut iter) = current_iteration {
            let mut parts = line.split_ascii_whitespace();

            let index = parts.next().unwrap().parse::<usize>().unwrap_or(0);
            let weight = parts.next().unwrap().parse::<f64>().unwrap_or(0.0);
            let dim = parts.next().unwrap().parse::<usize>().unwrap_or(0);
            if dim != D {
                panic!("Graph dimension from data file did not match the compiled graph dimension");
            }

            let coordinates =
                DVec::from_fn(|_| {
                    parts.next().expect(
                    "Graph dimension from data file did not match the compiled graph dimension")
                        .parse::<f32>()
                        .unwrap()
                });

            iter.positions.push(Position {
                index,
                weight,
                coordinates,
            });
        }
        buffer.clear()
    }

    // Don't forget the last iteration
    if let Some(iter) = current_iteration {
        iterations.push(iter);
    }

    Ok(iterations)
}

impl<const D: usize> Iteration<D> {
    pub fn coordinates(&self) -> impl Iterator<Item = DVec<D>> {
        self.positions.iter().map(|x| x.coordinates)
    }
}
