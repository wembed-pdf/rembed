use memmap::MmapOptions;

use crate::dvec::DVec;
use core::panic;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

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
    let mut iterations: Vec<Iteration<D>> = Vec::new();

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    // Read header: n (nodes) and dim (dimensions)
    let (buffer, mmap) = mmap.split_first_chunk().unwrap();
    let n = u64::from_le_bytes(*buffer) as usize;

    let (buffer, mmap) = mmap.split_first_chunk().unwrap();
    let dim = u64::from_le_bytes(*buffer) as usize;

    if dim != D {
        panic!(
            "Graph dimension from data file ({}) did not match the compiled graph dimension ({})",
            dim, D
        );
    }
    let mut mmap = mmap;

    // Read iterations until EOF
    while !mmap.is_empty() {
        let (buffer, new_mmap) = mmap.split_first_chunk().unwrap();
        let iteration_number = u64::from_le_bytes(*buffer) as usize;
        println!("iteration {}", iteration_number);
        mmap = new_mmap;

        let mut positions = Vec::with_capacity(n);

        // Read position data
        for index in 0..n {
            let coordinates = DVec::from_fn(|_| {
                let (coord_buffer, new_mmap) = mmap.split_first_chunk().unwrap();
                mmap = new_mmap;
                f32::from_le_bytes(*coord_buffer)
            });

            positions.push(Position {
                index,
                weight: 1.0, // Default weight since it's not stored in binary format
                coordinates,
            });
        }

        iterations.push(Iteration {
            number: iteration_number,
            positions,
        });
    }

    Ok(iterations)
}

impl<const D: usize> Iteration<D> {
    pub fn coordinates(&self) -> impl Iterator<Item = DVec<D>> + '_ {
        self.positions.iter().map(|x| x.coordinates)
    }
}
