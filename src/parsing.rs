use memmap::{Mmap, MmapOptions};

use crate::dvec::DVec;
use core::panic;
use std::fs::File;
use std::io::{self};
use std::mem::ManuallyDrop;
use std::path::Path;

#[derive(Debug)]
pub struct Iteration<const D: usize> {
    pub number: usize,
    pub positions: ManuallyDrop<Vec<DVec<D>>>,
}
pub struct Iterations<const D: usize>(Vec<Iteration<D>>, Option<ManuallyDrop<Mmap>>);

pub fn parse_positions_file<P: AsRef<Path>, const D: usize>(path: P) -> io::Result<Iterations<D>> {
    let file = File::open(path)?;
    let mut iterations: Vec<Iteration<D>> = Vec::new();

    let original_mmap = ManuallyDrop::new(unsafe { MmapOptions::new().map(&file)? });
    // Read header: n (nodes) and dim (dimensions)
    let (buffer, mmap) = original_mmap.split_first_chunk().unwrap();
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
        mmap = new_mmap;

        let byte_size = n * D * core::mem::size_of::<f32>();
        assert!(mmap.len() >= byte_size, "len: {}", mmap.len());
        let (iteration, new_mmap) = mmap.split_at(byte_size);
        mmap = new_mmap;

        // Read position data
        let buffer = unsafe { Vec::from_raw_parts(iteration.as_ptr() as *mut DVec<D>, n, n) };

        iterations.push(Iteration {
            number: iteration_number,
            positions: ManuallyDrop::new(buffer),
        });
    }

    Ok(Iterations(iterations, Some(original_mmap)))
}

impl<const D: usize> Drop for Iterations<D> {
    fn drop(&mut self) {
        ManuallyDrop::into_inner(self.1.take().unwrap());
    }
}

impl<const D: usize> Iterations<D> {
    pub fn iterations(&self) -> &[Iteration<D>] {
        self.0.as_slice()
    }
}

pub fn write_test_file<const D: usize>(
    file_path: &str,
    iterations: &[(u64, Vec<DVec<D>>)],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufWriter, Write};
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    // Write number of nodes
    if let Some(first_iteration) = iterations.first() {
        let num_nodes = first_iteration.1.len() as u64;
        writer.write_all(&num_nodes.to_le_bytes())?;
        writer.write_all(&(D as u64).to_le_bytes())?;
    } else {
        return Err("No iterations found".into());
    }

    // Write iterations
    for (num, iteration) in iterations {
        writer.write_all(&num.to_le_bytes())?;
        for position in iteration {
            // Write neighbor IDs as u32
            for i in 0..D {
                writer.write_all(&(position[i]).to_le_bytes())?;
            }
        }
    }
    Ok(())
}
