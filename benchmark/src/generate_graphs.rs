use std::process::Command;

use indicatif::{ProgressBar, ProgressStyle};

struct Seed {
    wseed: i32, // weight seed default 12
    pseed: i32, // position seed default 130
    sseed: i32, // sampling seed default 1400
}

pub struct GraphGenerator {
    pub girgs_path: String,
    pub output_path: String,
}

impl GraphGenerator {
    pub fn new(girgs_path: String, output_path: String) -> Self {
        Self {
            girgs_path,
            output_path,
        }
    }

    pub fn generate(&self) {
        println!("Generating graphs using GIRGs at: {}", self.girgs_path);
        println!("Output will be saved to: {}", self.output_path);
        let n_s = half_log10(1000.0, 1000000.0);
        let seeds = generate_seeds();
        let avg_degrees = generate_avg_degrees();
        let pb = ProgressBar::new(n_s.len() as u64 * seeds.len() as u64 * avg_degrees.len() as u64);
        for seed in seeds {
            for &avg_degree in &avg_degrees {
                for &n in &n_s {
                    Command::new(&self.girgs_path)
                        .stdout(std::process::Stdio::null())
                        .arg("-n")
                        .arg(n.to_string())
                        .arg("-deg")
                        .arg(avg_degree.to_string())
                        .arg("-file")
                        .arg(format!(
                            "{}/genhrg_n-{}_deg-{}_wseed-{}_pseed-{}_sseed-{}",
                            self.output_path, n, avg_degree, seed.wseed, seed.pseed, seed.sseed
                        ))
                        .arg("-edge")
                        .arg("1")
                        .arg("-wseed")
                        .arg(seed.wseed.to_string())
                        .arg("-pseed")
                        .arg(seed.pseed.to_string())
                        .arg("-sseed")
                        .arg(seed.sseed.to_string())
                        .status()
                        .expect("Failed to execute GIRGs command");
                    pb.inc(1);
                }
            }
        }
        pb.finish_with_message("Graph generation complete");
    }
}

fn half_log10(start: f64, end: f64) -> Vec<i32> {
    // Generate successive values by multiplying by √10 and rounding.
    (0..)
        .scan(start, |state, _| {
            if *state > end {
                return None;
            }
            let current = *state as i32;
            *state *= 10f64.powf(0.5); // multiply by √10 ≈ 3.16227766
            Some(current)
        })
        .collect()
}

fn generate_seeds() -> Vec<Seed> {
    // Generate seeds for the graphs
    let mut seeds = Vec::new();
    for i in 0..3 {
        seeds.push(Seed {
            wseed: 12 + i as i32,
            pseed: 130 + i as i32,
            sseed: 1400 + i as i32,
        });
    }
    seeds
}

fn generate_avg_degrees() -> Vec<i32> {
    let mut avg_degrees = Vec::new();
    for i in (5..20).step_by(5) {
        avg_degrees.push(i);
    }
    avg_degrees
}
