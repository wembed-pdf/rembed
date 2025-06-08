use perf_event::events::Hardware;
use perf_event::{Builder, Counter, Group};
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone, Copy)]
pub struct PerfMeasurement {
    pub wall_time: Duration,
    pub instructions: u64,
    pub cycles: u64,
    pub iterations: u64,
}

impl PerfMeasurement {
    pub fn new(wall_time: Duration, instructions: u64, cycles: u64, iterations: u64) -> Self {
        Self {
            wall_time,
            instructions,
            cycles,
            iterations,
        }
    }
}

pub struct PerfCounter {
    start_time: Instant,
    perf_group: Group,
    instruction_counter: Counter,
    cycles_counter: Counter,
}

impl Default for PerfCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfCounter {
    fn create_perf_group() -> Result<(Group, Counter, Counter), Box<dyn std::error::Error>> {
        let mut group = Group::new()?;

        // Add instruction counter
        let instructions = group.add(&Builder::new(Hardware::INSTRUCTIONS))?;

        // Add cycles counter
        let cycles = group.add(&Builder::new(Hardware::CPU_CYCLES))?;

        Ok((group, instructions, cycles))
    }
}

impl PerfCounter {
    pub fn new() -> Self {
        let (mut perf_group, instructions, cycles) =
            Self::create_perf_group().expect("Failed to create perf event group");

        perf_group
            .disable()
            .expect("Failed to enable perf counters");
        let start_time = Instant::now();

        PerfCounter {
            start_time,
            perf_group,
            instruction_counter: instructions,
            cycles_counter: cycles,
        }
    }
    pub fn start(&mut self) {
        self.perf_group.enable().unwrap();
        self.perf_group.reset().unwrap();
        self.start_time = Instant::now();
    }

    pub fn elapsed(&mut self, iterations: u64) -> PerfMeasurement {
        let wall_time = self.start_time.elapsed();

        let counts = self
            .perf_group
            .read()
            .expect("Failed to read perf counters");
        self.perf_group
            .disable()
            .expect("Failed to disable perf counters");

        let instructions = counts
            .get(&self.instruction_counter)
            .map(|c| c.value())
            .unwrap_or(0);
        let cycles = counts
            .get(&self.cycles_counter)
            .map(|c| c.value())
            .unwrap_or(0);

        PerfMeasurement::new(wall_time, instructions, cycles, iterations)
    }
}

#[derive(Default)]
pub struct PerfMeasurements {
    samples: Vec<PerfMeasurement>,
    profiler: PerfCounter,
}

impl PerfMeasurements {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            profiler: PerfCounter::new(),
        }
    }
    pub fn start(&mut self) {
        self.profiler.start();
    }
    pub fn stop(&mut self, iterations: u64) -> Duration {
        let elapsed = self.profiler.elapsed(iterations);
        self.samples.push(elapsed);
        elapsed.wall_time
    }
    /// Get statistics from all collected measurements
    pub fn get_statistics(&self, queries: usize, warmup: Duration) -> PerfStatistics {
        let samples = queries as u64;
        let mut measurements: Vec<_> = self
            .samples
            .iter()
            .map(|x| {
                let divisor = samples * x.iterations;
                PerfMeasurement {
                    wall_time: x.wall_time / divisor as u32,
                    instructions: x.instructions / divisor,
                    cycles: x.cycles / divisor,
                    iterations: x.iterations,
                }
            })
            .collect();

        let mut acc = Duration::default();
        measurements.retain(|x| {
            acc += x.wall_time;
            acc > warmup / (samples * x.iterations) as u32
        });

        let count = measurements.len() as f64;

        if measurements.is_empty() {
            return PerfStatistics::default();
        }

        // Calculate means
        let wall_time_mean = measurements
            .iter()
            .map(|m| m.wall_time.as_nanos() as f64)
            .sum::<f64>()
            / count;

        let instructions_mean = measurements
            .iter()
            .map(|m| m.instructions as f64)
            .sum::<f64>()
            / count;

        let cycles_mean = measurements.iter().map(|m| m.cycles as f64).sum::<f64>() / count;

        // Calculate standard deviations
        let wall_time_variance = measurements
            .iter()
            .map(|m| {
                let diff = m.wall_time.as_nanos() as f64 - wall_time_mean;
                diff * diff
            })
            .sum::<f64>()
            / count;

        let instructions_variance = measurements
            .iter()
            .map(|m| {
                let diff = m.instructions as f64 - instructions_mean;
                diff * diff
            })
            .sum::<f64>()
            / count;

        let cycles_variance = measurements
            .iter()
            .map(|m| {
                let diff = m.cycles as f64 - cycles_mean;
                diff * diff
            })
            .sum::<f64>()
            / count;

        PerfStatistics {
            wall_time_mean: Duration::from_nanos(wall_time_mean as u64),
            wall_time_stddev: Duration::from_nanos(wall_time_variance.sqrt() as u64),
            instructions_mean,
            instructions_stddev: instructions_variance.sqrt(),
            cycles_mean,
            cycles_stddev: cycles_variance.sqrt(),
        }
    }

    pub(crate) fn num_samples(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Debug, Clone, Default)]
pub struct PerfStatistics {
    pub wall_time_mean: Duration,
    pub wall_time_stddev: Duration,
    pub instructions_mean: f64,
    pub instructions_stddev: f64,
    pub cycles_mean: f64,
    pub cycles_stddev: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_perf_counter_basic() {
        let mut counter = PerfCounter::new();
        let instructions = Builder::new(Hardware::INSTRUCTIONS);

        let mut instructions = instructions.build().unwrap();
        instructions.enable().unwrap();
        // Simulate a measurement
        thread::sleep(Duration::from_millis(10));
        let result = counter.elapsed(1);

        assert!(result.wall_time > Duration::ZERO);
        assert!(result.instructions > 0);
        // Cycles might be 0 in some virtualized environments, so we don't assert on it
    }
}
