use sprk::simd::PDVec;
use rand::random;
use std::io;
use std::time::{Duration, Instant};
const W: usize = 8;
fn main() -> io::Result<()> {
    let query = std::hint::black_box([0.; 500]);
    // bench_snn_single(query);
    // bench_snn(query);
    // bench_snn_4(query);
    // return Ok(());
    // CSV header with metadata columns
    let results = vec![
        bench_dim::<2>(),
        bench_dim::<3>(),
        bench_dim::<4>(),
        bench_dim::<5>(),
        bench_dim::<6>(),
        bench_dim::<7>(),
        bench_dim::<8>(),
        bench_dim::<9>(),
        bench_dim::<10>(),
        bench_dim::<11>(),
        bench_dim::<12>(),
        bench_dim::<13>(),
        bench_dim::<14>(),
        bench_dim::<15>(),
        bench_dim::<16>(),
        bench_dim::<17>(),
        bench_dim::<18>(),
        bench_dim::<19>(),
        bench_dim::<20>(),
        // bench_dim::<21>(),
        // bench_dim::<22>(),
        // bench_dim::<23>(),
        // bench_dim::<24>(),
        bench_dim::<25>(),
        // bench_dim::<26>(),
        // bench_dim::<27>(),
        // bench_dim::<28>(),
        // bench_dim::<29>(),
        bench_dim::<30>(),
        // bench_dim::<31>(),
        // bench_dim::<32>(),
        // bench_dim::<33>(),
        // bench_dim::<34>(),
        // bench_dim::<35>(),
        // bench_dim::<36>(),
        // bench_dim::<37>(),
        // bench_dim::<38>(),
        // bench_dim::<39>(),
        bench_dim::<40>(),
        bench_dim::<50>(),
        bench_dim::<60>(),
        bench_dim::<70>(),
        bench_dim::<80>(),
        bench_dim::<90>(),
        bench_dim::<100>(),
        // bench_dim::<120>(),
        // bench_dim::<140>(),
        // bench_dim::<160>(),
        // bench_dim::<180>(),
        // bench_dim::<200>(),
        // bench_dim::<250>(),
        // bench_dim::<300>(),
        // bench_dim::<350>(),
        // bench_dim::<400>(),
        // bench_dim::<500>(),
        // bench_dim::<600>(),
        // bench_dim::<700>(),
        // bench_dim::<800>(),
        // bench_dim::<900>(),
    ];

    // println!("d, no_fma, direct, snn");
    for result in results {
        println!(
            "{}, {:?}, {:?}, {:?}, {}, {}, {}",
            result.dim,
            result.no_fma.as_micros(),
            result.direct.as_micros(),
            result.snn_1.as_micros(),
            result.snn_2.as_micros(),
            result.snn_4.as_micros(),
            result.snn_8.as_micros(),
        );
    }

    // println!("{:#?}", results);

    Ok(())
}
#[derive(Debug)]
struct Timings {
    dim: usize,
    no_fma: Duration,
    direct: Duration,
    snn_2: Duration,
    snn_1: Duration,
    snn_4: Duration,
    snn_8: Duration,
}

fn bench_dim<const D: usize>() -> Timings {
    let query = std::hint::black_box([0.; D]);
    eprintln!("benching dim: {D}");
    let no_fma = bench_no_fma(query);
    let direct = bench_direct(query);
    let snn_1 = bench_snn_single(query);
    let snn_2 = bench_snn(query);
    let snn_4 = bench_snn_4(query);
    let snn_8 = bench_snn_8(query);

    Timings {
        dim: D,
        no_fma,
        direct,
        snn_2,
        snn_1,
        snn_4,
        snn_8,
    }
}
const REP: usize = 1000000;

#[inline(never)]
fn bench_no_fma<const D: usize>(query: [f32; D]) -> Duration {
    let (pdvecs, start) = (generate_pvecs::<D>(), Instant::now());
    for _ in 0..REP {
        for vec in &pdvecs {
            std::hint::black_box(vec.dist_squared_no_fma(query));
        }
    }
    start.elapsed()
}

#[inline(never)]
fn bench_direct<const D: usize>(query: [f32; D]) -> Duration {
    let (pdvecs, start) = (generate_pvecs::<D>(), Instant::now());
    for _ in 0..REP {
        for vec in &pdvecs {
            std::hint::black_box(vec.dist_squared(query));
        }
    }
    start.elapsed()
}

#[inline(never)]
fn bench_snn<const D: usize>(query: [f32; D]) -> Duration {
    let pdvecs = generate_pvecs::<D>();
    let start = Instant::now();
    for _ in 0..REP {
        for vec in &pdvecs {
            std::hint::black_box(vec.dist_half_squared(query, std::hint::black_box(1000.)));
        }
    }
    start.elapsed()
}

#[inline(never)]
fn bench_snn_single<const D: usize>(query: [f32; D]) -> Duration {
    let pdvecs = generate_pvecs::<D>();
    let start = Instant::now();
    for _ in 0..REP {
        for vec in &pdvecs {
            std::hint::black_box(
                vec.dist_half_squared_single_acc(query, std::hint::black_box(1000.)),
            );
        }
    }
    start.elapsed()
}

#[inline(never)]
fn bench_snn_4<const D: usize>(query: [f32; D]) -> Duration {
    let pdvecs = generate_pvecs::<D>();
    let start = Instant::now();
    for _ in 0..REP {
        for vec in &pdvecs {
            std::hint::black_box(vec.dist_half_squared_4_acc(query, std::hint::black_box(1000.)));
        }
    }
    start.elapsed()
}

#[inline(never)]
fn bench_snn_8<const D: usize>(query: [f32; D]) -> Duration {
    let pdvecs = generate_pvecs::<D>();
    let start = Instant::now();
    for _ in 0..REP {
        for vec in &pdvecs {
            std::hint::black_box(
                vec.dist_half_squared_unrolled(query, std::hint::black_box(1000.)),
            );
        }
    }
    start.elapsed()
}

fn generate_pvecs<const D: usize>() -> Vec<PDVec<D, W, f32, u32>> {
    (0..128)
        .step_by(W)
        .map(|i| {
            // let random_arr: [f32; W] = std::array::from_fn(|_| random());
            PDVec::new((0..W).map(|j| (std::array::from_fn(|_| random()), i + j)))
        })
        .collect()
}
