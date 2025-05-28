use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use uwuembed::{lsh::Lsh, query::Embedder, *};

fn criterion_benchmark(c: &mut Criterion) {
    let graph_name = "bio-grid-fruitfly";

    let dim = 8;
    let dim_hint = 8;

    let (graph, iterations) = load_graph(graph_name, dim, dim_hint).unwrap();

    let embeddings: Vec<Embedding<8>> = iterations
        .iter()
        .map(|x| Embedding {
            positions: x.coordinates().collect(),
            graph: &graph,
        })
        .collect();

    let iter = &embeddings[200];
    let naive = iter;
    let lsh = Lsh::new(iter.clone());
    c.bench_function("biogrid-fruitfly-naive", |b| {
        b.iter(|| {
            let sum: usize = black_box(
                (0..(iter.positions.len()))
                    .step_by(10)
                    .map(|i| naive.repelling_nodes(i).len())
                    .sum::<usize>(),
            );
            assert!(sum > 0);
        })
    });
    c.bench_function("biogrid-fruitfly-lsh", |b| {
        b.iter(|| {
            let sum: usize = black_box(
                (0..(iter.positions.len()))
                    .step_by(10)
                    .map(|i| lsh.repelling_nodes(i).len())
                    .sum::<usize>(),
            );
            assert!(sum > 0);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
