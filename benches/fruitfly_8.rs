use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use uwuembed::{query::Embedder, *};

fn criterion_benchmark(c: &mut Criterion) {
    let graph = graph::Graph::parse_from_edge_list_file("bio-grid-fruitfly", 8, 8).unwrap();

    // let positions_path = "positions.log";
    let positions_path = "data/bio-grid-fruitfly/positions_8_8.log";

    let iterations = parsing::parse_positions_file(positions_path).unwrap();
    let embeddings: Vec<Embedding<8>> = iterations
        .iter()
        .map(|x| Embedding {
            positions: x.coordinates().collect(),
            graph: &graph,
        })
        .collect();

    let iter = &embeddings[200];
    let naive = iter;
    c.bench_function("biogrid-fruitfly", |b| {
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
