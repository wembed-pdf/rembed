use criterion::{BenchmarkGroup, measurement::WallTime};
use rembed::{Embedding, lsh::Lsh, query::Query};

fn data_structures<'a, const D: usize>(
    embedding: &Embedding<'a, D>,
) -> impl Iterator<Item = Box<dyn Query + 'a>> {
    [
        Box::new(embedding.clone()) as Box<dyn Query + 'a>,
        Box::new(Lsh::<D>::new(embedding.clone())) as Box<dyn Query + 'a>,
        Box::new(rembed::wrtree::WRTree::<D>::new(embedding.clone())) as Box<dyn Query + 'a>,
    ]
    .into_iter()
}

pub fn profile_datastructures<'a, const D: usize>(
    embedding: &Embedding<'a, D>,
    c: &mut BenchmarkGroup<WallTime>,
) {
    for structure in data_structures(embedding) {
        c.bench_function(structure.name(), |b| {
            b.iter(|| {
                for i in 0..1000 {
                    structure.nearest_neighbors(i * embedding.positions.len() / 1000, 1.);
                }
            });
        });
    }
}
