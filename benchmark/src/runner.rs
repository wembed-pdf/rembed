use criterion::Criterion;
use rembed::{Embedding, lsh::Lsh, query::Query};
use tracing::instrument::WithSubscriber;

fn data_structures<'a, 'b, const D: usize>(
    embedding: &Embedding<'a, D>,
) -> impl Iterator<Item = Box<dyn Query + 'a>> {
    [
        Box::new(embedding.clone()) as Box<dyn Query + 'a>,
        Box::new(Lsh::<D>::new(embedding.clone())) as Box<dyn Query + 'a>,
    ]
    .into_iter()
}

fn profile_datastructures<'a, const D: usize>(embedding: &Embedding<'a, D>, c: &mut Criterion) {
    for structure in data_structures(embedding) {
        c.bench_function(&structure.name(), |b| {
            b.iter(|| {
                for i in 0..1000 {
                    structure.nearest_neighbors(i * embedding.positions.len() / 1000, 1.);
                }
            });
        });
    }
}
