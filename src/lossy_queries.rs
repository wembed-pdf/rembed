use rand::{rngs::SmallRng, seq::SliceRandom};

use crate::{
    NodeId, Query,
    dvec::DVec,
    query::{self, Embedder, Graph, Position, SpatialIndex, Update},
};

#[derive(Clone, Copy, Debug)]
pub enum LossyStrategy {
    Random,
    InOrder,
    ID,
    Droplist,
    Closest,
    Furthest,
    Heavy,
    Light,
}

#[derive(Clone)]
pub struct LossyQuery<'a, const D: usize, ID: Embedder<'a, D>> {
    recall: f64,
    loss_strategy: LossyStrategy,
    structure: ID,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, const D: usize, ID: Embedder<'a, D>> LossyQuery<'a, D, ID> {
    pub fn new(embedding: &crate::Embedding<'a, D>, recall: f64, strategy: LossyStrategy) -> Self {
        let mut query = LossyQuery {
            recall,
            loss_strategy: strategy,
            structure: ID::new(embedding),
            _phantom: std::marker::PhantomData,
        };
        query.update_positions(&embedding.positions, None);
        query
    }
}

impl<'a, const D: usize, ID: Embedder<'a, D>> crate::query::Graph for LossyQuery<'a, D, ID> {
    fn is_connected(&self, first: NodeId, second: NodeId) -> bool {
        self.structure.is_connected(first, second)
    }

    fn neighbors(&self, index: NodeId) -> &[NodeId] {
        self.structure.neighbors(index)
    }

    fn weight(&self, index: NodeId) -> f64 {
        self.structure.weight(index)
    }
}

impl<'a, const D: usize, ID: Embedder<'a, D>> query::Position<D> for LossyQuery<'a, D, ID> {
    fn position(&self, index: NodeId) -> &DVec<D> {
        self.structure.position(index)
    }

    fn num_nodes(&self) -> usize {
        self.structure.num_nodes()
    }
}
impl<'a, const D: usize, ID: Embedder<'a, D>> query::Update<D> for LossyQuery<'a, D, ID> {
    fn update_positions(&mut self, positions: &[DVec<D>], last_delta: Option<f64>) {
        self.structure.update_positions(positions, last_delta);
    }
}

impl<'a, const D: usize, ID: Embedder<'a, D>> Query for LossyQuery<'a, D, ID> {
    fn nearest_neighbors(&self, index: usize, radius: f64, results: &mut Vec<usize>) {
        self.structure.nearest_neighbors(index, radius, results)
    }
}
impl<'a, const D: usize, ID: Embedder<'a, D>> SpatialIndex<D> for LossyQuery<'a, D, ID> {
    fn name(&self) -> String {
        String::from("lossy queries")
    }
    fn implementation_string(&self) -> &'static str {
        include_str!("lossy_queries.rs")
    }
}

impl<'a, const D: usize, ID: Embedder<'a, D>> query::Embedder<'a, D> for LossyQuery<'a, D, ID> {
    fn new(embedding: &crate::Embedding<'a, D>) -> Self {
        let mut query = LossyQuery {
            recall: 1.0,
            loss_strategy: LossyStrategy::Random,
            structure: ID::new(embedding),
            _phantom: std::marker::PhantomData,
        };
        query.update_positions(&embedding.positions, None);
        query
    }
    fn repelling_nodes(&self, index: usize, result: &mut Vec<NodeId>) {
        self.nearest_neighbors(index, 1., result);

        // Only keep true positive repelling nodes
        let pos = self.position(index);
        let weight = self.weight(index);

        result.retain(|&x| {
            index != x
                && !self.is_connected(index, x)
                && (weight > self.weight(x) || (weight == self.weight(x) && index > x))
                && (self.position(x).distance_squared(pos) as f64)
                    < (weight * self.weight(x)).powi(2)
        });

        match self.loss_strategy {
            LossyStrategy::Closest | LossyStrategy::Furthest => {
                result.sort_by(|a, b| {
                    self.position(*a)
                        .distance_squared(self.position(index))
                        .total_cmp(&self.position(*b).distance_squared(self.position(index)))
                });
            }
            LossyStrategy::Heavy | LossyStrategy::Light => {
                result.sort_by(|a, b| self.weight(*a).total_cmp(&self.weight(*b)));
            }
            LossyStrategy::Random => {
                result.shuffle(&mut rand::rng());
            }
            LossyStrategy::Droplist => {
                let mut droplist: Vec<_> = (0..self.num_nodes()).collect();
                let mut rng: SmallRng = rand::SeedableRng::seed_from_u64(index as u64);
                droplist.shuffle(&mut rng);
                result.sort_by_key(|id| droplist.iter().position(|x| x == id));
            }
            _ => (),
        };
        match self.loss_strategy {
            LossyStrategy::Closest | LossyStrategy::Light | LossyStrategy::Droplist => {
                result.reverse()
            }
            _ => (),
        }
        result.truncate((result.len() as f64 * self.recall).round() as usize);
    }
}
