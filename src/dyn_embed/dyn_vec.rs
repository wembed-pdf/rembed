use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use crate::dvec::Vector;

/// A heap-allocated vector of arbitrary dimension.
///
/// Runtime equivalent of `DVec<D>`, implementing the [`Vector`] trait so it
/// can be used with the generic embedder.
#[derive(Clone, Debug, PartialEq)]
pub struct DynVec {
    pub components: Vec<f32>,
}

impl DynVec {
    pub fn new(components: Vec<f32>) -> Self {
        Self { components }
    }
}

impl Vector for DynVec {
    fn zero(dim: usize) -> Self {
        Self {
            components: vec![0.0; dim],
        }
    }

    fn from_fn(dim: usize, f: impl FnMut(usize) -> f32) -> Self {
        Self {
            components: (0..dim).map(f).collect(),
        }
    }

    fn magnitude(&self) -> f32 {
        self.components.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    fn magnitude_squared(&self) -> f32 {
        self.components.iter().map(|&x| x * x).sum()
    }

    fn distance_squared(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.components.len(), other.components.len());
        self.components
            .iter()
            .zip(&other.components)
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    fn map(&self, mut f: impl FnMut(f32) -> f32) -> Self {
        Self {
            components: self.components.iter().map(|&x| f(x)).collect(),
        }
    }

    fn dim(&self) -> usize {
        self.components.len()
    }
}

impl Add for DynVec {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        for (a, b) in self.components.iter_mut().zip(&other.components) {
            *a += b;
        }
        self
    }
}

impl AddAssign for DynVec {
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.components.iter_mut().zip(&other.components) {
            *a += b;
        }
    }
}

impl Sub for DynVec {
    type Output = Self;
    fn sub(mut self, other: Self) -> Self {
        for (a, b) in self.components.iter_mut().zip(&other.components) {
            *a -= b;
        }
        self
    }
}

impl SubAssign for DynVec {
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.components.iter_mut().zip(&other.components) {
            *a -= b;
        }
    }
}

impl Mul<f32> for DynVec {
    type Output = Self;
    fn mul(mut self, scalar: f32) -> Self {
        for a in &mut self.components {
            *a *= scalar;
        }
        self
    }
}

impl Mul<DynVec> for f32 {
    type Output = DynVec;
    fn mul(self, vec: DynVec) -> DynVec {
        vec * self
    }
}

impl Mul<DynVec> for DynVec {
    type Output = Self;
    fn mul(mut self, other: DynVec) -> Self {
        for (a, b) in self.components.iter_mut().zip(&other.components) {
            *a *= b;
        }
        self
    }
}

impl Div<f32> for DynVec {
    type Output = Self;
    fn div(mut self, scalar: f32) -> Self {
        assert!(scalar != 0.0, "Division by zero");
        let recip = scalar.recip();
        for a in &mut self.components {
            *a *= recip;
        }
        self
    }
}

impl Div<DynVec> for DynVec {
    type Output = Self;
    fn div(mut self, other: DynVec) -> Self {
        for (a, b) in self.components.iter_mut().zip(&other.components) {
            *a /= b;
        }
        self
    }
}

impl Neg for DynVec {
    type Output = Self;
    fn neg(mut self) -> Self {
        for a in &mut self.components {
            *a = -*a;
        }
        self
    }
}

impl Index<usize> for DynVec {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        &self.components[index]
    }
}

impl IndexMut<usize> for DynVec {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.components[index]
    }
}

impl std::fmt::Display for DynVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{c}")?;
        }
        write!(f, ")")
    }
}
