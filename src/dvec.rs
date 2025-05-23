use std::fmt;
use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DVec<const D: usize> {
    pub components: [f64; D],
}

impl<const D: usize> Default for DVec<D> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize> DVec<D> {
    pub const fn new(components: [f64; D]) -> Self {
        Self { components }
    }

    pub const fn zero() -> Self {
        Self {
            components: [0.0; D],
        }
    }

    pub const fn unit(direction: usize) -> Self {
        assert!(direction < D, "Direction index out of bounds");
        let mut components = [0.0; D];
        components[direction] = 1.0;
        Self { components }
    }

    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> f64,
    {
        Self {
            components: std::array::from_fn(f),
        }
    }

    // Euclidean/L2 norm
    pub fn magnitude(&self) -> f64 {
        self.components.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.components.iter().map(|&x| x * x).sum()
    }

    // Manhattan/L1 norm
    pub fn manhattan_norm(&self) -> f64 {
        self.components.iter().map(|&x| x.abs()).sum()
    }

    // Infinity/L∞ norm
    pub fn infinity_norm(&self) -> f64 {
        self.components.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    // General Lp norm
    pub fn lp_norm(&self, p: f64) -> f64 {
        assert!(p > 0.0, "p must be positive for Lp norm");
        if p == 1.0 {
            return self.manhattan_norm();
        }
        if p == 2.0 {
            return self.magnitude();
        }
        if p.is_infinite() && p.is_sign_positive() {
            return self.infinity_norm();
        }

        self.components
            .iter()
            .map(|&x| x.abs().powf(p))
            .sum::<f64>()
            .powf(1.0 / p)
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        assert!(mag > 0.0, "Cannot normalize a zero vector");
        *self / mag
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    // Distance functions
    pub fn distance(&self, other: &Self) -> f64 {
        (*self - *other).magnitude()
    }
    pub fn distance_squared(&self, other: &Self) -> f64 {
        (*self - *other).magnitude_squared()
    }

    pub fn manhattan_distance(&self, other: &Self) -> f64 {
        (*self - *other).manhattan_norm()
    }

    pub fn infinity_distance(&self, other: &Self) -> f64 {
        (*self - *other).infinity_norm()
    }

    pub fn lp_distance(&self, other: &Self, p: f64) -> f64 {
        (*self - *other).lp_norm(p)
    }

    pub fn angle(&self, other: &Self) -> f64 {
        let dot_product = self.dot(other);
        let magnitudes = self.magnitude() * other.magnitude();

        if magnitudes == 0.0 {
            0.0 // Return 0 for zero vectors (convention)
        } else {
            let cosine = dot_product / magnitudes;
            let clamped_cosine = cosine.clamp(-1.0, 1.0);
            clamped_cosine.acos()
        }
    }

    pub fn truncate<const N: usize>(&self) -> DVec<N> {
        assert!(N <= D, "Cannot truncate to a larger dimension");
        let mut result = [0.0; N];
        result[..N].copy_from_slice(&self.components[..N]);
        DVec::<N> { components: result }
    }

    pub fn extend<const N: usize>(&self) -> DVec<N> {
        assert!(N >= D, "Cannot extend to a smaller dimension");
        let mut result = [0.0; N];
        result[..N].copy_from_slice(&self.components[..N]);
        DVec::<N> { components: result }
    }

    #[inline]
    pub fn map<F>(&self, mut f: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        let mut result = [0.0; D];
        for (item, component) in result.iter_mut().zip(self.components.into_iter()) {
            *item = f(component);
        }
        Self { components: result }
    }

    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }
}

// From implementations
impl<const D: usize> From<[f64; D]> for DVec<D> {
    fn from(components: [f64; D]) -> Self {
        Self { components }
    }
}

// For 2D vector
impl From<(f64, f64)> for DVec<2> {
    fn from((x, y): (f64, f64)) -> Self {
        Self { components: [x, y] }
    }
}

// For 3D vector
impl From<(f64, f64, f64)> for DVec<3> {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self {
            components: [x, y, z],
        }
    }
}

// For 4D vector
impl From<(f64, f64, f64, f64)> for DVec<4> {
    fn from((x, y, z, w): (f64, f64, f64, f64)) -> Self {
        Self {
            components: [x, y, z, w],
        }
    }
}

// Operators
impl<const D: usize> Add for DVec<D> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut result = [0.0; D];
        for (i, item) in result.iter_mut().enumerate() {
            *item = self.components[i] + other.components[i];
        }
        Self { components: result }
    }
}

impl<const D: usize> AddAssign for DVec<D> {
    fn add_assign(&mut self, other: Self) {
        for i in 0..D {
            self.components[i] += other.components[i];
        }
    }
}

impl<const D: usize> Sub for DVec<D> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut result = [0.0; D];
        for (i, item) in result.iter_mut().enumerate() {
            *item = self.components[i] - other.components[i];
        }
        Self { components: result }
    }
}

impl<const D: usize> SubAssign for DVec<D> {
    fn sub_assign(&mut self, other: Self) {
        for i in 0..D {
            self.components[i] -= other.components[i];
        }
    }
}

impl<const D: usize> Mul<f64> for DVec<D> {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        self.map(|x| x * scalar)
    }
}

impl<const D: usize> Mul<DVec<D>> for f64 {
    type Output = DVec<D>;

    fn mul(self, vector: DVec<D>) -> Self::Output {
        vector * self
    }
}

impl<const D: usize> MulAssign<f64> for DVec<D> {
    fn mul_assign(&mut self, scalar: f64) {
        for i in 0..D {
            self.components[i] *= scalar;
        }
    }
}

impl<const D: usize> Div<f64> for DVec<D> {
    type Output = Self;

    fn div(self, scalar: f64) -> Self::Output {
        assert!(scalar != 0.0, "Division by zero");
        let divisor_recip = scalar.recip();
        self.map(|x| x * divisor_recip)
    }
}

impl<const D: usize> DivAssign<f64> for DVec<D> {
    fn div_assign(&mut self, scalar: f64) {
        assert!(scalar != 0.0, "Division by zero");
        for i in 0..D {
            self.components[i] /= scalar;
        }
    }
}

impl<const D: usize> Neg for DVec<D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

// Indexing
impl<const D: usize> Index<usize> for DVec<D> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.components[index]
    }
}

impl<const D: usize> IndexMut<usize> for DVec<D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.components[index]
    }
}

// Display
impl<const D: usize> fmt::Display for DVec<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, component) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", component)?;
        }
        write!(f, ")")
    }
}

// Sum trait implementation
impl<const D: usize> Sum for DVec<D> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}
