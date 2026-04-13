use std::fmt;
use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Trait abstracting over fixed-size `DVec<D>` and heap-allocated `DynVec`.
///
/// Enables the embedder and optimizer to be generic over the vector
/// representation, so the same algorithm works with compile-time and
/// runtime dimensionality.
pub trait Vector:
    Clone
    + Send
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Neg<Output = Self>
    + fmt::Debug
{
    fn zero(dim: usize) -> Self;
    fn from_fn(dim: usize, f: impl FnMut(usize) -> f32) -> Self;
    fn magnitude(&self) -> f32;
    fn magnitude_squared(&self) -> f32;
    fn distance_squared(&self, other: &Self) -> f32;
    fn map(&self, f: impl FnMut(f32) -> f32) -> Self;
    fn dim(&self) -> usize;
}

impl<const D: usize> Vector for DVec<D> {
    fn zero(dim: usize) -> Self {
        debug_assert_eq!(dim, D);
        Self::zero()
    }

    fn from_fn(dim: usize, f: impl FnMut(usize) -> f32) -> Self {
        debug_assert_eq!(dim, D);
        Self::from_fn(f)
    }

    fn magnitude(&self) -> f32 {
        self.magnitude()
    }

    fn magnitude_squared(&self) -> f32 {
        self.magnitude_squared()
    }

    fn distance_squared(&self, other: &Self) -> f32 {
        self.distance_squared(other)
    }

    fn map(&self, f: impl FnMut(f32) -> f32) -> Self {
        self.map(f)
    }

    fn dim(&self) -> usize {
        D
    }
}

#[derive(Clone, Copy, Debug, PartialOrd)]
#[repr(transparent)]
pub struct DVec<const D: usize> {
    pub components: [f32; D],
}

impl<const D: usize> std::hash::Hash for DVec<D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for &component in &self.components {
            // Hash the bit representation of f32
            state.write_u32(component.to_bits());
        }
    }
}

impl<const D: usize> Eq for DVec<D> {}

impl<const D: usize> PartialEq for DVec<D> {
    fn eq(&self, other: &Self) -> bool {
        self.components
            .iter()
            .zip(other.components.iter())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
    }
}

impl<const D: usize> Default for DVec<D> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize> DVec<D> {
    pub const fn new(components: [f32; D]) -> Self {
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
    pub fn units(direction_mask: usize) -> Self {
        let mut unit = DVec::zero();
        for i in 0..D {
            if direction_mask & 1 << i != 0 {
                unit += DVec::unit(i);
            }
        }
        unit
    }

    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> f32,
    {
        Self {
            components: std::array::from_fn(f),
        }
    }

    // Euclidean/L2 norm
    pub fn magnitude(&self) -> f32 {
        self.components.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    pub fn magnitude_squared(&self) -> f32 {
        self.components.iter().map(|&x| x * x).sum()
    }

    // Manhattan/L1 norm
    pub fn manhattan_norm(&self) -> f32 {
        self.components.iter().map(|&x| x.abs()).sum()
    }

    // Infinity/L∞ norm
    pub fn infinity_norm(&self) -> f32 {
        self.components.iter().map(|&x| x.abs()).fold(0.0, f32::max)
    }

    // General Lp norm
    pub fn lp_norm(&self, p: f32) -> f32 {
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
            .sum::<f32>()
            .powf(1.0 / p)
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        assert!(mag > 0.0, "Cannot normalize a zero vector");
        *self / mag
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    // Distance functions
    pub fn distance(&self, other: &Self) -> f32 {
        (*self - *other).magnitude()
    }
    // #[inline(never)]
    pub fn distance_squared(&self, other: &Self) -> f32 {
        let a = &self.components;
        let b = &other.components;
        if D.is_multiple_of(4) {
            let mut acc = [0.0f32; 4];
            let chunks = D / 4;
            for i in 0..chunks {
                let base = i * 4;
                let d0 = a[base] - b[base];
                let d1 = a[base + 1] - b[base + 1];
                let d2 = a[base + 2] - b[base + 2];
                let d3 = a[base + 3] - b[base + 3];
                acc[0] += d0 * d0;
                acc[1] += d1 * d1;
                acc[2] += d2 * d2;
                acc[3] += d3 * d3;
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else if D % 4 == 2 {
            let mut acc = [0.0f32; 4];
            let chunks = D / 4;
            for i in 0..chunks {
                let base = i * 4;
                let d0 = a[base] - b[base];
                let d1 = a[base + 1] - b[base + 1];
                let d2 = a[base + 2] - b[base + 2];
                let d3 = a[base + 3] - b[base + 3];
                acc[0] += d0 * d0;
                acc[1] += d1 * d1;
                acc[2] += d2 * d2;
                acc[3] += d3 * d3;
            }
            let tail = chunks * 4;
            let d0 = a[tail] - b[tail];
            let d1 = a[tail + 1] - b[tail + 1];
            (acc[0] + acc[1]) + (acc[2] + acc[3]) + (d0 * d0 + d1 * d1)
        } else if D.is_multiple_of(2) {
            let mut acc = [0.0f32; 2];
            let chunks = D / 2;
            for i in 0..chunks {
                let base = i * 2;
                let d0 = a[base] - b[base];
                let d1 = a[base + 1] - b[base + 1];
                acc[0] += d0 * d0;
                acc[1] += d1 * d1;
            }
            acc[0] + acc[1]
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = x - y;
                    d * d
                })
                .sum()
        }
    }

    pub fn manhattan_distance(&self, other: &Self) -> f32 {
        (*self - *other).manhattan_norm()
    }

    pub fn infinity_distance(&self, other: &Self) -> f32 {
        (*self - *other).infinity_norm()
    }

    pub fn lp_distance(&self, other: &Self, p: f32) -> f32 {
        (*self - *other).lp_norm(p)
    }

    pub fn angle(&self, other: &Self) -> f32 {
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
        result[..D].copy_from_slice(&self.components[..D]);
        DVec::<N> { components: result }
    }

    #[inline]
    pub fn map<F>(&self, mut f: F) -> Self
    where
        F: FnMut(f32) -> f32,
    {
        let mut result = [0.0; D];
        for (item, component) in result.iter_mut().zip(self.components.into_iter()) {
            *item = f(component);
        }
        Self { components: result }
    }

    #[inline]
    pub fn to_int_array(&self) -> [i32; D] {
        let mut result = [0; D];
        for (item, component) in result.iter_mut().zip(self.components.into_iter()) {
            *item = component as i32;
        }
        result
    }

    pub fn abs(&self) -> Self {
        self.map(|x| x.abs())
    }

    pub fn splat(value: f32) -> DVec<D> {
        Self {
            components: [value; D],
        }
    }
}

// From implementations
impl<const D: usize> From<[f32; D]> for DVec<D> {
    fn from(components: [f32; D]) -> Self {
        Self { components }
    }
}
// From implementations
impl<const D: usize> From<DVec<D>> for [f32; D] {
    fn from(vec: DVec<D>) -> Self {
        vec.components
    }
}

// For 2D vector
impl From<(f32, f32)> for DVec<2> {
    fn from((x, y): (f32, f32)) -> Self {
        Self { components: [x, y] }
    }
}

// For 3D vector
impl From<(f32, f32, f32)> for DVec<3> {
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self {
            components: [x, y, z],
        }
    }
}

// For 4D vector
impl From<(f32, f32, f32, f32)> for DVec<4> {
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Self {
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

impl<const D: usize> Mul<f32> for DVec<D> {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        self.map(|x| x * scalar)
    }
}

impl<const D: usize> Mul<DVec<D>> for DVec<D> {
    type Output = Self;

    fn mul(self, other: DVec<D>) -> Self::Output {
        let result = core::array::from_fn(|i| self.components[i] * other.components[i]);
        Self { components: result }
    }
}

impl<const D: usize> Mul<DVec<D>> for f32 {
    type Output = DVec<D>;

    fn mul(self, vector: DVec<D>) -> Self::Output {
        vector * self
    }
}

impl<const D: usize> MulAssign<f32> for DVec<D> {
    fn mul_assign(&mut self, scalar: f32) {
        for i in 0..D {
            self.components[i] *= scalar;
        }
    }
}

impl<const D: usize> Div<f32> for DVec<D> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, scalar: f32) -> Self::Output {
        assert!(scalar != 0.0, "Division by zero");
        let divisor_recip = scalar.recip();
        self.map(|x| x * divisor_recip)
    }
}
impl<const D: usize> Div<DVec<D>> for DVec<D> {
    type Output = Self;

    fn div(self, other: DVec<D>) -> Self::Output {
        let arr = core::array::from_fn(|i| self[i] / other[i]);
        Self { components: arr }
    }
}

impl<const D: usize> DivAssign<f32> for DVec<D> {
    fn div_assign(&mut self, scalar: f32) {
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
    type Output = f32;

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
