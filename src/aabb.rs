use std::ops::{Mul, Sub};

use crate::{
    vector::Vector,
    vectorize::{VMask, VOrd, VProduct, Vectorize},
};

#[derive(Clone, Copy, Debug)]
pub struct Aabb<T, const DIMENSIONS: usize> {
    min: Vector<T, DIMENSIONS>,
    max: Vector<T, DIMENSIONS>,
}

impl<T, const DIMENSIONS: usize> Aabb<T, DIMENSIONS> {
    pub fn new(min: Vector<T, DIMENSIONS>, max: Vector<T, DIMENSIONS>) -> Self {
        Self { min, max }
    }

    pub fn extent(self) -> Vector<T::Output, DIMENSIONS>
    where
        T: Sub,
    {
        self.max - self.min
    }

    pub fn area(self) -> T::Output
    where
        T: Sub,
        T::Output: Mul<Output = T::Output> + Vectorize,
    {
        self.extent().v_product()
    }

    pub fn empty(self) -> <T::Output as Vectorize>::Mask
    where
        T: Sub,
        T::Output: VOrd,
    {
        self.extent().v_le_zero().v_any()
    }

    pub fn union(self, other: Self) -> Self
    where
        T: VOrd,
    {
        Aabb {
            min: self.min.v_min(other.min),
            max: self.max.v_max(other.max),
        }
    }

    pub fn intersection(self, other: Self) -> Self
    where
        T: VOrd,
    {
        Aabb {
            min: self.min.v_max(other.min),
            max: self.max.v_min(other.max),
        }
    }
}
