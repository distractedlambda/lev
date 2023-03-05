use std::ops::{BitOr, Mul, Sub};

use crate::{
    vector::Vector,
    vops::{VAny, VLe, VMax, VMin, VProduct, VZero},
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
        T::Output: Mul<Output = T::Output>,
    {
        self.extent().vproduct()
    }

    pub fn empty(self) -> <T::Output as VLe>::Output
    where
        T: Sub,
        T::Output: VLe + VZero,
        <T::Output as VLe>::Output: BitOr<Output = <T::Output as VLe>::Output>,
    {
        self.extent().vle(VZero::vzero()).vany()
    }

    pub fn union(self, other: Self) -> Aabb<<T as VMax>::Output, DIMENSIONS>
    where
        T: VMax + VMin<Output = <T as VMax>::Output>,
    {
        Aabb {
            min: self.min.vmin(other.min),
            max: self.max.vmax(other.max),
        }
    }

    pub fn intersection(self, other: Self) -> Aabb<<T as VMax>::Output, DIMENSIONS>
    where
        T: VMax + VMin<Output = <T as VMax>::Output>,
    {
        Aabb {
            min: self.min.vmax(other.min),
            max: self.max.vmin(other.max),
        }
    }
}
