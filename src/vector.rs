use std::{
    borrow::{Borrow, BorrowMut},
    ops::{Add, Index, IndexMut, Mul},
};

use crate::vectorize::{VMask, VSelect, VSum, Vectorize, VSqrt};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Vector<T, N> {
    #[inline]
    pub fn each_from<U>(other: Vector<U, N>) -> Self
    where
        T: From<U>,
    {
        Self(other.0.map(T::from))
    }

    #[inline]
    pub fn each_into<U>(self) -> Vector<U, N>
    where
        T: Into<U>,
    {
        Vector(self.0.map(T::into))
    }

    #[inline]
    pub fn each_ref(&self) -> Vector<&T, N> {
        Vector(self.0.each_ref())
    }

    #[inline]
    pub fn each_mut(&mut self) -> Vector<&mut T, N> {
        Vector(self.0.each_mut())
    }

    #[inline]
    pub fn zip<U>(self, other: Vector<U, N>) -> Vector<(T, U), N> {
        Vector(self.0.zip(other.0))
    }

    #[inline]
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Vector<U, N> {
        Vector(self.0.map(f))
    }

    #[inline]
    pub fn map2<U, V>(self, other: Vector<U, N>, mut f: impl FnMut(T, U) -> V) -> Vector<V, N> {
        self.zip(other).map(|(t, u)| f(t, u))
    }

    #[inline]
    pub fn update(&mut self, f: impl FnMut(&mut T)) {
        self.0.iter_mut().for_each(f)
    }

    #[inline]
    pub fn update2<U>(&mut self, other: Vector<U, N>, mut f: impl FnMut(&mut T, U)) {
        self.0
            .each_mut()
            .zip(other.0)
            .into_iter()
            .for_each(|(t, u)| f(t, u))
    }

    #[inline]
    pub fn reduce(self, f: impl FnMut(T, T) -> T) -> T {
        self.0.into_iter().reduce(f).unwrap()
    }

    #[inline]
    pub fn dot<U>(self, other: Vector<U, N>) -> T::Output
    where
        T: Mul<U>,
        T::Output: Vectorize + Add<Output = T::Output>,
    {
        (self * other).v_sum()
    }

    #[inline]
    pub fn norm_squared(self) -> T::Output
    where
        T: Clone + Mul,
        T::Output: Vectorize + Add<Output = T::Output>,
    {
        self.clone().dot(self)
    }

    #[inline]
    pub fn norm(self) -> T::Output
    where
        T: Clone + Mul,
        T::Output: Add<Output = T::Output> + VSqrt,
    {
        self.norm_squared().v_sqrt()
    }
}

impl<T: Vectorize, const N: usize> Vectorize for Vector<T, N> {
    type Lane = T;

    type Mask = Vector<T::Mask, N>;

    const LANES: usize = N;
}

impl<T: VMask, const N: usize> VMask for Vector<T, N> {
    type Lane = T;

    const LANES: usize = N;

    #[inline]
    fn v_any(self) -> Self::Lane {
        self.reduce(T::bitor)
    }

    #[inline]
    fn v_all(self) -> Self::Lane {
        self.reduce(T::bitand)
    }
}

impl<T: VSelect<U>, U, const N: usize> VSelect<Vector<U, N>> for Vector<T, N> {
    #[inline]
    fn v_select(self, if_true: Vector<U, N>, if_false: Vector<U, N>) -> Vector<U, N> {
        self.map2(if_true.zip(if_false), |c, (t, f)| c.v_select(t, f))
    }
}

impl<T: Vectorize + Add<Output = T>, const N: usize> VSum for Vector<T, N> {
    fn v_sum(self) -> T {
        self.reduce(T::add)
    }
}

impl<T: Default, const N: usize> Default for Vector<T, N> {
    #[inline]
    fn default() -> Self {
        Self([(); N].map(|_| T::default()))
    }
}

impl<T: Clone, const N: usize> From<T> for Vector<T, N> {
    #[inline]
    fn from(value: T) -> Self {
        Self([(); N].map(|_| value.clone()))
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    #[inline]
    fn from(value: Vector<T, N>) -> Self {
        value.0
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<T, const N: usize> Borrow<[T; N]> for Vector<T, N> {
    #[inline]
    fn borrow(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> BorrowMut<[T; N]> for Vector<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

macro_rules! impl_unary_ops {
    ($($op_trait:ident $op_func:ident),* $(,)?) => {
        $(impl<T: std::ops::$op_trait, const N: usize> std::ops::$op_trait for Vector<T, N> {
            type Output = Vector<T::Output, N>;

            #[inline]
            fn $op_func(self) -> Self::Output {
                self.map(T::$op_func)
            }
        })*
    }
}

impl_unary_ops!(Neg neg, Not not);

macro_rules! impl_binary_ops {
    ($($op_trait:ident $op_func:ident),* $(,)?) => {
        $(impl<T: std::ops::$op_trait<U>, U, const N: usize> std::ops::$op_trait<Vector<U, N>> for Vector<T, N> {
            type Output = Vector<T::Output, N>;

            #[inline]
            fn $op_func(self, other: Vector<U, N>) -> Self::Output {
                self.map2(other, T::$op_func)
            }
        })*
    }
}

impl_binary_ops!(
    Add add,
    Sub sub,
    Mul mul,
    Div div,
    Rem rem,
    BitAnd bitand,
    BitOr bitor,
    BitXor bitxor,
    Shl shl,
    Shr shr,
);

macro_rules! impl_assign_ops {
    ($($op_trait:ident $op_func:ident),* $(,)?) => {
        $(impl<T: std::ops::$op_trait<U>, U, const N: usize> std::ops::$op_trait<Vector<U, N>> for Vector<T, N> {
            #[inline]
            fn $op_func(&mut self, other: Vector<U, N>) {
                self.update2(other, T::$op_func)
            }
        })*
    }
}

impl_assign_ops!(
    AddAssign add_assign,
    SubAssign sub_assign,
    MulAssign mul_assign,
    DivAssign div_assign,
    RemAssign rem_assign,
    BitAndAssign bitand_assign,
    BitOrAssign bitor_assign,
    BitXorAssign bitxor_assign,
    ShlAssign shl_assign,
    ShrAssign shr_assign,
);
