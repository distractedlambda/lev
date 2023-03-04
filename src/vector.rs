use std::{
    borrow::{Borrow, BorrowMut},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr,
        ShrAssign, Sub, SubAssign,
    },
};

use crate::vops::*;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Vector<T, N> {
    pub fn from_each<U>(other: Vector<U, N>) -> Self
    where
        T: From<U>,
    {
        Self(other.0.map(T::from))
    }

    pub fn into_each<U>(self) -> Vector<U, N>
    where
        T: Into<U>,
    {
        Vector(self.0.map(T::into))
    }

    pub fn each_ref(&self) -> Vector<&T, N> {
        Vector(self.0.each_ref())
    }

    pub fn each_mut(&mut self) -> Vector<&mut T, N> {
        Vector(self.0.each_mut())
    }

    pub fn zip<U>(self, other: Vector<U, N>) -> Vector<(T, U), N> {
        Vector(self.0.zip(other.0))
    }

    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Vector<U, N> {
        Vector(self.0.map(f))
    }

    pub fn map2<U, V>(self, other: Vector<U, N>, mut f: impl FnMut(T, U) -> V) -> Vector<V, N> {
        self.zip(other).map(|(t, u)| f(t, u))
    }

    pub fn update(&mut self, f: impl FnMut(&mut T)) {
        self.0.iter_mut().for_each(f)
    }

    pub fn update2<U>(&mut self, other: Vector<U, N>, mut f: impl FnMut(&mut T, U)) {
        self.0
            .each_mut()
            .zip(other.0)
            .into_iter()
            .for_each(|(t, u)| f(t, u))
    }

    pub fn dot<U>(self, other: Vector<U, N>) -> T::Output
    where
        T: Mul<U>,
        T::Output: Add<Output = T::Output>,
    {
        (self * other).vsum()
    }

    pub fn norm_squared(self) -> T::Output
    where
        T: Clone + Mul,
        T::Output: Add<Output = T::Output>,
    {
        self.clone().dot(self)
    }

    pub fn norm(self) -> <T::Output as VSqrt>::Output
    where
        T: Clone + Mul,
        T::Output: Add<Output = T::Output> + VSqrt,
    {
        self.norm_squared().vsqrt()
    }
}

impl<T: Default, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Self([(); N].map(|_| T::default()))
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
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

macro_rules! impl_trivial_unops {
    ($([$op_trait:ident $op_func:ident])*) => {
        $(
            impl<T: $op_trait, const N: usize> $op_trait for Vector<T, N> {
                type Output = Vector<T::Output, N>;

                fn $op_func(self) -> Self::Output {
                    self.map(T::$op_func)
                }
            }
        )*
    }
}

impl_trivial_unops! {
    [Neg neg]
    [Not not]
    [VAbs vabs]
    [VCeil vceil]
    [VCls vcls]
    [VClz vclz]
    [VFloor vfloor]
    [VPopcount vpopcount]
    [VRound vround]
    [VSqrt vsqrt]
    [VTrunc vtrunc]
}

macro_rules! impl_trivial_binops {
    ($([$op_trait:ident $op_func:ident])*) => {
        $(
            impl<T: $op_trait<U>, U, const N: usize> $op_trait<Vector<U, N>> for Vector<T, N> {
                type Output = Vector<T::Output, N>;

                fn $op_func(self, rhs: Vector<U, N>) -> Self::Output {
                    self.map2(rhs, T::$op_func)
                }
            }
        )*
    }
}

impl_trivial_binops! {
    [Add add]
    [BitAnd bitand]
    [BitOr bitor]
    [BitXor bitxor]
    [Div div]
    [Mul mul]
    [Rem rem]
    [Shl shl]
    [Shr shr]
    [Sub sub]
    [VEq veq]
    [VGe vge]
    [VGt vgt]
    [VLe vle]
    [VLt vlt]
    [VMax vmax]
    [VMin vmin]
    [VNe vne]
}

macro_rules! impl_trivial_assignops {
    ($([$op_trait:ident $op_func:ident])*) => {
        $(
            impl<T: $op_trait<U>, U, const N: usize> $op_trait<Vector<U, N>> for Vector<T, N> {
                fn $op_func(&mut self, rhs: Vector<U, N>) {
                    self.update2(rhs, T::$op_func)
                }
            }
        )*
    }
}

impl_trivial_assignops! {
    [AddAssign add_assign]
    [SubAssign sub_assign]
    [MulAssign mul_assign]
    [DivAssign div_assign]
    [RemAssign rem_assign]
    [BitAndAssign bitand_assign]
    [BitOrAssign bitor_assign]
    [BitXorAssign bitxor_assign]
    [ShlAssign shl_assign]
    [ShrAssign shr_assign]
}

macro_rules! impl_trivial_reduceops {
    ($([$reduce_trait:ident $reduce_func:ident $binop_trait:ident $binop_func:ident])*) => {
        $(
            impl<T: $binop_trait<Output = T>, const N: usize> $reduce_trait for Vector<T, N> {
                type Output = T;

                fn $reduce_func(self) -> Self::Output {
                    self.0.into_iter().reduce(T::$binop_func).unwrap()
                }
            }
        )*
    }
}

impl_trivial_reduceops! {
    [VAll vall BitAnd bitand]
    [VAny vany BitOr bitor]
    [VMaximum vmaximum VMax vmax]
    [VMinimum vminimum VMin vmin]
    [VProduct vproduct Mul mul]
    [VSum vsum Add add]
}
