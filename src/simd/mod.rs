use std::{
    fmt::Debug,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

use sealed::sealed;

#[cfg(target_arch = "aarch64")]
mod neon;

#[repr(transparent)]
#[derive(Debug)]
pub struct Simd<T, const LANES: usize>(<T as SimdRepr<LANES>>::Repr)
where
    T: SimdRepr<LANES>;

#[sealed]
pub trait SimdRepr<const LANES: usize> {
    type Repr: Copy + Debug;
}

impl<T, const LANES: usize> Clone for Simd<T, LANES>
where
    T: SimdRepr<LANES>,
{
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T, const LANES: usize> Copy for Simd<T, LANES> where T: SimdRepr<LANES> {}

macro_rules! impl_automatic_simd_ops {
    ($([$binop_trait:ident, $binop_func:ident, $assign_trait:ident, $assign_func:ident])*) => {
        $(
            impl<T, const LANES: usize> $assign_trait for Simd<T, LANES>
            where
                T: SimdRepr<LANES>,
                Self: $binop_trait<Output = Self>,
            {
                #[inline]
                fn $assign_func(&mut self, rhs: Self) {
                    *self = self.$binop_func(rhs)
                }
            }
        )*
    }
}

impl_automatic_simd_ops! {
    [Add, add, AddAssign, add_assign]
    [Sub, sub, SubAssign, sub_assign]
    [Mul, mul, MulAssign, mul_assign]
    [Div, div, DivAssign, div_assign]
    [Rem, rem, RemAssign, rem_assign]
    [BitOr, bitor, BitOrAssign, bitor_assign]
    [BitAnd, bitand, BitAndAssign, bitand_assign]
    [BitXor, bitxor, BitXorAssign, bitxor_assign]
    [Shl, shl, ShlAssign, shl_assign]
    [Shr, shr, ShrAssign, shr_assign]
}
