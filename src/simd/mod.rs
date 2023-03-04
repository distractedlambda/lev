use std::{fmt::Debug, marker::PhantomData};

use sealed::sealed;

#[cfg(target_arch = "aarch64")]
mod neon;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Simd<T, const LANES: usize>(<SimdSupport<T, LANES> as SupportedSimd>::Repr)
where
    SimdSupport<T, LANES>: SupportedSimd;

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct SimdMask<T, const LANES: usize>(<SimdSupport<T, LANES> as SupportedSimd>::MaskRepr)
where
    SimdSupport<T, LANES>: SupportedSimd;

pub struct SimdSupport<T, const LANES: usize>(PhantomData<T>);

#[sealed]
pub trait SupportedSimd {
    type Repr: Copy + Debug;

    type MaskRepr: Copy + Debug;
}

macro_rules! impl_automatic_simd_ops {
    ($([$binop_trait:ident $binop_func:ident $assign_trait:ident $assign_func:ident])*) => {
        $(
            impl<T, const LANES: usize> std::ops::$binop_trait<T> for Simd<T, LANES>
            where
                SimdSupport<T, LANES>: SupportedSimd,
                Self: From<T> + std::ops::$binop_trait,
            {
                type Output = <Self as std::ops::$binop_trait<Self>>::Output;

                #[inline]
                fn $binop_func(self, rhs: T) -> Self::Output {
                    self.$binop_func(Self::from(rhs))
                }
            }

            impl<T, const LANES: usize> std::ops::$assign_trait for Simd<T, LANES>
            where
                SimdSupport<T, LANES>: SupportedSimd,
                Self: Copy + std::ops::$binop_trait<Output = Self>,
            {
                #[inline]
                fn $assign_func(&mut self, rhs: Self) {
                    use std::ops::$binop_trait;
                    *self = self.$binop_func(rhs)
                }
            }

            impl<T, const LANES: usize> std::ops::$assign_trait<T> for Simd<T, LANES>
            where
                SimdSupport<T, LANES>: SupportedSimd,
                Self: Copy + From<T> + std::ops::$binop_trait<Output = Self>,
            {
                #[inline]
                fn $assign_func(&mut self, rhs: T) {
                    self.$assign_func(Self::from(rhs))
                }
            }
        )*
    }
}

impl_automatic_simd_ops! {
    [Add add AddAssign add_assign]
    [Sub sub SubAssign sub_assign]
    [Mul mul MulAssign mul_assign]
    [Div div DivAssign div_assign]
    [Rem rem RemAssign rem_assign]
    [BitOr bitor BitOrAssign bitor_assign]
    [BitAnd bitand BitAndAssign bitand_assign]
    [BitXor bitxor BitXorAssign bitxor_assign]
    [Shl shl ShlAssign shl_assign]
    [Shr shr ShrAssign shr_assign]
}
