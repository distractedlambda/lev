use crate::vectorize::{VMask, VSelect, Vectorize, VEq, VOrd};

#[cfg(target_arch = "aarch64")]
mod neon;

pub trait SimdVector<const LANES: usize> {
    type Type;
}

pub type Simd<T, const LANES: usize> = <T as SimdVector<LANES>>::Type;

pub type SimdMask<T, const LANES: usize> = <<T as SimdVector<LANES>>::Type as Vectorize>::Mask;

#[derive(Clone, Copy, Debug)]
pub struct ExtendedSimd<T, const N: usize>([T; N]);

impl<T: Vectorize, const N: usize> Vectorize for ExtendedSimd<T, N> {
    type Lane = T::Lane;

    type Mask = ExtendedSimd<T::Mask, N>;

    const LANES: usize = T::LANES * N;

    #[inline]
    fn v_splat(value: Self::Lane) -> Self {
        Self([T::v_splat(value); N])
    }

    #[inline]
    fn v_zero() -> Self {
        Self([T::v_zero(); N])
    }

    #[inline]
    fn v_one() -> Self {
        Self([T::v_one(); N])
    }

    #[inline]
    fn v_get(&self, index: usize) -> &Self::Lane {
        self.0[index / T::LANES].v_get(index % T::LANES)
    }

    #[inline]
    fn v_get_mut(&mut self, index: usize) -> &mut Self::Lane {
        self.0[index / T::LANES].v_get_mut(index % T::LANES)
    }
}

impl<T: VMask, const N: usize> VMask for ExtendedSimd<T, N> {
    type MaskLane = T::MaskLane;

    const MASK_LANES: usize = T::MASK_LANES * N;

    #[inline]
    fn v_splat_mask(value: Self::MaskLane) -> Self {
        Self([T::v_splat_mask(value); N])
    }

    #[inline]
    fn v_any(self) -> Self::MaskLane {
        self.0.into_iter().reduce(T::bitor).unwrap().v_any()
    }

    #[inline]
    fn v_all(self) -> Self::MaskLane {
        self.0.into_iter().reduce(T::bitand).unwrap().v_all()
    }
}

impl<T: VEq, const N: usize> VEq for ExtendedSimd<T, N> {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        ExtendedSimd(self.0.zip(other.0).map(|(l, r)| l.v_eq(r)))
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        ExtendedSimd(self.0.zip(other.0).map(|(l, r)| l.v_ne(r)))
    }

    #[inline]
    fn v_eq_zero(self) -> Self::Mask {
        ExtendedSimd(self.0.map(T::v_eq_zero))
    }

    #[inline]
    fn v_ne_zero(self) -> Self::Mask {
        ExtendedSimd(self.0.map(T::v_ne_zero))
    }
}

impl<T: VOrd, const N: usize> VOrd for ExtendedSimd<T, N> {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        ExtendedSimd(self.0.zip(other.0).map(|(l, r)| l.v_lt(r)))
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        ExtendedSimd(self.0.zip(other.0).map(|(l, r)| l.v_le(r)))
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        ExtendedSimd(self.0.zip(other.0).map(|(l, r)| l.v_gt(r)))
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
        ExtendedSimd(self.0.zip(other.0).map(|(l, r)| l.v_ge(r)))
    }

    #[inline]
    fn v_lt_zero(self) -> Self::Mask {
        ExtendedSimd(self.0.map(T::v_lt_zero))
    }

    #[inline]
    fn v_le_zero(self) -> Self::Mask {
        ExtendedSimd(self.0.map(T::v_le_zero))
    }

    #[inline]
    fn v_gt_zero(self) -> Self::Mask {
        ExtendedSimd(self.0.map(T::v_gt_zero))
    }

    #[inline]
    fn v_ge_zero(self) -> Self::Mask {
        ExtendedSimd(self.0.map(T::v_ge_zero))
    }
}

impl<T: VSelect<U>, U, const N: usize> VSelect<ExtendedSimd<U, N>> for ExtendedSimd<T, N> {
    #[inline]
    fn v_select(
        self,
        if_true: ExtendedSimd<U, N>,
        if_false: ExtendedSimd<U, N>,
    ) -> ExtendedSimd<U, N> {
        ExtendedSimd(
            self.0
                .zip(if_true.0.zip(if_false.0))
                .map(|(a, (b, c))| a.v_select(b, c)),
        )
    }
}

macro_rules! impl_extended_simd_unary_ops {
    ($([$op:ident, $func:ident])*) => {
        $(impl<T: std::ops::$op, const N: usize> std::ops::$op for ExtendedSimd<T, N> {
            type Output = ExtendedSimd<T::Output, N>;

            #[inline]
            fn $func(self) -> Self::Output {
                ExtendedSimd(self.0.map(std::ops::$op::$func))
            }
        })*
    }
}

impl_extended_simd_unary_ops! {
    [Neg, neg]
    [Not, not]
}

macro_rules! impl_extended_simd_binary_ops {
    ($([$op:ident, $func:ident])*) => {
        $(impl<T: std::ops::$op<U>, U, const N: usize> std::ops::$op<ExtendedSimd<U, N>> for ExtendedSimd<T, N> {
            type Output = ExtendedSimd<T::Output, N>;

            #[inline]
            fn $func(self, rhs: ExtendedSimd<U, N>) -> Self::Output {
                ExtendedSimd(self.0.zip(rhs.0).map(|(l, r)| std::ops::$op::$func(l, r)))
            }
        })*
    }
}

impl_extended_simd_binary_ops! {
    [Add, add]
    [Sub, sub]
    [Mul, mul]
    [Div, div]
    [Rem, rem]
    [BitAnd, bitand]
    [BitOr, bitor]
    [BitXor, bitxor]
    [Shl, shl]
    [Shr, shr]
}

macro_rules! impl_extended_simd_assign_ops {
    ($([$op:ident, $func:ident])*) => {
        $(impl<T: std::ops::$op<U>, U, const N: usize> std::ops::$op<ExtendedSimd<U, N>> for ExtendedSimd<T, N> {
            #[inline]
            fn $func(&mut self, rhs: ExtendedSimd<U, N>) {
                for (l, r) in self.0.each_mut().zip(rhs.0) {
                    std::ops::$op::$func(l, r)
                }
            }
        })*
    }
}

impl_extended_simd_assign_ops! {
    [AddAssign, add_assign]
    [SubAssign, sub_assign]
    [MulAssign, mul_assign]
    [DivAssign, div_assign]
    [RemAssign, rem_assign]
    [BitAndAssign, bitand_assign]
    [BitOrAssign, bitor_assign]
    [BitXorAssign, bitxor_assign]
    [ShlAssign, shl_assign]
    [ShrAssign, shr_assign]
}
