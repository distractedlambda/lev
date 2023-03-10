use std::{
    arch::aarch64::*,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub},
};

use sealed::sealed;

use super::*;
use crate::vops::*;

macro_rules! impl_simd_repr {
    ($([$typ:ty, $lanes:literal, $repr:ty])*) => {
        $(
            #[sealed]
            impl SimdRepr<$lanes> for $typ {
                type Repr = $repr;
            }
        )*
    }
}

impl_simd_repr! {
    [u8, 8, uint8x8_t]
    [u8, 16, uint8x16_t]
    [u16, 4, uint16x4_t]
    [u16, 8, uint16x8_t]
    [u32, 2, uint32x2_t]
    [u32, 4, uint32x4_t]
    [u64, 2, uint64x2_t]
    [i8, 8, int8x8_t]
    [i8, 16, int8x16_t]
    [i16, 4, int16x4_t]
    [i16, 8, int16x8_t]
    [i32, 2, int32x2_t]
    [i32, 4, int32x4_t]
    [i64, 2, int64x2_t]
    [f32, 2, float32x2_t]
    [f32, 4, float32x4_t]
    [f64, 2, float64x2_t]
}

macro_rules! mask_type {
    ($([$name:ident, $repr:ty])*) => {
        $(
            #[repr(transparent)]
            #[derive(Clone, Copy, Debug)]
            pub struct $name($repr);

            impl From<$repr> for $name {
                #[inline]
                fn from(value: $repr) -> Self {
                    Self(value)
                }
            }

            impl From<$name> for $repr {
                #[inline]
                fn from(value: $name) -> Self {
                    value.0
                }
            }
        )*
    }
}

mask_type! {
    [Mask8x8, uint8x8_t]
    [Mask8x16, uint8x16_t]
    [Mask16x4, uint16x4_t]
    [Mask16x8, uint16x8_t]
    [Mask32x2, uint32x2_t]
    [Mask32x4, uint32x4_t]
    [Mask64x2, uint64x2_t]
}

macro_rules! impl_trivial_from_scalar {
    ($([$typ:ty, $lanes:literal, $impl_func:ident])*) => {
        $(
            impl From<$typ> for Simd<$typ, $lanes> {
                #[inline]
                fn from(value: $typ) -> Self {
                    Self(unsafe { $impl_func(value) })
                }
            }
        )*
    }
}

impl_trivial_from_scalar! {
    [u8, 8, vdup_n_u8]
    [u8, 16, vdupq_n_u8]
    [u16, 4, vdup_n_u16]
    [u16, 8, vdupq_n_u16]
    [u32, 2, vdup_n_u32]
    [u32, 4, vdupq_n_u32]
    [u64, 2, vdupq_n_u64]
    [i8, 8, vdup_n_s8]
    [i8, 16, vdupq_n_s8]
    [i16, 4, vdup_n_s16]
    [i16, 8, vdupq_n_s16]
    [i32, 2, vdup_n_s32]
    [i32, 4, vdupq_n_s32]
    [i64, 2, vdupq_n_s64]
    [f32, 2, vdup_n_f32]
    [f32, 4, vdupq_n_f32]
    [f64, 2, vdupq_n_f64]
}

macro_rules! impl_trivial_unary_op {
    ($($typ:ty {$([$op_trait:path, $op_func:ident, $impl_func:ident])*})*) => {
        $(
            $(
                impl $op_trait for $typ {
                    type Output = Self;

                    #[inline]
                    fn $op_func(self) -> Self::Output {
                        Self(unsafe { $impl_func(self.0) })
                    }
                }
            )*
        )*
    }
}

impl_trivial_unary_op! {
    Simd<u8, 8> {
        [Not, not, vmvn_u8]
    }

    Simd<u8, 16> {
        [Not, not, vmvnq_u8]
    }

    Simd<u16, 4> {
        [Not, not, vmvn_u16]
    }

    Simd<u16, 8> {
        [Not, not, vmvnq_u16]
    }

    Simd<u32, 2> {
        [Not, not, vmvn_u32]
    }

    Simd<u32, 4> {
        [Not, not, vmvnq_u32]
    }

    Simd<i8, 8> {
        [Neg, neg, vneg_s8]
        [Not, not, vmvn_s8]
    }

    Simd<i8, 16> {
        [Neg, neg, vnegq_s8]
        [Not, not, vmvnq_s8]
    }

    Simd<i16, 4> {
        [Neg, neg, vneg_s16]
        [Not, not, vmvn_s16]
    }

    Simd<i16, 8> {
        [Neg, neg, vnegq_s16]
        [Not, not, vmvnq_s16]
    }

    Simd<i32, 2> {
        [Neg, neg, vneg_s32]
        [Not, not, vmvn_s32]
    }

    Simd<i32, 4> {
        [Neg, neg, vnegq_s32]
        [Not, not, vmvnq_s32]
    }

    Simd<i64, 2> {
        [Neg, neg, vnegq_s64]
    }

    Simd<f32, 2> {
        [Neg, neg, vneg_f32]
    }

    Simd<f32, 4> {
        [Neg, neg, vnegq_f32]
    }

    Simd<f64, 2> {
        [Neg, neg, vnegq_f64]
    }

    Mask8x8 {
        [Not, not, vmvn_u8]
    }

    Mask8x16 {
        [Not, not, vmvnq_u8]
    }

    Mask16x4 {
        [Not, not, vmvn_u16]
    }

    Mask16x8 {
        [Not, not, vmvnq_u16]
    }

    Mask32x2 {
        [Not, not, vmvn_u32]
    }

    Mask32x4 {
        [Not, not, vmvnq_u32]
    }
}

impl Not for Simd<u64, 2> {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(self.0))) })
    }
}

impl Not for Simd<i64, 2> {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { vreinterpretq_s64_u8(vmvnq_u8(vreinterpretq_u8_s64(self.0))) })
    }
}

impl Not for Mask64x2 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(self.0))) })
    }
}

macro_rules! impl_trivial_binary_op {
    ($($typ:ty {$([$op_trait:path, $op_func:ident, $impl_func:ident])*})*) => {
        $(
            $(
                impl $op_trait for $typ {
                    type Output = Self;

                    #[inline]
                    fn $op_func(self, other: Self) -> Self::Output {
                        Self(unsafe { $impl_func(self.0, other.0) })
                    }
                }
            )*
        )*
    }
}

impl_trivial_binary_op! {
    Simd<u8, 8> {
        [Add, add, vadd_u8]
        [Sub, sub, vsub_u8]
        [Mul, mul, vmul_u8]
        [BitAnd, bitand, vand_u8]
        [BitOr, bitor, vorr_u8]
        [BitXor, bitxor, veor_u8]
    }

    Simd<u8, 16> {
        [Add, add, vaddq_u8]
        [Sub, sub, vsubq_u8]
        [Mul, mul, vmulq_u8]
        [BitAnd, bitand, vandq_u8]
        [BitOr, bitor, vorrq_u8]
        [BitXor, bitxor, veorq_u8]
    }

    Simd<u16, 4> {
        [Add, add, vadd_u16]
        [Sub, sub, vsub_u16]
        [Mul, mul, vmul_u16]
        [BitAnd, bitand, vand_u16]
        [BitOr, bitor, vorr_u16]
        [BitXor, bitxor, veor_u16]
    }

    Simd<u16, 8> {
        [Add, add, vaddq_u16]
        [Sub, sub, vsubq_u16]
        [Mul, mul, vmulq_u16]
        [BitAnd, bitand, vandq_u16]
        [BitOr, bitor, vorrq_u16]
        [BitXor, bitxor, veorq_u16]
    }

    Simd<u32, 2> {
        [Add, add, vadd_u32]
        [Sub, sub, vsub_u32]
        [Mul, mul, vmul_u32]
        [BitAnd, bitand, vand_u32]
        [BitOr, bitor, vorr_u32]
        [BitXor, bitxor, veor_u32]
    }

    Simd<u32, 4> {
        [Add, add, vaddq_u32]
        [Sub, sub, vsubq_u32]
        [Mul, mul, vmulq_u32]
        [BitAnd, bitand, vandq_u32]
        [BitOr, bitor, vorrq_u32]
        [BitXor, bitxor, veorq_u32]
    }

    Simd<u64, 2> {
        [Add, add, vaddq_u64]
        [Sub, sub, vsubq_u64]
        [BitAnd, bitand, vandq_u64]
        [BitOr, bitor, vorrq_u64]
        [BitXor, bitxor, veorq_u64]
    }

    Simd<i8, 8> {
        [Add, add, vadd_s8]
        [Sub, sub, vsub_s8]
        [Mul, mul, vmul_s8]
        [BitAnd, bitand, vand_s8]
        [BitOr, bitor, vorr_s8]
        [BitXor, bitxor, veor_s8]
        [Shl, shl, vshl_s8]
    }

    Simd<i8, 16> {
        [Add, add, vaddq_s8]
        [Sub, sub, vsubq_s8]
        [Mul, mul, vmulq_s8]
        [BitAnd, bitand, vandq_s8]
        [BitOr, bitor, vorrq_s8]
        [BitXor, bitxor, veorq_s8]
        [Shl, shl, vshlq_s8]
    }

    Simd<i16, 4> {
        [Add, add, vadd_s16]
        [Sub, sub, vsub_s16]
        [Mul, mul, vmul_s16]
        [BitAnd, bitand, vand_s16]
        [BitOr, bitor, vorr_s16]
        [BitXor, bitxor, veor_s16]
        [Shl, shl, vshl_s16]
    }

    Simd<i16, 8> {
        [Add, add, vaddq_s16]
        [Sub, sub, vsubq_s16]
        [Mul, mul, vmulq_s16]
        [BitAnd, bitand, vandq_s16]
        [BitOr, bitor, vorrq_s16]
        [BitXor, bitxor, veorq_s16]
        [Shl, shl, vshlq_s16]
    }

    Simd<i32, 2> {
        [Add, add, vadd_s32]
        [Sub, sub, vsub_s32]
        [Mul, mul, vmul_s32]
        [BitAnd, bitand, vand_s32]
        [BitOr, bitor, vorr_s32]
        [BitXor, bitxor, veor_s32]
        [Shl, shl, vshl_s32]
    }

    Simd<i32, 4> {
        [Add, add, vaddq_s32]
        [Sub, sub, vsubq_s32]
        [Mul, mul, vmulq_s32]
        [BitAnd, bitand, vandq_s32]
        [BitOr, bitor, vorrq_s32]
        [BitXor, bitxor, veorq_s32]
        [Shl, shl, vshlq_s32]
    }

    Simd<i64, 2> {
        [Add, add, vaddq_s64]
        [Sub, sub, vsubq_s64]
        [BitAnd, bitand, vandq_s64]
        [BitOr, bitor, vorrq_s64]
        [BitXor, bitxor, veorq_s64]
        [Shl, shl, vshlq_s64]
    }

    Simd<f32, 2> {
        [Add, add, vadd_f32]
        [Sub, sub, vsub_f32]
        [Mul, mul, vmul_f32]
        [Div, div, vdiv_f32]
    }

    Simd<f32, 4> {
        [Add, add, vaddq_f32]
        [Sub, sub, vsubq_f32]
        [Mul, mul, vmulq_f32]
        [Div, div, vdivq_f32]
    }

    Simd<f64, 2> {
        [Add, add, vaddq_f64]
        [Sub, sub, vsubq_f64]
        [Mul, mul, vmulq_f64]
        [Div, div, vdivq_f64]
    }

    Mask8x8 {
        [BitAnd, bitand, vand_u8]
        [BitOr, bitor, vorr_u8]
        [BitXor, bitxor, veor_u8]
    }

    Mask8x16 {
        [BitAnd, bitand, vandq_u8]
        [BitOr, bitor, vorrq_u8]
        [BitXor, bitxor, veorq_u8]
    }

    Mask16x4 {
        [BitAnd, bitand, vand_u16]
        [BitOr, bitor, vorr_u16]
        [BitXor, bitxor, veor_u16]
    }

    Mask16x8 {
        [BitAnd, bitand, vandq_u16]
        [BitOr, bitor, vorrq_u16]
        [BitXor, bitxor, veorq_u16]
    }

    Mask32x2 {
        [BitAnd, bitand, vand_u32]
        [BitOr, bitor, vorr_u32]
        [BitXor, bitxor, veor_u32]
    }

    Mask32x4 {
        [BitAnd, bitand, vandq_u32]
        [BitOr, bitor, vorrq_u32]
        [BitXor, bitxor, veorq_u32]
    }

    Mask64x2 {
        [BitAnd, bitand, vandq_u64]
        [BitOr, bitor, vorrq_u64]
        [BitXor, bitxor, veorq_u64]
    }
}

macro_rules! impl_mask_assign_ops {
    ($($mask_typ:ty),* $(,)?) => {
        $(
            impl BitAndAssign for $mask_typ {
                #[inline]
                fn bitand_assign(&mut self, other: Self) {
                    *self = *self & other
                }
            }

            impl BitOrAssign for $mask_typ {
                #[inline]
                fn bitor_assign(&mut self, other: Self) {
                    *self = *self | other
                }
            }

            impl BitXorAssign for $mask_typ {
                #[inline]
                fn bitxor_assign(&mut self, other: Self) {
                    *self = *self ^ other
                }
            }
        )*
    }
}

impl_mask_assign_ops!(Mask8x8, Mask8x16, Mask16x4, Mask16x8, Mask32x2, Mask32x4, Mask64x2);

macro_rules! impl_vmask {
    ($($mask_typ:ty { $([$simd_typ:ty, $func:ident])* })*) => {
        $(
            $(
                impl VMask<$simd_typ> for $mask_typ {
                    #[inline]
                    fn v_select(self, if_true: $simd_typ, if_false: $simd_typ) -> $simd_typ {
                        Simd(unsafe { $func(self.0, if_true.0, if_false.0) })
                    }
                }
            )*
        )*
    }
}

impl_vmask! {
    Mask8x8 {
        [Simd<u8, 8>, vbsl_u8]
        [Simd<i8, 8>, vbsl_s8]
    }

    Mask8x16 {
        [Simd<u8, 16>, vbslq_u8]
        [Simd<i8, 16>, vbslq_s8]
    }

    Mask16x4 {
        [Simd<u16, 4>, vbsl_u16]
        [Simd<i16, 4>, vbsl_s16]
    }

    Mask16x8 {
        [Simd<u16, 8>, vbslq_u16]
        [Simd<i16, 8>, vbslq_s16]
    }

    Mask32x2 {
        [Simd<u32, 2>, vbsl_u32]
        [Simd<i32, 2>, vbsl_s32]
        [Simd<f32, 2>, vbsl_f32]
    }

    Mask32x4 {
        [Simd<u32, 4>, vbslq_u32]
        [Simd<i32, 4>, vbslq_s32]
        [Simd<f32, 4>, vbslq_f32]
    }

    Mask64x2 {
        [Simd<u64, 2>, vbslq_u64]
        [Simd<i64, 2>, vbslq_s64]
        [Simd<f64, 2>, vbslq_f64]
    }
}

macro_rules! impl_vectorize {
    ($($mask_typ:ty [$(Simd<$scalar:ty, $lanes:literal>),* $(,)?])*) => {
        $(
            $(
                impl Vectorize for Simd<$scalar, $lanes> {
                    type Scalar = $scalar;

                    type Mask = $mask_typ;
                }
            )*
        )*
    }
}

impl_vectorize! {
    Mask8x8 [Simd<u8, 8>, Simd<i8, 8>]
    Mask8x16 [Simd<u8, 16>, Simd<i8, 16>]
    Mask16x4 [Simd<u16, 4>, Simd<i16, 4>]
    Mask16x8 [Simd<u16, 8>, Simd<i16, 8>]
    Mask32x2 [Simd<u32, 2>, Simd<i32, 2>, Simd<f32, 2>]
    Mask32x4 [Simd<u32, 4>, Simd<i32, 4>, Simd<f32, 4>]
    Mask64x2 [Simd<u64, 2>, Simd<i64, 2>, Simd<f64, 2>]
}

macro_rules! impl_v_partial_eq {
    ($([$typ:ty, $eq_func:ident])*) => {
        $(
            impl VPartialEq for $typ {
                #[inline]
                fn v_eq(self, other: Self) -> Self::Mask {
                    (unsafe { $eq_func(self.0, other.0) }).into()
                }
            }
        )*
    }
}

impl_v_partial_eq! {
    [Simd<u8, 8>, vceq_u8]
    [Simd<u8, 16>, vceqq_u8]
    [Simd<u16, 4>, vceq_u16]
    [Simd<u16, 8>, vceqq_u16]
    [Simd<u32, 2>, vceq_u32]
    [Simd<u32, 4>, vceqq_u32]
    [Simd<u64, 2>, vceqq_u64]
    [Simd<i8, 8>, vceq_s8]
    [Simd<i8, 16>, vceqq_s8]
    [Simd<i16, 4>, vceq_s16]
    [Simd<i16, 8>, vceqq_s16]
    [Simd<i32, 2>, vceq_s32]
    [Simd<i32, 4>, vceqq_s32]
    [Simd<i64, 2>, vceqq_s64]
    [Simd<f32, 2>, vceq_f32]
    [Simd<f32, 4>, vceqq_f32]
    [Simd<f64, 2>, vceqq_f64]
}

macro_rules! impl_v_partial_ord {
    ($([$typ:ty, $lt_func:ident, $le_func:ident, $gt_func:ident, $ge_func:ident, $min_func:ident, $max_func:ident])*) => {
        $(
            impl VPartialOrd for $typ {
                #[inline]
                fn v_lt(self, other: Self) -> Self::Mask {
                    (unsafe { $lt_func(self.0, other.0) }).into()
                }

                #[inline]
                fn v_le(self, other: Self) -> Self::Mask {
                    (unsafe { $le_func(self.0, other.0) }).into()
                }

                #[inline]
                fn v_gt(self, other: Self) -> Self::Mask {
                    (unsafe { $gt_func(self.0, other.0) }).into()
                }

                #[inline]
                fn v_ge(self, other: Self) -> Self::Mask {
                    (unsafe { $ge_func(self.0, other.0) }).into()
                }

                #[inline]
                fn v_min(self, other: Self) -> Self {
                    Self(unsafe { $min_func(self.0, other.0) })
                }

                #[inline]
                fn v_max(self, other: Self) -> Self {
                    Self(unsafe { $max_func(self.0, other.0) })
                }
            }
        )*
    }
}

impl_v_partial_ord! {
    [Simd<u8, 8>, vclt_u8, vcle_u8, vcgt_u8, vcge_u8, vmin_u8, vmax_u8]
    [Simd<u8, 16>, vcltq_u8, vcleq_u8, vcgtq_u8, vcgeq_u8, vminq_u8, vmaxq_u8]
    [Simd<u16, 4>, vclt_u16, vcle_u16, vcgt_u16, vcge_u16, vmin_u16, vmax_u16]
    [Simd<u16, 8>, vcltq_u16, vcleq_u16, vcgtq_u16, vcgeq_u16, vminq_u16, vmaxq_u16]
    [Simd<u32, 2>, vclt_u32, vcle_u32, vcgt_u32, vcge_u32, vmin_u32, vmax_u32]
    [Simd<u32, 4>, vcltq_u32, vcleq_u32, vcgtq_u32, vcgeq_u32, vminq_u32, vmaxq_u32]
    [Simd<i8, 8>, vclt_s8, vcle_s8, vcgt_s8, vcge_s8, vmin_s8, vmax_s8]
    [Simd<i8, 16>, vcltq_s8, vcleq_s8, vcgtq_s8, vcgeq_s8, vminq_s8, vmaxq_s8]
    [Simd<i16, 4>, vclt_s16, vcle_s16, vcgt_s16, vcge_s16, vmin_s16, vmax_s16]
    [Simd<i16, 8>, vcltq_s16, vcleq_s16, vcgtq_s16, vcgeq_s16, vminq_s16, vmaxq_s16]
    [Simd<i32, 2>, vclt_s32, vcle_s32, vcgt_s32, vcge_s32, vmin_s32, vmax_s32]
    [Simd<i32, 4>, vcltq_s32, vcleq_s32, vcgtq_s32, vcgeq_s32, vminq_s32, vmaxq_s32]
    [Simd<f32, 2>, vclt_f32, vcle_f32, vcgt_f32, vcge_f32, vminnm_f32, vmaxnm_f32]
    [Simd<f32, 4>, vcltq_f32, vcleq_f32, vcgtq_f32, vcgeq_f32, vminnmq_f32, vmaxnmq_f32]
    [Simd<f64, 2>, vcltq_f64, vcleq_f64, vcgtq_f64, vcgeq_f64, vminnmq_f64, vmaxnmq_f64]
}

impl VPartialOrd for Simd<u64, 2> {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcltq_u64(self.0, other.0) })
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcleq_u64(self.0, other.0) })
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcgtq_u64(self.0, other.0) })
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcgeq_u64(self.0, other.0) })
    }

    #[inline]
    fn v_min(self, other: Self) -> Self {
        self.v_lt(other).v_select(self, other)
    }

    #[inline]
    fn v_max(self, other: Self) -> Self {
        self.v_gt(other).v_select(self, other)
    }
}

impl VPartialOrd for Simd<i64, 2> {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcltq_s64(self.0, other.0) })
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcleq_s64(self.0, other.0) })
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcgtq_s64(self.0, other.0) })
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
        Mask64x2(unsafe { vcgeq_s64(self.0, other.0) })
    }

    #[inline]
    fn v_min(self, other: Self) -> Self {
        self.v_lt(other).v_select(self, other)
    }

    #[inline]
    fn v_max(self, other: Self) -> Self {
        self.v_gt(other).v_select(self, other)
    }
}

impl VEq for Simd<u8, 8> {}
impl VEq for Simd<u8, 16> {}
impl VEq for Simd<u16, 4> {}
impl VEq for Simd<u16, 8> {}
impl VEq for Simd<u32, 2> {}
impl VEq for Simd<u32, 4> {}
impl VEq for Simd<u64, 2> {}
impl VEq for Simd<i8, 8> {}
impl VEq for Simd<i8, 16> {}
impl VEq for Simd<i16, 4> {}
impl VEq for Simd<i16, 8> {}
impl VEq for Simd<i32, 2> {}
impl VEq for Simd<i32, 4> {}
impl VEq for Simd<i64, 2> {}

impl VOrd for Simd<u8, 8> {}
impl VOrd for Simd<u8, 16> {}
impl VOrd for Simd<u16, 4> {}
impl VOrd for Simd<u16, 8> {}
impl VOrd for Simd<u32, 2> {}
impl VOrd for Simd<u32, 4> {}
impl VOrd for Simd<u64, 2> {}
impl VOrd for Simd<i8, 8> {}
impl VOrd for Simd<i8, 16> {}
impl VOrd for Simd<i16, 4> {}
impl VOrd for Simd<i16, 8> {}
impl VOrd for Simd<i32, 2> {}
impl VOrd for Simd<i32, 4> {}
impl VOrd for Simd<i64, 2> {}

macro_rules! impl_lanewise_div_rem {
    ($(Simd<$scalar:ty, $lanes:literal>),* $(,)?) => {
        $(
            impl Div for Simd<$scalar, $lanes> {
                type Output = Self;

                #[inline]
                fn div(self, other: Self) -> Self::Output {
                    Self(unsafe {
                        std::mem::transmute(
                            std::mem::transmute::<_, [$scalar; $lanes]>(self)
                                .zip(std::mem::transmute::<_, [$scalar; $lanes]>(other))
                                .map(|(l, r)| l.wrapping_div(r))
                        )
                    })
                }
            }

            impl Rem for Simd<$scalar, $lanes> {
                type Output = Self;

                #[inline]
                fn rem(self, other: Self) -> Self::Output {
                    Self(unsafe {
                        std::mem::transmute(
                            std::mem::transmute::<_, [$scalar; $lanes]>(self)
                                .zip(std::mem::transmute::<_, [$scalar; $lanes]>(other))
                                .map(|(l, r)| l.wrapping_rem(r))
                        )
                    })
                }
            }
        )*
    }
}

impl_lanewise_div_rem!(
    Simd<u8, 8>,
    Simd<u8, 16>,
    Simd<u16, 4>,
    Simd<u16, 8>,
    Simd<u32, 2>,
    Simd<u32, 4>,
    Simd<u64, 2>,
    Simd<i8, 8>,
    Simd<i8, 16>,
    Simd<i16, 4>,
    Simd<i16, 8>,
    Simd<i32, 2>,
    Simd<i32, 4>,
    Simd<i64, 2>,
);
