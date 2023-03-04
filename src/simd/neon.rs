use std::{
    arch::aarch64::*,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub},
};

use sealed::sealed;

use super::*;
use crate::vops::*;

#[sealed]
impl SupportedSimd for SimdSupport<u8, 8> {
    type Repr = uint8x8_t;

    type MaskRepr = uint8x8_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<u8, 16> {
    type Repr = uint8x16_t;

    type MaskRepr = uint8x16_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<u16, 4> {
    type Repr = uint16x4_t;

    type MaskRepr = uint16x4_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<u16, 8> {
    type Repr = uint16x8_t;

    type MaskRepr = uint16x8_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<u32, 2> {
    type Repr = uint32x2_t;

    type MaskRepr = uint32x2_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<u32, 4> {
    type Repr = uint32x4_t;

    type MaskRepr = uint32x4_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<u64, 2> {
    type Repr = uint64x2_t;

    type MaskRepr = uint64x2_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i8, 8> {
    type Repr = int8x8_t;

    type MaskRepr = uint8x8_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i8, 16> {
    type Repr = int8x16_t;

    type MaskRepr = uint8x16_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i16, 4> {
    type Repr = int16x4_t;

    type MaskRepr = uint16x4_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i16, 8> {
    type Repr = int16x8_t;

    type MaskRepr = uint16x8_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i32, 2> {
    type Repr = int32x2_t;

    type MaskRepr = uint32x2_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i32, 4> {
    type Repr = int32x4_t;

    type MaskRepr = uint32x4_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<i64, 2> {
    type Repr = int64x2_t;

    type MaskRepr = uint64x2_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<f32, 2> {
    type Repr = float32x2_t;

    type MaskRepr = uint32x2_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<f32, 4> {
    type Repr = float32x4_t;

    type MaskRepr = uint32x4_t;
}

#[sealed]
impl SupportedSimd for SimdSupport<f64, 2> {
    type Repr = float64x2_t;

    type MaskRepr = uint64x2_t;
}

macro_rules! impl_from_scalar {
    ($([$scalar:ident $lanes:literal $intrinsic:ident])*) => {
        $(
            impl From<$scalar> for Simd<$scalar, $lanes> {
                #[inline]
                fn from(value: $scalar) -> Self {
                    Self(unsafe { $intrinsic(value) })
                }
            }
        )*
    }
}

impl_from_scalar! {
    [u8 8 vdup_n_u8]
    [u8 16 vdupq_n_u8]
    [u16 4 vdup_n_u16]
    [u16 8 vdupq_n_u16]
    [u32 2 vdup_n_u32]
    [u32 4 vdupq_n_u32]
    [u64 2 vdupq_n_u64]
    [i8 8 vdup_n_s8]
    [i8 16 vdupq_n_s8]
    [i16 4 vdup_n_s16]
    [i16 8 vdupq_n_s16]
    [i32 2 vdup_n_s32]
    [i32 4 vdupq_n_s32]
    [i64 2 vdupq_n_s64]
    [f32 2 vdup_n_f32]
    [f32 4 vdupq_n_f32]
    [f64 2 vdupq_n_f64]
}

macro_rules! impl_trivial_unops {
    ($(<$typ:ty, $lanes:literal> [$($trait:ident $func:ident $intrinsic:ident),* $(,)?])*) => {
        $(
            $(
                impl $trait for Simd<$typ, $lanes> {
                    type Output = Self;

                    #[inline]
                    fn $func(self) -> Self::Output {
                        Self(unsafe { $intrinsic(self.0) })
                    }
                }
            )*
        )*
    }
}

impl_trivial_unops! {
    <u8, 8> [
        Not not vmvn_u8,
    ]

    <u8, 16> [
        Not not vmvnq_u8,
    ]

    <u16, 4> [
        Not not vmvn_u16,
    ]

    <u16, 8> [
        Not not vmvnq_u16,
    ]

    <u32, 2> [
        Not not vmvn_u32,
    ]

    <u32, 4> [
        Not not vmvnq_u32,
    ]

    <i8, 8> [
        Not not vmvn_s8,
        Neg neg vneg_s8,
        VAbs vabs vabs_s8,
    ]

    <i8, 16> [
        Not not vmvnq_s8,
        Neg neg vnegq_s8,
        VAbs vabs vabsq_s8,
    ]

    <i16, 4> [
        Not not vmvn_s16,
        Neg neg vneg_s16,
        VAbs vabs vabs_s16,
    ]

    <i16, 8> [
        Not not vmvnq_s16,
        Neg neg vnegq_s16,
        VAbs vabs vabsq_s16,
    ]

    <i32, 2> [
        Not not vmvn_s32,
        Neg neg vneg_s32,
        VAbs vabs vabs_s32,
    ]

    <i32, 4> [
        Not not vmvnq_s32,
        Neg neg vnegq_s32,
        VAbs vabs vabsq_s32,
    ]

    <i64, 2> [
        Neg neg vnegq_s64,
        VAbs vabs vabsq_s64,
    ]

    <f32, 2> [
        Neg neg vneg_f32,
        VAbs vabs vabs_f32,
        VSqrt vsqrt vsqrt_f32,
        VTrunc vtrunc vrnd_f32,
        VFloor vfloor vrndm_f32,
        VCeil vceil vrndp_f32,
        VRound vround vrndn_f32,
    ]

    <f32, 4> [
        Neg neg vnegq_f32,
        VAbs vabs vabsq_f32,
        VSqrt vsqrt vsqrtq_f32,
        VTrunc vtrunc vrndq_f32,
        VFloor vfloor vrndmq_f32,
        VCeil vceil vrndpq_f32,
        VRound vround vrndnq_f32,
    ]

    <f64, 2> [
        Neg neg vnegq_f64,
        VAbs vabs vabsq_f64,
        VSqrt vsqrt vsqrtq_f64,
        VTrunc vtrunc vrndq_f64,
        VFloor vfloor vrndmq_f64,
        VCeil vceil vrndpq_f64,
        VRound vround vrndnq_f64,
    ]
}

macro_rules! impl_trivial_binops {
    ($(<$typ:ty, $lanes:literal> [$($trait:ident $func:ident $intrinsic:ident),* $(,)?])*) => {
        $(
            $(
                impl $trait for Simd<$typ, $lanes> {
                    type Output = Self;

                    #[inline]
                    fn $func(self, rhs: Self) -> Self::Output {
                        Self(unsafe { $intrinsic(self.0, rhs.0) })
                    }
                }
            )*
        )*
    }
}

impl_trivial_binops! {
    <u8, 8> [
        Add add vadd_u8,
        Sub sub vsub_u8,
        Mul mul vmul_u8,
        BitAnd bitand vand_u8,
        BitOr bitor vorr_u8,
        BitXor bitxor veor_u8,
        VMin vmin vmin_u8,
        VMax vmax vmax_u8,
    ]

    <u8, 16> [
        Add add vaddq_u8,
        Sub sub vsubq_u8,
        Mul mul vmulq_u8,
        BitAnd bitand vandq_u8,
        BitOr bitor vorrq_u8,
        BitXor bitxor veorq_u8,
        VMin vmin vminq_u8,
        VMax vmax vmaxq_u8,
    ]

    <u16, 4> [
        Add add vadd_u16,
        Sub sub vsub_u16,
        Mul mul vmul_u16,
        BitAnd bitand vand_u16,
        BitOr bitor vorr_u16,
        BitXor bitxor veor_u16,
        VMin vmin vmin_u16,
        VMax vmax vmax_u16,
    ]

    <u16, 8> [
        Add add vaddq_u16,
        Sub sub vsubq_u16,
        Mul mul vmulq_u16,
        BitAnd bitand vandq_u16,
        BitOr bitor vorrq_u16,
        BitXor bitxor veorq_u16,
        VMin vmin vminq_u16,
        VMax vmax vmaxq_u16,
    ]

    <u32, 2> [
        Add add vadd_u32,
        Sub sub vsub_u32,
        Mul mul vmul_u32,
        BitAnd bitand vand_u32,
        BitOr bitor vorr_u32,
        BitXor bitxor veor_u32,
        VMin vmin vmin_u32,
        VMax vmax vmax_u32,
    ]

    <u32, 4> [
        Add add vaddq_u32,
        Sub sub vsubq_u32,
        Mul mul vmulq_u32,
        BitAnd bitand vandq_u32,
        BitOr bitor vorrq_u32,
        BitXor bitxor veorq_u32,
        VMin vmin vminq_u32,
        VMax vmax vmaxq_u32,
    ]

    <u64, 2> [
        Add add vaddq_u64,
        Sub sub vsubq_u64,
        BitAnd bitand vandq_u64,
        BitOr bitor vorrq_u64,
        BitXor bitxor veorq_u64,
    ]

    <i8, 8> [
        Add add vadd_s8,
        Sub sub vsub_s8,
        Mul mul vmul_s8,
        BitAnd bitand vand_s8,
        BitOr bitor vorr_s8,
        BitXor bitxor veor_s8,
        VMin vmin vmin_s8,
        VMax vmax vmax_s8,
    ]

    <i8, 16> [
        Add add vaddq_s8,
        Sub sub vsubq_s8,
        Mul mul vmulq_s8,
        BitAnd bitand vandq_s8,
        BitOr bitor vorrq_s8,
        BitXor bitxor veorq_s8,
        VMin vmin vminq_s8,
        VMax vmax vmaxq_s8,
    ]

    <i16, 4>[
        Add add vadd_s16,
        Sub sub vsub_s16,
        Mul mul vmul_s16,
        BitAnd bitand vand_s16,
        BitOr bitor vorr_s16,
        BitXor bitxor veor_s16,
        VMin vmin vmin_s16,
        VMax vmax vmax_s16,
    ]

    <i16, 8> [
        Add add vaddq_s16,
        Sub sub vsubq_s16,
        Mul mul vmulq_s16,
        BitAnd bitand vandq_s16,
        BitOr bitor vorrq_s16,
        BitXor bitxor veorq_s16,
        VMin vmin vminq_s16,
        VMax vmax vmaxq_s16,
    ]

    <i32, 2> [
        Add add vadd_s32,
        Sub sub vsub_s32,
        Mul mul vmul_s32,
        BitAnd bitand vand_s32,
        BitOr bitor vorr_s32,
        BitXor bitxor veor_s32,
        VMin vmin vmin_s32,
        VMax vmax vmax_s32,
    ]

    <i32, 4> [
        Add add vaddq_s32,
        Sub sub vsubq_s32,
        Mul mul vmulq_s32,
        BitAnd bitand vandq_s32,
        BitOr bitor vorrq_s32,
        BitXor bitxor veorq_s32,
        VMin vmin vminq_s32,
        VMax vmax vmaxq_s32,
    ]

    <i64, 2> [
        Add add vaddq_s64,
        Sub sub vsubq_s64,
        BitAnd bitand vandq_s64,
        BitOr bitor vorrq_s64,
        BitXor bitxor veorq_s64,
    ]

    <f32, 2> [
        Add add vadd_f32,
        Sub sub vsub_f32,
        Mul mul vmul_f32,
        Div div vdiv_f32,
        VMin vmin vmin_f32,
        VMax vmax vmax_f32,
    ]

    <f32, 4> [
        Add add vaddq_f32,
        Sub sub vsubq_f32,
        Mul mul vmulq_f32,
        Div div vdivq_f32,
        VMin vmin vminq_f32,
        VMax vmax vmaxq_f32,
    ]

    <f64, 2> [
        Add add vaddq_f64,
        Sub sub vsubq_f64,
        Mul mul vmulq_f64,
        Div div vdivq_f64,
        VMin vmin vminq_f64,
        VMax vmax vmaxq_f64,
    ]
}
