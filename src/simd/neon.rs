use std::{arch::aarch64 as arch, mem::transmute, ops::Not};

use super::Mask;

macro_rules! simd_types {
    ($($name:ident($repr:ident) { $($clause:tt)* })*) => {
        $(
            #[repr(transparent)]
            #[derive(Clone, Copy, Debug)]
            pub struct $name(arch::$repr);

            impl From<arch::$repr> for $name {
                #[inline]
                fn from(value: arch::$repr) -> Self {
                    Self(value)
                }
            }

            impl From<$name> for arch::$repr {
                #[inline]
                fn from(value: $name) -> Self {
                    value.0
                }
            }

            $(simd_types_clause!($name, $clause);)*
        )*
    };
}

macro_rules! simd_types_clause {
    ($name:ident, [Not => $func:ident]) => {
        impl std::ops::Not for $name {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [Neg => $func:ident]) => {
        impl std::ops::Neg for $name {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [Add => $func:ident]) => {
        impl std::ops::Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                *self = *self + other
            }
        }
    };

    ($name:ident, [Sub => $func:ident]) => {
        impl std::ops::Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                *self = *self - other
            }
        }
    };

    ($name:ident, [Mul => $func:ident]) => {
        impl std::ops::Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::MulAssign for $name {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                *self = *self * other
            }
        }
    };

    ($name:ident, [Div => $func:ident]) => {
        impl std::ops::Div for $name {
            type Output = Self;

            #[inline]
            fn div(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::DivAssign for $name {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                *self = *self / other
            }
        }

        impl std::ops::Rem for $name {
            type Output = Self;

            #[inline]
            fn rem(self, other: Self) -> Self {
                use super::Trunc;
                self - (self / other).trunc() * other
            }
        }

        impl std::ops::RemAssign for $name {
            #[inline]
            fn rem_assign(&mut self, other: Self) {
                *self = *self % other
            }
        }
    };

    ($name:ident, [BitAnd => $func:ident]) => {
        impl std::ops::BitAnd for $name {
            type Output = Self;

            #[inline]
            fn bitand(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, other: Self) {
                *self = *self & other
            }
        }
    };

    ($name:ident, [BitOr => $func:ident]) => {
        impl std::ops::BitOr for $name {
            type Output = Self;

            #[inline]
            fn bitor(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, other: Self) {
                *self = *self | other
            }
        }
    };

    ($name:ident, [BitXor => $func:ident]) => {
        impl std::ops::BitXor for $name {
            type Output = Self;

            #[inline]
            fn bitxor(self, other: Self) -> Self {
                Self(unsafe { arch::$func(self.0, other.0) })
            }
        }

        impl std::ops::BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, other: Self) {
                *self = *self ^ other
            }
        }
    };

    ($name:ident, [Simd => $scalar:ty, $mask:ty, $lanes:literal, $splat_func:ident]) => {
        impl super::Simd for $name {
            type Scalar = $scalar;

            type Mask = $mask;

            const LANES: usize = $lanes;

            #[inline]
            fn splat(value: $scalar) -> Self {
                Self(unsafe { arch::$splat_func(value) })
            }
        }
    };

    (
        $name:ident,
        [Mask => $scalar:ty, $lanes:literal, $splat_func:ident, $max_func:ident, $min_func:ident]
    ) => {
        impl super::Mask for $name {
            const LANES: usize = $lanes;

            #[inline]
            fn splat(value: bool) -> Self {
                Self(unsafe { arch::$splat_func(if value { <$scalar>::MAX } else { 0 }) })
            }

            #[inline]
            fn any(self) -> bool {
                (unsafe { arch::$max_func(self.0) }) != 0
            }

            #[inline]
            fn all(self) -> bool {
                (unsafe { arch::$min_func(self.0) }) != 0
            }
        }
    };

    ($name:ident, [Select<$typ:ty> => $func:ident]) => {
        impl super::Select<$typ> for $name {
            #[inline]
            fn select(self, if_true: $typ, if_false: $typ) -> $typ {
                (unsafe { arch::$func(self.0, if_true.0, if_false.0) }).into()
            }
        }
    };

    ($name:ident, [SimdEq => $func:ident]) => {
        impl super::SimdEq for $name {
            #[inline]
            fn simd_eq(self, other: Self) -> Self::Mask {
                (unsafe { arch::$func(self.0, other.0) }).into()
            }
        }
    };

    (
        $name:ident,
        [SimdOrd =>
            $lt_func:ident,
            $le_func:ident,
            $gt_func:ident,
            $ge_func:ident
            $(, $min_func:ident, $max_func:ident)?
        ]
    ) => {
        impl super::SimdOrd for $name {
            #[inline]
            fn simd_lt(self, other: Self) -> Self::Mask {
                (unsafe { arch::$lt_func(self.0, other.0) }).into()
            }

            #[inline]
            fn simd_le(self, other: Self) -> Self::Mask {
                (unsafe { arch::$le_func(self.0, other.0) }).into()
            }

            #[inline]
            fn simd_gt(self, other: Self) -> Self::Mask {
                (unsafe { arch::$gt_func(self.0, other.0) }).into()
            }

            #[inline]
            fn simd_ge(self, other: Self) -> Self::Mask {
                (unsafe { arch::$ge_func(self.0, other.0) }).into()
            }

            $(
                #[inline]
                fn simd_min(self, other: Self) -> Self {
                    Self(unsafe { arch::$min_func(self.0, other.0) })
                }

                #[inline]
                fn simd_max(self, other: Self) -> Self {
                    Self(unsafe { arch::$max_func(self.0, other.0) })
                }
            )?
        }
    };

    ($name:ident, [ReduceAdd => $func:ident]) => {
        impl super::ReduceAdd for $name {
            #[inline]
            fn reduce_add(self) -> Self::Scalar {
                unsafe { arch::$func(self.0) }
            }
        }
    };

    ($name:ident, [ReduceMin => $func:ident]) => {
        impl super::ReduceMin for $name {
            #[inline]
            fn reduce_min(self) -> Self::Scalar {
                unsafe { arch::$func(self.0) }
            }
        }
    };

    ($name:ident, [ReduceMax => $func:ident]) => {
        impl super::ReduceMax for $name {
            #[inline]
            fn reduce_max(self) -> Self::Scalar {
                unsafe { arch::$func(self.0) }
            }
        }
    };

    ($name:ident, [Floor => $func:ident]) => {
        impl super::Floor for $name {
            #[inline]
            fn floor(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [Ceil => $func:ident]) => {
        impl super::Ceil for $name {
            #[inline]
            fn ceil(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [Trunc => $func:ident]) => {
        impl super::Trunc for $name {
            #[inline]
            fn trunc(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [Round => $func:ident]) => {
        impl super::Round for $name {
            #[inline]
            fn round(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [Sqrt => $func:ident]) => {
        impl super::Sqrt for $name {
            #[inline]
            fn sqrt(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };
}

simd_types! {
    Mask8x8(uint8x8_t) {
        [Mask => u8, 8, vdup_n_u8, vmaxv_u8, vminv_u8]
        [Select<U8x8> => vbsl_u8]
        [Select<I8x8> => vbsl_s8]
        [Not => vmvn_u8]
        [BitAnd => vand_u8]
        [BitOr => vorr_u8]
        [BitXor => veor_u8]
    }

    Mask8x16(uint8x16_t) {
        [Mask => u8, 16, vdupq_n_u8, vmaxvq_u8, vminvq_u8]
        [Select<U8x16> => vbslq_u8]
        [Select<I8x16> => vbslq_s8]
        [Not => vmvnq_u8]
        [BitAnd => vandq_u8]
        [BitOr => vorrq_u8]
        [BitXor => veorq_u8]
    }

    Mask16x4(uint16x4_t) {
        [Mask => u16, 4, vdup_n_u16, vmaxv_u16, vminv_u16]
        [Select<U16x4> => vbsl_u16]
        [Select<I16x4> => vbsl_s16]
        [Not => vmvn_u16]
        [BitAnd => vand_u16]
        [BitOr => vorr_u16]
        [BitXor => veor_u16]
    }

    Mask16x8(uint16x8_t) {
        [Mask => u16, 8, vdupq_n_u16, vmaxvq_u16, vminvq_u16]
        [Select<U16x8> => vbslq_u16]
        [Select<I16x8> => vbslq_s16]
        [Not => vmvnq_u16]
        [BitAnd => vandq_u16]
        [BitOr => vorrq_u16]
        [BitXor => veorq_u16]
    }

    Mask32x2(uint32x2_t) {
        [Mask => u32, 2, vdup_n_u32, vmaxv_u32, vminv_u32]
        [Select<U32x2> => vbsl_u32]
        [Select<I32x2> => vbsl_s32]
        [Select<F32x2> => vbsl_f32]
        [Not => vmvn_u32]
        [BitAnd => vand_u32]
        [BitOr => vorr_u32]
        [BitXor => veor_u32]
    }

    Mask32x4(uint32x4_t) {
        [Mask => u32, 4, vdupq_n_u32, vmaxvq_u32, vminvq_u32]
        [Select<U32x4> => vbslq_u32]
        [Select<I32x4> => vbslq_s32]
        [Select<F32x4> => vbslq_f32]
        [Not => vmvnq_u32]
        [BitAnd => vandq_u32]
        [BitOr => vorrq_u32]
        [BitXor => veorq_u32]
    }

    Mask64x2(uint64x2_t) {
        [Select<U64x2> => vbslq_u64]
        [Select<I64x2> => vbslq_s64]
        [Select<F64x2> => vbslq_f64]
        [BitAnd => vandq_u64]
        [BitOr => vorrq_u64]
        [BitXor => veorq_u64]
    }

    U8x8(uint8x8_t) {
        [Simd => u8, Mask8x8, 8, vdup_n_u8]
        [SimdEq => vceq_u8]
        [SimdOrd => vclt_u8, vcle_u8, vcgt_u8, vcge_u8, vmin_u8, vmax_u8]
        [Not => vmvn_u8]
        [Add => vadd_u8]
        [Sub => vsub_u8]
        [Mul => vmul_u8]
        [BitAnd => vand_u8]
        [BitOr => vorr_u8]
        [BitXor => veor_u8]
        [ReduceAdd => vaddv_u8]
        [ReduceMin => vminv_u8]
        [ReduceMax => vmaxv_u8]
    }

    U8x16(uint8x16_t) {
        [Simd => u8, Mask8x16, 16, vdupq_n_u8]
        [SimdEq => vceqq_u8]
        [SimdOrd => vcltq_u8, vcleq_u8, vcgtq_u8, vcgeq_u8, vminq_u8, vmaxq_u8]
        [Not => vmvnq_u8]
        [Add => vaddq_u8]
        [Sub => vsubq_u8]
        [Mul => vmulq_u8]
        [BitAnd => vandq_u8]
        [BitOr => vorrq_u8]
        [BitXor => veorq_u8]
        [ReduceAdd => vaddvq_u8]
        [ReduceMin => vminvq_u8]
        [ReduceMax => vmaxvq_u8]
    }

    U16x4(uint16x4_t) {
        [Simd => u16, Mask16x4, 4, vdup_n_u16]
        [SimdEq => vceq_u16]
        [SimdOrd => vclt_u16, vcle_u16, vcgt_u16, vcge_u16, vmin_u16, vmax_u16]
        [Not => vmvn_u16]
        [Add => vadd_u16]
        [Sub => vsub_u16]
        [Mul => vmul_u16]
        [BitAnd => vand_u16]
        [BitOr => vorr_u16]
        [BitXor => veor_u16]
        [ReduceAdd => vaddv_u16]
        [ReduceMin => vminv_u16]
        [ReduceMax => vmaxv_u16]
    }

    U16x8(uint16x8_t) {
        [Simd => u16, Mask16x8, 8, vdupq_n_u16]
        [SimdEq => vceqq_u16]
        [SimdOrd => vcltq_u16, vcleq_u16, vcgtq_u16, vcgeq_u16, vminq_u16, vmaxq_u16]
        [Not => vmvnq_u16]
        [Add => vaddq_u16]
        [Sub => vsubq_u16]
        [Mul => vmulq_u16]
        [BitAnd => vandq_u16]
        [BitOr => vorrq_u16]
        [BitXor => veorq_u16]
        [ReduceAdd => vaddvq_u16]
        [ReduceMin => vminvq_u16]
        [ReduceMax => vmaxvq_u16]
    }

    U32x2(uint32x2_t) {
        [Simd => u32, Mask32x2, 2, vdup_n_u32]
        [SimdEq => vceq_u32]
        [SimdOrd => vclt_u32, vcle_u32, vcgt_u32, vcge_u32, vmin_u32, vmax_u32]
        [Not => vmvn_u32]
        [Add => vadd_u32]
        [Sub => vsub_u32]
        [Mul => vmul_u32]
        [BitAnd => vand_u32]
        [BitOr => vorr_u32]
        [BitXor => veor_u32]
        [ReduceAdd => vaddv_u32]
        [ReduceMin => vminv_u32]
        [ReduceMax => vmaxv_u32]
    }

    U32x4(uint32x4_t) {
        [Simd => u32, Mask32x4, 4, vdupq_n_u32]
        [SimdEq => vceqq_u32]
        [SimdOrd => vcltq_u32, vcleq_u32, vcgtq_u32, vcgeq_u32, vminq_u32, vmaxq_u32]
        [Not => vmvnq_u32]
        [Add => vaddq_u32]
        [Sub => vsubq_u32]
        [Mul => vmulq_u32]
        [BitAnd => vandq_u32]
        [BitOr => vorrq_u32]
        [BitXor => veorq_u32]
        [ReduceAdd => vaddvq_u32]
        [ReduceMin => vminvq_u32]
        [ReduceMax => vmaxvq_u32]
    }

    U64x2(uint64x2_t) {
        [Simd => u64, Mask64x2, 2, vdupq_n_u64]
        [SimdEq => vceqq_u64]
        [SimdOrd => vcltq_u64, vcleq_u64, vcgtq_u64, vcgeq_u64]
        [Add => vaddq_u64]
        [Sub => vsubq_u64]
        [BitAnd => vandq_u64]
        [BitOr => vorrq_u64]
        [BitXor => veorq_u64]
        [ReduceAdd => vaddvq_u64]
    }

    I8x8(int8x8_t) {
        [Simd => i8, Mask8x8, 8, vdup_n_s8]
        [SimdEq => vceq_s8]
        [SimdOrd => vclt_s8, vcle_s8, vcgt_s8, vcge_s8, vmin_s8, vmax_s8]
        [Neg => vneg_s8]
        [Not => vmvn_s8]
        [Add => vadd_s8]
        [Sub => vsub_s8]
        [Mul => vmul_s8]
        [BitAnd => vand_s8]
        [BitOr => vorr_s8]
        [BitXor => veor_s8]
        [ReduceAdd => vaddv_s8]
        [ReduceMin => vminv_s8]
        [ReduceMax => vmaxv_s8]
    }

    I8x16(int8x16_t) {
        [Simd => i8, Mask8x16, 16, vdupq_n_s8]
        [SimdEq => vceqq_s8]
        [SimdOrd => vcltq_s8, vcleq_s8, vcgtq_s8, vcgeq_s8, vminq_s8, vmaxq_s8]
        [Neg => vnegq_s8]
        [Not => vmvnq_s8]
        [Add => vaddq_s8]
        [Sub => vsubq_s8]
        [Mul => vmulq_s8]
        [BitAnd => vandq_s8]
        [BitOr => vorrq_s8]
        [BitXor => veorq_s8]
        [ReduceAdd => vaddvq_s8]
        [ReduceMin => vminvq_s8]
        [ReduceMax => vmaxvq_s8]
    }

    I16x4(int16x4_t) {
        [Simd => i16, Mask16x4, 4, vdup_n_s16]
        [SimdEq => vceq_s16]
        [SimdOrd => vclt_s16, vcle_s16, vcgt_s16, vcge_s16, vmin_s16, vmax_s16]
        [Neg => vneg_s16]
        [Not => vmvn_s16]
        [Add => vadd_s16]
        [Sub => vsub_s16]
        [Mul => vmul_s16]
        [BitAnd => vand_s16]
        [BitOr => vorr_s16]
        [BitXor => veor_s16]
        [ReduceAdd => vaddv_s16]
        [ReduceMin => vminv_s16]
        [ReduceMax => vmaxv_s16]
    }

    I16x8(int16x8_t) {
        [Simd => i16, Mask16x8, 8, vdupq_n_s16]
        [SimdEq => vceqq_s16]
        [SimdOrd => vcltq_s16, vcleq_s16, vcgtq_s16, vcgeq_s16, vminq_s16, vmaxq_s16]
        [Neg => vnegq_s16]
        [Not => vmvnq_s16]
        [Add => vaddq_s16]
        [Sub => vsubq_s16]
        [Mul => vmulq_s16]
        [BitAnd => vandq_s16]
        [BitOr => vorrq_s16]
        [BitXor => veorq_s16]
        [ReduceAdd => vaddvq_s16]
        [ReduceMin => vminvq_s16]
        [ReduceMax => vmaxvq_s16]
    }

    I32x2(int32x2_t) {
        [Simd => i32, Mask32x2, 2, vdup_n_s32]
        [SimdEq => vceq_s32]
        [SimdOrd => vclt_s32, vcle_s32, vcgt_s32, vcge_s32, vmin_s32, vmax_s32]
        [Neg => vneg_s32]
        [Not => vmvn_s32]
        [Add => vadd_s32]
        [Sub => vsub_s32]
        [Mul => vmul_s32]
        [BitAnd => vand_s32]
        [BitOr => vorr_s32]
        [BitXor => veor_s32]
        [ReduceAdd => vaddv_s32]
        [ReduceMin => vminv_s32]
        [ReduceMax => vmaxv_s32]
    }

    I32x4(int32x4_t) {
        [Simd => i32, Mask32x4, 4, vdupq_n_s32]
        [SimdEq => vceqq_s32]
        [SimdOrd => vcltq_s32, vcleq_s32, vcgtq_s32, vcgeq_s32, vminq_s32, vmaxq_s32]
        [Neg => vnegq_s32]
        [Not => vmvnq_s32]
        [Add => vaddq_s32]
        [Sub => vsubq_s32]
        [Mul => vmulq_s32]
        [BitAnd => vandq_s32]
        [BitOr => vorrq_s32]
        [BitXor => veorq_s32]
        [ReduceAdd => vaddvq_s32]
        [ReduceMin => vminvq_s32]
        [ReduceMax => vmaxvq_s32]
    }

    I64x2(int64x2_t) {
        [Simd => i64, Mask64x2, 2, vdupq_n_s64]
        [SimdEq => vceqq_s64]
        [SimdOrd => vcltq_s64, vcleq_s64, vcgtq_s64, vcgeq_s64]
        [Neg => vnegq_s64]
        [Add => vaddq_s64]
        [Sub => vsubq_s64]
        [BitAnd => vandq_s64]
        [BitOr => vorrq_s64]
        [BitXor => veorq_s64]
        [ReduceAdd => vaddvq_s64]
    }

    F32x2(float32x2_t) {
        [Simd => f32, Mask32x2, 2, vdup_n_f32]
        [SimdEq => vceq_f32]
        [SimdOrd => vclt_f32, vcle_f32, vcgt_f32, vcge_f32, vminnm_f32, vmaxnm_f32]
        [Neg => vneg_f32]
        [Add => vadd_f32]
        [Sub => vsub_f32]
        [Mul => vmul_f32]
        [Div => vdiv_f32]
        [ReduceAdd => vaddv_f32]
        [ReduceMin => vminnmv_f32]
        [ReduceMax => vmaxnmv_f32]
        [Floor => vrndm_f32]
        [Ceil => vrndp_f32]
        [Trunc => vrnd_f32]
        [Round => vrndn_f32]
        [Sqrt => vsqrt_f32]
    }

    F32x4(float32x4_t) {
        [Simd => f32, Mask32x4, 4, vdupq_n_f32]
        [SimdEq => vceqq_f32]
        [SimdOrd => vcltq_f32, vcleq_f32, vcgtq_f32, vcgeq_f32, vminnmq_f32, vmaxnmq_f32]
        [Neg => vnegq_f32]
        [Add => vaddq_f32]
        [Sub => vsubq_f32]
        [Mul => vmulq_f32]
        [Div => vdivq_f32]
        [ReduceAdd => vaddvq_f32]
        [ReduceMin => vminnmvq_f32]
        [ReduceMax => vmaxnmvq_f32]
        [Floor => vrndmq_f32]
        [Ceil => vrndpq_f32]
        [Trunc => vrndq_f32]
        [Round => vrndnq_f32]
        [Sqrt => vsqrtq_f32]
    }

    F64x2(float64x2_t) {
        [Simd => f64, Mask64x2, 2, vdupq_n_f64]
        [SimdEq => vceqq_f64]
        [SimdOrd => vcltq_f64, vcleq_f64, vcgtq_f64, vcgeq_f64, vminnmq_f64, vmaxnmq_f64]
        [Neg => vnegq_f64]
        [Add => vaddq_f64]
        [Sub => vsubq_f64]
        [Mul => vmulq_f64]
        [Div => vdivq_f64]
        [ReduceAdd => vaddvq_f64]
        [ReduceMin => vminnmvq_f64]
        [ReduceMax => vmaxnmvq_f64]
        [Floor => vrndmq_f64]
        [Ceil => vrndpq_f64]
        [Trunc => vrndq_f64]
        [Round => vrndnq_f64]
        [Sqrt => vsqrtq_f64]
    }
}

impl Not for Mask64x2 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { transmute(arch::vmvnq_u8(transmute(self.0))) })
    }
}

impl Mask for Mask64x2 {
    const LANES: usize = 2;

    #[inline]
    fn splat(value: bool) -> Self {
        Self(unsafe { arch::vdupq_n_u64(if value { u64::MAX } else { 0 }) })
    }

    #[inline]
    fn any(self) -> bool {
        (unsafe { arch::vgetq_lane_u64::<0>(self.0) | arch::vgetq_lane_u64::<1>(self.0) }) != 0
    }

    #[inline]
    fn all(self) -> bool {
        (unsafe { arch::vgetq_lane_u64::<0>(self.0) & arch::vgetq_lane_u64::<1>(self.0) })
            == u64::MAX
    }
}

impl Not for U64x2 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { transmute(arch::vmvnq_u8(transmute(self.0))) })
    }
}

impl Not for I64x2 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { transmute(arch::vmvnq_u8(transmute(self.0))) })
    }
}
