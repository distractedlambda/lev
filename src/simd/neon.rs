use std::{arch::aarch64 as arch, mem::transmute, ops::Not};

use crate::vectorize::VMask;

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
                use crate::vectorize::VTrunc;
                self - (self / other).v_trunc() * other
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

    ($name:ident, [Vectorize => $scalar:ty, $mask:ty, $lanes:literal, $splat_func:ident]) => {
        impl From<$scalar> for $name {
            #[inline]
            fn from(value: $scalar) -> Self {
                Self(unsafe { arch::$splat_func(value) })
            }
        }

        impl crate::vectorize::Vectorize for $name {
            type Lane = $scalar;

            type Mask = $mask;

            const LANES: usize = $lanes;
        }
    };

    (
        $name:ident,
        [VMask => $scalar:ty, $lanes:literal, $splat_func:ident, $max_func:ident, $min_func:ident]
    ) => {
        impl From<bool> for $name {
            #[inline]
            fn from(value: bool) -> Self {
                Self(unsafe { arch::$splat_func(if value { <$scalar>::MAX } else { 0 }) })
            }
        }

        impl crate::vectorize::VMask for $name {
            type Lane = bool;

            const LANES: usize = $lanes;

            #[inline]
            fn v_any(self) -> bool {
                (unsafe { arch::$max_func(self.0) }) != 0
            }

            #[inline]
            fn v_all(self) -> bool {
                (unsafe { arch::$min_func(self.0) }) != 0
            }
        }
    };

    ($name:ident, [VSelect<$typ:ty> => $func:ident]) => {
        impl crate::vectorize::VSelect<$typ> for $name {
            #[inline]
            fn v_select(self, if_true: $typ, if_false: $typ) -> $typ {
                (unsafe { arch::$func(self.0, if_true.0, if_false.0) }).into()
            }
        }
    };

    ($name:ident, [VEq => $func:ident]) => {
        impl crate::vectorize::VEq for $name {
            #[inline]
            fn v_eq(self, other: Self) -> Self::Mask {
                (unsafe { arch::$func(self.0, other.0) }).into()
            }
        }
    };

    (
        $name:ident,
        [VOrd =>
            $lt_func:ident,
            $le_func:ident,
            $gt_func:ident,
            $ge_func:ident
            $(, $min_func:ident, $max_func:ident)?
        ]
    ) => {
        impl crate::vectorize::VOrd for $name {
            #[inline]
            fn v_lt(self, other: Self) -> Self::Mask {
                (unsafe { arch::$lt_func(self.0, other.0) }).into()
            }

            #[inline]
            fn v_le(self, other: Self) -> Self::Mask {
                (unsafe { arch::$le_func(self.0, other.0) }).into()
            }

            #[inline]
            fn v_gt(self, other: Self) -> Self::Mask {
                (unsafe { arch::$gt_func(self.0, other.0) }).into()
            }

            #[inline]
            fn v_ge(self, other: Self) -> Self::Mask {
                (unsafe { arch::$ge_func(self.0, other.0) }).into()
            }

            $(
                #[inline]
                fn v_min(self, other: Self) -> Self {
                    Self(unsafe { arch::$min_func(self.0, other.0) })
                }

                #[inline]
                fn v_max(self, other: Self) -> Self {
                    Self(unsafe { arch::$max_func(self.0, other.0) })
                }
            )?
        }
    };

    ($name:ident, [VSum => $func:ident]) => {
        impl crate::vectorize::VSum for $name {
            #[inline]
            fn v_sum(self) -> Self::Lane {
                unsafe { arch::$func(self.0) }
            }
        }
    };

    ($name:ident, [VLeast => $func:ident]) => {
        impl crate::vectorize::VLeast for $name {
            #[inline]
            fn v_least(self) -> Self::Lane {
                unsafe { arch::$func(self.0) }
            }
        }
    };

    ($name:ident, [VGreatest => $func:ident]) => {
        impl crate::vectorize::VGreatest for $name {
            #[inline]
            fn v_greatest(self) -> Self::Lane {
                unsafe { arch::$func(self.0) }
            }
        }
    };

    ($name:ident, [VFloor => $func:ident]) => {
        impl crate::vectorize::VFloor for $name {
            #[inline]
            fn v_floor(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [VCeil => $func:ident]) => {
        impl crate::vectorize::VCeil for $name {
            #[inline]
            fn v_ceil(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [VTrunc => $func:ident]) => {
        impl crate::vectorize::VTrunc for $name {
            #[inline]
            fn v_trunc(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [VRound => $func:ident]) => {
        impl crate::vectorize::VRound for $name {
            #[inline]
            fn v_round(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };

    ($name:ident, [VSqrt => $func:ident]) => {
        impl crate::vectorize::VSqrt for $name {
            #[inline]
            fn v_sqrt(self) -> Self {
                Self(unsafe { arch::$func(self.0) })
            }
        }
    };
}

simd_types! {
    Mask8x8(uint8x8_t) {
        [VMask => u8, 8, vdup_n_u8, vmaxv_u8, vminv_u8]
        [VSelect<U8x8> => vbsl_u8]
        [VSelect<I8x8> => vbsl_s8]
        [Not => vmvn_u8]
        [BitAnd => vand_u8]
        [BitOr => vorr_u8]
        [BitXor => veor_u8]
    }

    Mask8x16(uint8x16_t) {
        [VMask => u8, 16, vdupq_n_u8, vmaxvq_u8, vminvq_u8]
        [VSelect<U8x16> => vbslq_u8]
        [VSelect<I8x16> => vbslq_s8]
        [Not => vmvnq_u8]
        [BitAnd => vandq_u8]
        [BitOr => vorrq_u8]
        [BitXor => veorq_u8]
    }

    Mask16x4(uint16x4_t) {
        [VMask => u16, 4, vdup_n_u16, vmaxv_u16, vminv_u16]
        [VSelect<U16x4> => vbsl_u16]
        [VSelect<I16x4> => vbsl_s16]
        [Not => vmvn_u16]
        [BitAnd => vand_u16]
        [BitOr => vorr_u16]
        [BitXor => veor_u16]
    }

    Mask16x8(uint16x8_t) {
        [VMask => u16, 8, vdupq_n_u16, vmaxvq_u16, vminvq_u16]
        [VSelect<U16x8> => vbslq_u16]
        [VSelect<I16x8> => vbslq_s16]
        [Not => vmvnq_u16]
        [BitAnd => vandq_u16]
        [BitOr => vorrq_u16]
        [BitXor => veorq_u16]
    }

    Mask32x2(uint32x2_t) {
        [VMask => u32, 2, vdup_n_u32, vmaxv_u32, vminv_u32]
        [VSelect<U32x2> => vbsl_u32]
        [VSelect<I32x2> => vbsl_s32]
        [VSelect<F32x2> => vbsl_f32]
        [Not => vmvn_u32]
        [BitAnd => vand_u32]
        [BitOr => vorr_u32]
        [BitXor => veor_u32]
    }

    Mask32x4(uint32x4_t) {
        [VMask => u32, 4, vdupq_n_u32, vmaxvq_u32, vminvq_u32]
        [VSelect<U32x4> => vbslq_u32]
        [VSelect<I32x4> => vbslq_s32]
        [VSelect<F32x4> => vbslq_f32]
        [Not => vmvnq_u32]
        [BitAnd => vandq_u32]
        [BitOr => vorrq_u32]
        [BitXor => veorq_u32]
    }

    Mask64x2(uint64x2_t) {
        [VSelect<U64x2> => vbslq_u64]
        [VSelect<I64x2> => vbslq_s64]
        [VSelect<F64x2> => vbslq_f64]
        [BitAnd => vandq_u64]
        [BitOr => vorrq_u64]
        [BitXor => veorq_u64]
    }

    U8x8(uint8x8_t) {
        [Vectorize => u8, Mask8x8, 8, vdup_n_u8]
        [VEq => vceq_u8]
        [VOrd => vclt_u8, vcle_u8, vcgt_u8, vcge_u8, vmin_u8, vmax_u8]
        [Not => vmvn_u8]
        [Add => vadd_u8]
        [Sub => vsub_u8]
        [Mul => vmul_u8]
        [BitAnd => vand_u8]
        [BitOr => vorr_u8]
        [BitXor => veor_u8]
        [VSum => vaddv_u8]
        [VLeast => vminv_u8]
        [VGreatest => vmaxv_u8]
    }

    U8x16(uint8x16_t) {
        [Vectorize => u8, Mask8x16, 16, vdupq_n_u8]
        [VEq => vceqq_u8]
        [VOrd => vcltq_u8, vcleq_u8, vcgtq_u8, vcgeq_u8, vminq_u8, vmaxq_u8]
        [Not => vmvnq_u8]
        [Add => vaddq_u8]
        [Sub => vsubq_u8]
        [Mul => vmulq_u8]
        [BitAnd => vandq_u8]
        [BitOr => vorrq_u8]
        [BitXor => veorq_u8]
        [VSum => vaddvq_u8]
        [VLeast => vminvq_u8]
        [VGreatest => vmaxvq_u8]
    }

    U16x4(uint16x4_t) {
        [Vectorize => u16, Mask16x4, 4, vdup_n_u16]
        [VEq => vceq_u16]
        [VOrd => vclt_u16, vcle_u16, vcgt_u16, vcge_u16, vmin_u16, vmax_u16]
        [Not => vmvn_u16]
        [Add => vadd_u16]
        [Sub => vsub_u16]
        [Mul => vmul_u16]
        [BitAnd => vand_u16]
        [BitOr => vorr_u16]
        [BitXor => veor_u16]
        [VSum => vaddv_u16]
        [VLeast => vminv_u16]
        [VGreatest => vmaxv_u16]
    }

    U16x8(uint16x8_t) {
        [Vectorize => u16, Mask16x8, 8, vdupq_n_u16]
        [VEq => vceqq_u16]
        [VOrd => vcltq_u16, vcleq_u16, vcgtq_u16, vcgeq_u16, vminq_u16, vmaxq_u16]
        [Not => vmvnq_u16]
        [Add => vaddq_u16]
        [Sub => vsubq_u16]
        [Mul => vmulq_u16]
        [BitAnd => vandq_u16]
        [BitOr => vorrq_u16]
        [BitXor => veorq_u16]
        [VSum => vaddvq_u16]
        [VLeast => vminvq_u16]
        [VGreatest => vmaxvq_u16]
    }

    U32x2(uint32x2_t) {
        [Vectorize => u32, Mask32x2, 2, vdup_n_u32]
        [VEq => vceq_u32]
        [VOrd => vclt_u32, vcle_u32, vcgt_u32, vcge_u32, vmin_u32, vmax_u32]
        [Not => vmvn_u32]
        [Add => vadd_u32]
        [Sub => vsub_u32]
        [Mul => vmul_u32]
        [BitAnd => vand_u32]
        [BitOr => vorr_u32]
        [BitXor => veor_u32]
        [VSum => vaddv_u32]
        [VLeast => vminv_u32]
        [VGreatest => vmaxv_u32]
    }

    U32x4(uint32x4_t) {
        [Vectorize => u32, Mask32x4, 4, vdupq_n_u32]
        [VEq => vceqq_u32]
        [VOrd => vcltq_u32, vcleq_u32, vcgtq_u32, vcgeq_u32, vminq_u32, vmaxq_u32]
        [Not => vmvnq_u32]
        [Add => vaddq_u32]
        [Sub => vsubq_u32]
        [Mul => vmulq_u32]
        [BitAnd => vandq_u32]
        [BitOr => vorrq_u32]
        [BitXor => veorq_u32]
        [VSum => vaddvq_u32]
        [VLeast => vminvq_u32]
        [VGreatest => vmaxvq_u32]
    }

    U64x2(uint64x2_t) {
        [Vectorize => u64, Mask64x2, 2, vdupq_n_u64]
        [VEq => vceqq_u64]
        [VOrd => vcltq_u64, vcleq_u64, vcgtq_u64, vcgeq_u64]
        [Add => vaddq_u64]
        [Sub => vsubq_u64]
        [BitAnd => vandq_u64]
        [BitOr => vorrq_u64]
        [BitXor => veorq_u64]
        [VSum => vaddvq_u64]
    }

    I8x8(int8x8_t) {
        [Vectorize => i8, Mask8x8, 8, vdup_n_s8]
        [VEq => vceq_s8]
        [VOrd => vclt_s8, vcle_s8, vcgt_s8, vcge_s8, vmin_s8, vmax_s8]
        [Neg => vneg_s8]
        [Not => vmvn_s8]
        [Add => vadd_s8]
        [Sub => vsub_s8]
        [Mul => vmul_s8]
        [BitAnd => vand_s8]
        [BitOr => vorr_s8]
        [BitXor => veor_s8]
        [VSum => vaddv_s8]
        [VLeast => vminv_s8]
        [VGreatest => vmaxv_s8]
    }

    I8x16(int8x16_t) {
        [Vectorize => i8, Mask8x16, 16, vdupq_n_s8]
        [VEq => vceqq_s8]
        [VOrd => vcltq_s8, vcleq_s8, vcgtq_s8, vcgeq_s8, vminq_s8, vmaxq_s8]
        [Neg => vnegq_s8]
        [Not => vmvnq_s8]
        [Add => vaddq_s8]
        [Sub => vsubq_s8]
        [Mul => vmulq_s8]
        [BitAnd => vandq_s8]
        [BitOr => vorrq_s8]
        [BitXor => veorq_s8]
        [VSum => vaddvq_s8]
        [VLeast => vminvq_s8]
        [VGreatest => vmaxvq_s8]
    }

    I16x4(int16x4_t) {
        [Vectorize => i16, Mask16x4, 4, vdup_n_s16]
        [VEq => vceq_s16]
        [VOrd => vclt_s16, vcle_s16, vcgt_s16, vcge_s16, vmin_s16, vmax_s16]
        [Neg => vneg_s16]
        [Not => vmvn_s16]
        [Add => vadd_s16]
        [Sub => vsub_s16]
        [Mul => vmul_s16]
        [BitAnd => vand_s16]
        [BitOr => vorr_s16]
        [BitXor => veor_s16]
        [VSum => vaddv_s16]
        [VLeast => vminv_s16]
        [VGreatest => vmaxv_s16]
    }

    I16x8(int16x8_t) {
        [Vectorize => i16, Mask16x8, 8, vdupq_n_s16]
        [VEq => vceqq_s16]
        [VOrd => vcltq_s16, vcleq_s16, vcgtq_s16, vcgeq_s16, vminq_s16, vmaxq_s16]
        [Neg => vnegq_s16]
        [Not => vmvnq_s16]
        [Add => vaddq_s16]
        [Sub => vsubq_s16]
        [Mul => vmulq_s16]
        [BitAnd => vandq_s16]
        [BitOr => vorrq_s16]
        [BitXor => veorq_s16]
        [VSum => vaddvq_s16]
        [VLeast => vminvq_s16]
        [VGreatest => vmaxvq_s16]
    }

    I32x2(int32x2_t) {
        [Vectorize => i32, Mask32x2, 2, vdup_n_s32]
        [VEq => vceq_s32]
        [VOrd => vclt_s32, vcle_s32, vcgt_s32, vcge_s32, vmin_s32, vmax_s32]
        [Neg => vneg_s32]
        [Not => vmvn_s32]
        [Add => vadd_s32]
        [Sub => vsub_s32]
        [Mul => vmul_s32]
        [BitAnd => vand_s32]
        [BitOr => vorr_s32]
        [BitXor => veor_s32]
        [VSum => vaddv_s32]
        [VLeast => vminv_s32]
        [VGreatest => vmaxv_s32]
    }

    I32x4(int32x4_t) {
        [Vectorize => i32, Mask32x4, 4, vdupq_n_s32]
        [VEq => vceqq_s32]
        [VOrd => vcltq_s32, vcleq_s32, vcgtq_s32, vcgeq_s32, vminq_s32, vmaxq_s32]
        [Neg => vnegq_s32]
        [Not => vmvnq_s32]
        [Add => vaddq_s32]
        [Sub => vsubq_s32]
        [Mul => vmulq_s32]
        [BitAnd => vandq_s32]
        [BitOr => vorrq_s32]
        [BitXor => veorq_s32]
        [VSum => vaddvq_s32]
        [VLeast => vminvq_s32]
        [VGreatest => vmaxvq_s32]
    }

    I64x2(int64x2_t) {
        [Vectorize => i64, Mask64x2, 2, vdupq_n_s64]
        [VEq => vceqq_s64]
        [VOrd => vcltq_s64, vcleq_s64, vcgtq_s64, vcgeq_s64]
        [Neg => vnegq_s64]
        [Add => vaddq_s64]
        [Sub => vsubq_s64]
        [BitAnd => vandq_s64]
        [BitOr => vorrq_s64]
        [BitXor => veorq_s64]
        [VSum => vaddvq_s64]
    }

    F32x2(float32x2_t) {
        [Vectorize => f32, Mask32x2, 2, vdup_n_f32]
        [VEq => vceq_f32]
        [VOrd => vclt_f32, vcle_f32, vcgt_f32, vcge_f32, vminnm_f32, vmaxnm_f32]
        [Neg => vneg_f32]
        [Add => vadd_f32]
        [Sub => vsub_f32]
        [Mul => vmul_f32]
        [Div => vdiv_f32]
        [VSum => vaddv_f32]
        [VLeast => vminnmv_f32]
        [VGreatest => vmaxnmv_f32]
        [VFloor => vrndm_f32]
        [VCeil => vrndp_f32]
        [VTrunc => vrnd_f32]
        [VRound => vrndn_f32]
        [VSqrt => vsqrt_f32]
    }

    F32x4(float32x4_t) {
        [Vectorize => f32, Mask32x4, 4, vdupq_n_f32]
        [VEq => vceqq_f32]
        [VOrd => vcltq_f32, vcleq_f32, vcgtq_f32, vcgeq_f32, vminnmq_f32, vmaxnmq_f32]
        [Neg => vnegq_f32]
        [Add => vaddq_f32]
        [Sub => vsubq_f32]
        [Mul => vmulq_f32]
        [Div => vdivq_f32]
        [VSum => vaddvq_f32]
        [VLeast => vminnmvq_f32]
        [VGreatest => vmaxnmvq_f32]
        [VFloor => vrndmq_f32]
        [VCeil => vrndpq_f32]
        [VTrunc => vrndq_f32]
        [VRound => vrndnq_f32]
        [VSqrt => vsqrtq_f32]
    }

    F64x2(float64x2_t) {
        [Vectorize => f64, Mask64x2, 2, vdupq_n_f64]
        [VEq => vceqq_f64]
        [VOrd => vcltq_f64, vcleq_f64, vcgtq_f64, vcgeq_f64, vminnmq_f64, vmaxnmq_f64]
        [Neg => vnegq_f64]
        [Add => vaddq_f64]
        [Sub => vsubq_f64]
        [Mul => vmulq_f64]
        [Div => vdivq_f64]
        [VSum => vaddvq_f64]
        [VLeast => vminnmvq_f64]
        [VGreatest => vmaxnmvq_f64]
        [VFloor => vrndmq_f64]
        [VCeil => vrndpq_f64]
        [VTrunc => vrndq_f64]
        [VRound => vrndnq_f64]
        [VSqrt => vsqrtq_f64]
    }
}

impl Not for Mask64x2 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self(unsafe { transmute(arch::vmvnq_u8(transmute(self.0))) })
    }
}

impl From<bool> for Mask64x2 {
    #[inline]
    fn from(value: bool) -> Self {
        Self(unsafe { arch::vdupq_n_u64(if value { u64::MAX } else { 0 }) })
    }
}

impl VMask for Mask64x2 {
    type Lane = bool;

    const LANES: usize = 2;

    #[inline]
    fn v_any(self) -> bool {
        (unsafe { arch::vgetq_lane_u64::<0>(self.0) | arch::vgetq_lane_u64::<1>(self.0) }) != 0
    }

    #[inline]
    fn v_all(self) -> bool {
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
