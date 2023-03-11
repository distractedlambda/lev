use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[cfg(target_arch = "aarch64")]
mod neon;

pub trait Simd: Copy {
    type Scalar;

    type Mask: Mask + Select<Self>;

    const LANES: usize;

    fn splat(value: Self::Scalar) -> Self;
}

pub trait Mask:
    Sized
    + Not<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
{
    const LANES: usize;

    fn splat(value: bool) -> Self;

    fn any(self) -> bool;

    fn all(self) -> bool;

    #[inline]
    fn none(self) -> bool {
        !self.any()
    }
}

pub trait Select<T>: Mask {
    fn select(self, if_true: T, if_false: T) -> T;
}

pub trait SimdEq: Simd {
    fn simd_eq(self, other: Self) -> Self::Mask;

    #[inline]
    fn simd_ne(self, other: Self) -> Self::Mask {
        !self.simd_eq(other)
    }
}

pub trait SimdOrd: SimdEq {
    fn simd_lt(self, other: Self) -> Self::Mask;

    #[inline]
    fn simd_le(self, other: Self) -> Self::Mask {
        self.simd_lt(other) | self.simd_eq(other)
    }

    #[inline]
    fn simd_gt(self, other: Self) -> Self::Mask {
        !self.simd_le(other)
    }

    #[inline]
    fn simd_ge(self, other: Self) -> Self::Mask {
        !self.simd_lt(other)
    }

    #[inline]
    fn simd_min(self, other: Self) -> Self {
        self.simd_lt(other).select(self, other)
    }

    #[inline]
    fn simd_max(self, other: Self) -> Self {
        self.simd_gt(other).select(self, other)
    }

    #[inline]
    fn simd_clamp(self, min: Self, max: Self) -> Self {
        self.simd_max(min).simd_min(max)
    }
}

pub trait ReduceAdd: Simd {
    fn reduce_add(self) -> Self::Scalar;
}

pub trait ReduceMin: Simd {
    fn reduce_min(self) -> Self::Scalar;
}

pub trait ReduceMax: Simd {
    fn reduce_max(self) -> Self::Scalar;
}

pub trait Floor: Simd {
    fn floor(self) -> Self;
}

pub trait Ceil: Simd {
    fn ceil(self) -> Self;
}

pub trait Trunc: Simd {
    fn trunc(self) -> Self;
}

pub trait Round: Simd {
    fn round(self) -> Self;
}

pub trait Sqrt: Simd {
    fn sqrt(self) -> Self;
}

impl Mask for bool {
    const LANES: usize = 1;

    #[inline]
    fn splat(value: bool) -> Self {
        value
    }

    #[inline]
    fn any(self) -> bool {
        self
    }

    #[inline]
    fn all(self) -> bool {
        self
    }
}

impl<T> Select<T> for bool {
    fn select(self, if_true: T, if_false: T) -> T {
        if self {
            if_true
        } else {
            if_false
        }
    }
}

macro_rules! impl_scalar {
    ($($typ:ty),* $(,)?) => {$(
        impl Simd for $typ {
            type Scalar = Self;

            type Mask = bool;

            const LANES: usize = 1;

            #[inline]
            fn splat(value: Self::Scalar) -> Self {
                value
            }
        }

        impl SimdEq for $typ {
            #[inline]
            fn simd_eq(self, rhs: Self) -> Self::Mask {
                self == rhs
            }

            #[inline]
            fn simd_ne(self, rhs: Self) -> Self::Mask {
                self != rhs
            }
        }

        impl SimdOrd for $typ {
            #[inline]
            fn simd_lt(self, other: Self) -> Self::Mask {
                self < other
            }

            #[inline]
            fn simd_le(self, other: Self) -> Self::Mask {
                self <= other
            }

            #[inline]
            fn simd_gt(self, other: Self) -> Self::Mask {
                self > other
            }

            #[inline]
            fn simd_ge(self, other: Self) -> Self::Mask {
                self >= other
            }

            #[inline]
            fn simd_min(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline]
            fn simd_max(self, other: Self) -> Self {
                self.max(other)
            }
        }

        impl ReduceAdd for $typ {
            #[inline]
            fn reduce_add(self) -> Self::Scalar {
                self
            }
        }

        impl ReduceMin for $typ {
            #[inline]
            fn reduce_min(self) -> Self::Scalar {
                self
            }
        }

        impl ReduceMax for $typ {
            #[inline]
            fn reduce_max(self) -> Self::Scalar {
                self
            }
        }
    )*};
}

impl_scalar!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_fp_scalar {
    ($($typ:ty),* $(,)?) => {$(
        impl Floor for $typ {
            #[inline]
            fn floor(self) -> Self {
                self.floor()
            }
        }

        impl Ceil for $typ {
            #[inline]
            fn ceil(self) -> Self {
                self.ceil()
            }
        }

        impl Trunc for $typ {
            #[inline]
            fn trunc(self) -> Self {
                self.trunc()
            }
        }

        impl Round for $typ {
            #[inline]
            fn round(self) -> Self {
                self.round()
            }
        }

        impl Sqrt for $typ {
            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
        }
    )*};
}

impl_fp_scalar!(f32, f64);
