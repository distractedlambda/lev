use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

pub trait Vectorize: Copy + From<Self::Lane> {
    type Lane;

    type Mask: VMask + VSelect<Self>;

    const LANES: usize;
}

pub trait VMask:
    Copy
    + From<Self::Lane>
    + Not<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
{
    type Lane: Not<Output = Self::Lane>;

    const LANES: usize;

    fn v_any(self) -> Self::Lane;

    fn v_all(self) -> Self::Lane;

    #[inline]
    fn v_none(self) -> Self::Lane {
        !self.v_any()
    }
}

pub trait VSelect<T>: VMask {
    fn v_select(self, if_true: T, if_false: T) -> T;
}

pub trait VEq: Vectorize {
    fn v_eq(self, other: Self) -> Self::Mask;

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        !self.v_eq(other)
    }
}

pub trait VOrd: VEq {
    fn v_lt(self, other: Self) -> Self::Mask;

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self.v_lt(other) | self.v_eq(other)
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        !self.v_le(other)
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
        !self.v_lt(other)
    }

    #[inline]
    fn v_min(self, other: Self) -> Self {
        self.v_lt(other).v_select(self, other)
    }

    #[inline]
    fn v_max(self, other: Self) -> Self {
        self.v_gt(other).v_select(self, other)
    }

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.v_max(min).v_min(max)
    }
}

pub trait VSum<Output = <Self as Vectorize>::Lane>: Vectorize {
    fn v_sum(self) -> Output;
}

pub trait VLeast: VOrd {
    fn v_least(self) -> Self::Lane;
}

pub trait VGreatest: VOrd {
    fn v_greatest(self) -> Self::Lane;
}

pub trait VFloor: Vectorize {
    fn v_floor(self) -> Self;
}

pub trait VCeil: Vectorize {
    fn v_ceil(self) -> Self;
}

pub trait VTrunc: Vectorize {
    fn v_trunc(self) -> Self;
}

pub trait VRound: Vectorize {
    fn v_round(self) -> Self;
}

pub trait VSqrt: Vectorize {
    fn v_sqrt(self) -> Self;
}

impl VMask for bool {
    type Lane = bool;

    const LANES: usize = 1;

    #[inline]
    fn v_any(self) -> bool {
        self
    }

    #[inline]
    fn v_all(self) -> bool {
        self
    }
}

impl<T> VSelect<T> for bool {
    #[inline]
    fn v_select(self, if_true: T, if_false: T) -> T {
        if self {
            if_true
        } else {
            if_false
        }
    }
}

macro_rules! impl_scalar {
    ($($typ:ty),* $(,)?) => {$(
        impl Vectorize for $typ {
            type Lane = Self;

            type Mask = bool;

            const LANES: usize = 1;
        }

        impl VEq for $typ {
            #[inline]
            fn v_eq(self, rhs: Self) -> bool {
                self == rhs
            }

            #[inline]
            fn v_ne(self, rhs: Self) -> bool {
                self != rhs
            }
        }

        impl VOrd for $typ {
            #[inline]
            fn v_lt(self, other: Self) -> bool {
                self < other
            }

            #[inline]
            fn v_le(self, other: Self) -> bool {
                self <= other
            }

            #[inline]
            fn v_gt(self, other: Self) -> bool {
                self > other
            }

            #[inline]
            fn v_ge(self, other: Self) -> bool {
                self >= other
            }

            #[inline]
            fn v_min(self, other: Self) -> Self {
                self.min(other)
            }

            #[inline]
            fn v_max(self, other: Self) -> Self {
                self.max(other)
            }
        }

        impl VSum<Self> for $typ {
            #[inline]
            fn v_sum(self) -> Self {
                self
            }
        }

        impl VLeast for $typ {
            #[inline]
            fn v_least(self) -> Self {
                self
            }
        }

        impl VGreatest for $typ {
            #[inline]
            fn v_greatest(self) -> Self {
                self
            }
        }
    )*};
}

impl_scalar!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_fp_scalar {
    ($($typ:ty),* $(,)?) => {$(
        impl VFloor for $typ {
            #[inline]
            fn v_floor(self) -> Self {
                self.floor()
            }
        }

        impl VCeil for $typ {
            #[inline]
            fn v_ceil(self) -> Self {
                self.ceil()
            }
        }

        impl VTrunc for $typ {
            #[inline]
            fn v_trunc(self) -> Self {
                self.trunc()
            }
        }

        impl VRound for $typ {
            #[inline]
            fn v_round(self) -> Self {
                self.round()
            }
        }

        impl VSqrt for $typ {
            #[inline]
            fn v_sqrt(self) -> Self {
                self.sqrt()
            }
        }
    )*};
}

impl_fp_scalar!(f32, f64);
