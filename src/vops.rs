use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

pub trait VMask<T>:
    Sized
    + Not<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
{
    fn v_select(self, if_true: T, if_false: T) -> T;
}

pub trait Vectorize: Sized {
    type Scalar;

    type Mask: VMask<Self>;
}

pub trait VZero: Vectorize {
    fn v_zero() -> Self;
}

pub trait VOne: Vectorize {
    fn v_one() -> Self;
}

pub trait VPartialEq: Vectorize {
    fn v_eq(self, other: Self) -> Self::Mask;

    fn v_ne(self, other: Self) -> Self::Mask {
        !self.v_eq(other)
    }
}

pub trait VEq: VPartialEq {}

pub trait VPartialOrd: VPartialEq {
    fn v_lt(self, other: Self) -> Self::Mask;

    fn v_le(self, other: Self) -> Self::Mask;

    fn v_gt(self, other: Self) -> Self::Mask {
        !self.v_le(other)
    }

    fn v_ge(self, other: Self) -> Self::Mask {
        !self.v_lt(other)
    }

    fn v_min(self, other: Self) -> Self;

    fn v_max(self, other: Self) -> Self;

    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.v_max(min).v_min(max)
    }
}

pub trait VOrd: VPartialOrd + VEq {}

pub trait VNum:
    VZero
    + VOne
    + VPartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
{
}

pub trait VSigned: VNum + Neg<Output = Self> {}

pub trait VInt:
    VNum
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + BitXorAssign
{
}

pub trait VSignedInt: VSigned + VInt {}

pub trait VFloat: VSigned {}

impl<T> VMask<T> for bool {
    #[inline]
    fn v_select(self, if_true: T, if_false: T) -> T {
        if self {
            if_true
        } else {
            if_false
        }
    }
}

impl Vectorize for u8 {
    type Scalar = u8;

    type Mask = bool;
}

impl Vectorize for u16 {
    type Scalar = u16;

    type Mask = bool;
}

impl Vectorize for u32 {
    type Scalar = u32;

    type Mask = bool;
}

impl Vectorize for u64 {
    type Scalar = u64;

    type Mask = bool;
}

impl Vectorize for usize {
    type Scalar = usize;

    type Mask = bool;
}

impl Vectorize for i8 {
    type Scalar = i8;

    type Mask = bool;
}

impl Vectorize for i16 {
    type Scalar = i16;

    type Mask = bool;
}

impl Vectorize for i32 {
    type Scalar = i32;

    type Mask = bool;
}

impl Vectorize for i64 {
    type Scalar = i64;

    type Mask = bool;
}

impl Vectorize for isize {
    type Scalar = isize;

    type Mask = bool;
}

impl Vectorize for f32 {
    type Scalar = f32;

    type Mask = bool;
}

impl Vectorize for f64 {
    type Scalar = f64;

    type Mask = bool;
}

impl VZero for u8 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for u16 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for u32 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for u64 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for usize {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for i8 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for i16 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for i32 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for i64 {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for isize {
    #[inline]
    fn v_zero() -> Self {
        0
    }
}

impl VZero for f32 {
    #[inline]
    fn v_zero() -> Self {
        0.0
    }
}

impl VZero for f64 {
    #[inline]
    fn v_zero() -> Self {
        0.0
    }
}

impl VOne for u8 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for u16 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for u32 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for u64 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for usize {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for i8 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for i16 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for i32 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for i64 {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for isize {
    #[inline]
    fn v_one() -> Self {
        1
    }
}

impl VOne for f32 {
    #[inline]
    fn v_one() -> Self {
        1.0
    }
}

impl VOne for f64 {
    #[inline]
    fn v_one() -> Self {
        1.0
    }
}

impl VPartialEq for u8 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for u16 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for u32 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for u64 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for usize {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for i8 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for i16 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for i32 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for i64 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for isize {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for f32 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VPartialEq for f64 {
    #[inline]
    fn v_eq(self, other: Self) -> Self::Mask {
        self == other
    }

    #[inline]
    fn v_ne(self, other: Self) -> Self::Mask {
        self != other
    }
}

impl VEq for u8 {}

impl VEq for u16 {}

impl VEq for u32 {}

impl VEq for u64 {}

impl VEq for usize {}

impl VEq for i8 {}

impl VEq for i16 {}

impl VEq for i32 {}

impl VEq for i64 {}

impl VEq for isize {}

impl VPartialOrd for u8 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for u16 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for u32 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for u64 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for usize {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for i8 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for i16 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for i32 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for i64 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for isize {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

    #[inline]
    fn v_clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }
}

impl VPartialOrd for f32 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

impl VPartialOrd for f64 {
    #[inline]
    fn v_lt(self, other: Self) -> Self::Mask {
        self < other
    }

    #[inline]
    fn v_le(self, other: Self) -> Self::Mask {
        self <= other
    }

    #[inline]
    fn v_gt(self, other: Self) -> Self::Mask {
        self > other
    }

    #[inline]
    fn v_ge(self, other: Self) -> Self::Mask {
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

impl VOrd for u8 {}

impl VOrd for u16 {}

impl VOrd for u32 {}

impl VOrd for u64 {}

impl VOrd for usize {}

impl VOrd for i8 {}

impl VOrd for i16 {}

impl VOrd for i32 {}

impl VOrd for i64 {}

impl VOrd for isize {}

impl VNum for u8 {}

impl VNum for u16 {}

impl VNum for u32 {}

impl VNum for u64 {}

impl VNum for usize {}

impl VNum for i8 {}

impl VNum for i16 {}

impl VNum for i32 {}

impl VNum for i64 {}

impl VNum for isize {}

impl VNum for f32 {}

impl VNum for f64 {}

impl VSigned for i8 {}

impl VSigned for i16 {}

impl VSigned for i32 {}

impl VSigned for i64 {}

impl VSigned for isize {}

impl VSigned for f32 {}

impl VSigned for f64 {}

impl VInt for u8 {}

impl VInt for u16 {}

impl VInt for u32 {}

impl VInt for u64 {}

impl VInt for usize {}

impl VInt for i8 {}

impl VInt for i16 {}

impl VInt for i32 {}

impl VInt for i64 {}

impl VInt for isize {}

impl VSignedInt for i8 {}

impl VSignedInt for i16 {}

impl VSignedInt for i32 {}

impl VSignedInt for i64 {}

impl VSignedInt for isize {}

impl VFloat for f32 {}

impl VFloat for f64 {}
