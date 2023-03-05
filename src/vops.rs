pub trait VAbs {
    type Output;

    fn vabs(self) -> Self::Output;
}

pub trait VSqrt {
    type Output;

    fn vsqrt(self) -> Self::Output;
}

pub trait VTrunc {
    type Output;

    fn vtrunc(self) -> Self::Output;
}

pub trait VCls {
    type Output;

    fn vcls(self) -> Self::Output;
}

pub trait VClz {
    type Output;

    fn vclz(self) -> Self::Output;
}

pub trait VPopcount {
    type Output;

    fn vpopcount(self) -> Self::Output;
}

pub trait VEq<RHS = Self> {
    type Output;

    fn veq(self, rhs: RHS) -> Self::Output;
}

pub trait VNe<RHS = Self> {
    type Output;

    fn vne(self, rhs: RHS) -> Self::Output;
}

pub trait VLt<RHS = Self> {
    type Output;

    fn vlt(self, rhs: RHS) -> Self::Output;
}

pub trait VGt<RHS = Self> {
    type Output;

    fn vgt(self, rhs: RHS) -> Self::Output;
}

pub trait VLe<RHS = Self> {
    type Output;

    fn vle(self, rhs: RHS) -> Self::Output;
}

pub trait VGe<RHS = Self> {
    type Output;

    fn vge(self, rhs: RHS) -> Self::Output;
}

pub trait VFloor {
    type Output;

    fn vfloor(self) -> Self::Output;
}

pub trait VCeil {
    type Output;

    fn vceil(self) -> Self::Output;
}

pub trait VRound {
    type Output;

    fn vround(self) -> Self::Output;
}

pub trait VSum {
    type Output;

    fn vsum(self) -> Self::Output;
}

pub trait VProduct {
    type Output;

    fn vproduct(self) -> Self::Output;
}

pub trait VAny {
    type Output;

    fn vany(self) -> Self::Output;
}

pub trait VAll {
    type Output;

    fn vall(self) -> Self::Output;
}

pub trait VMinimum {
    type Output;

    fn vminimum(self) -> Self::Output;
}

pub trait VMaximum {
    type Output;

    fn vmaximum(self) -> Self::Output;
}

pub trait VMin<RHS = Self> {
    type Output;

    fn vmin(self, rhs: RHS) -> Self::Output;
}

pub trait VMax<RHS = Self> {
    type Output;

    fn vmax(self, rhs: RHS) -> Self::Output;
}

pub trait VZero {
    fn vzero() -> Self;
}

pub trait VOne {
    fn vone() -> Self;
}

impl VAny for bool {
    type Output = bool;

    #[inline]
    fn vany(self) -> Self::Output {
        self
    }
}

impl VAll for bool {
    type Output = bool;

    #[inline]
    fn vall(self) -> Self::Output {
        self
    }
}

macro_rules! impl_common_ops {
    ($($typ:ty),* $(,)?) => {
        $(
            impl VSum for $typ {
                type Output = Self;

                #[inline]
                fn vsum(self) -> Self::Output {
                    self
                }
            }

            impl VProduct for $typ {
                type Output = Self;

                #[inline]
                fn vproduct(self) -> Self::Output {
                    self
                }
            }

            impl VMinimum for $typ {
                type Output = Self;

                #[inline]
                fn vminimum(self) -> Self::Output {
                    self
                }
            }

            impl VMaximum for $typ {
                type Output = Self;

                #[inline]
                fn vmaximum(self) -> Self::Output {
                    self
                }
            }

            impl VMin for $typ {
                type Output = Self;

                #[inline]
                fn vmin(self, rhs: Self) -> Self::Output {
                    self.min(rhs)
                }
            }

            impl VMax for $typ {
                type Output = Self;

                #[inline]
                fn vmax(self, rhs: Self) -> Self::Output {
                    self.max(rhs)
                }
            }
        )*
    }
}

impl_common_ops!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_signed_ops {
    ($($typ:ty),* $(,)?) => {
        $(
            impl VAbs for $typ {
                type Output = Self;

                #[inline]
                fn vabs(self) -> Self::Output {
                    self.abs()
                }
            }
        )*
    }
}

impl_signed_ops!(i8, i16, i32, i64, isize, f32, f64);

macro_rules! impl_fp_ops {
    ($($typ:ty),* $(,)?) => {
        $(
            impl VSqrt for $typ {
                type Output = Self;

                #[inline]
                fn vsqrt(self) -> Self::Output {
                    self.sqrt()
                }
            }

            impl VTrunc for $typ {
                type Output = Self;

                #[inline]
                fn vtrunc(self) -> Self::Output {
                    self.trunc()
                }
            }

            impl VFloor for $typ {
                type Output = Self;

                #[inline]
                fn vfloor(self) -> Self::Output {
                    self.floor()
                }
            }

            impl VCeil for $typ {
                type Output = Self;

                #[inline]
                fn vceil(self) -> Self::Output {
                    self.ceil()
                }
            }

            impl VRound for $typ {
                type Output = Self;

                #[inline]
                fn vround(self) -> Self::Output {
                    self.round()
                }
            }
        )*
    }
}

impl_fp_ops!(f32, f64);
