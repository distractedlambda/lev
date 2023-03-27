use std::{
    marker::PhantomData,
    simd::{LaneCount, Simd, SupportedLaneCount},
};

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use super::{Fragment, SafeFragment};

macro_rules! op_decls {
    ($([$name:ident<$($typ_param:ident),* $(,)?>, $factory:ident])*) => {
        $(
        pub struct $name<$($typ_param),*>(PhantomData<($($typ_param,)*)>);

        impl<$($typ_param),*> Clone for $name<$($typ_param),*> {
            fn clone(&self) -> Self {
                $name(PhantomData)
            }
        }

        impl<$($typ_param),*> Copy for $name<$($typ_param),*> {}

        pub fn $factory<$($typ_param),*>() -> $name<$($typ_param),*> {
            $name(PhantomData)
        }
        )*
    }
}

op_decls!(
    [Abs<T>, abs]
    [Add<T>, add]
    [BitAnd<T>, bitand]
    [BitOr<T>, bitor]
    [BitXor<T>, bitxor]
    [Ceil<T>, ceil]
    [CopySign<T>, copysign]
    [CountLeadingOneBits<T>, count_leading_one_bits]
    [CountLeadingZeroBits<T>, count_leading_zero_bits]
    [CountOneBits<T>, count_one_bits]
    [CountTrailingZeroBits<T>, count_trailing_zero_bits]
    [Div<T>, div]
    [FMax<T>, fmax]
    [FMin<T>, fmin]
    [Floor<T>, floor]
    [Max<T>, max]
    [Min<T>, min]
    [Mul<T>, mul]
    [Nearest<T>, nearest]
    [Neg<T>, neg]
    [Not<T>, not]
    [Rem<T>, rem]
    [ReverseBits<T>, reverse_bits]
    [ReverseBytes<T>, reverse_bytes]
    [RotateLeft<L, R>, rotate_left]
    [RotateRight<L, R>, rotate_right]
    [Shl<L, R>, shl]
    [Shr<L, R>, shr]
    [Sqrt<T>, sqrt]
    [Sub<T>, sub]
    [Trunc<T>, trunc]
);

macro_rules! unary_op_impl {
    ($op_typ:ident, $val_typ:ty, $ins_func:ident) => {
        impl Fragment for $op_typ<$val_typ> {
            type Input = $val_typ;

            type Output = $val_typ;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: Value) -> Value {
                builder.ins().$ins_func(inputs)
            }
        }

        unsafe impl SafeFragment for $op_typ<$val_typ> {}
    };
}

macro_rules! simd_unary_op_impl {
    ($op_typ:ident, $val_typ:ty, $ins_func:ident) => {
        impl<const LANES: usize> Fragment for $op_typ<Simd<$val_typ, LANES>>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Input = Simd<$val_typ, LANES>;

            type Output = Simd<$val_typ, LANES>;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: Value) -> Value {
                builder.ins().$ins_func(inputs)
            }
        }

        unsafe impl<const LANES: usize> SafeFragment for $op_typ<Simd<$val_typ, LANES>> where
            LaneCount<LANES>: SupportedLaneCount
        {
        }
    };
}

macro_rules! vectorized_unary_op_impls {
    ($($op_typ:ident { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
        $($($(
        unary_op_impl!($op_typ, $val_typ, $ins_func);
        simd_unary_op_impl!($op_typ, $val_typ, $ins_func);
        )*)*)*
    }
}

vectorized_unary_op_impls! {
    Neg {
        ineg [i8, i16, i32, i64, isize]
        fneg [f32, f64]
    }

    Not {
        bnot [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    CountOneBits {
        popcnt [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    Sqrt {
        sqrt [f32, f64]
    }

    Abs {
        iabs [i8, i16, i32, i64, isize]
        fabs [f32, f64]
    }

    Ceil {
        ceil [f32, f64]
    }

    Floor {
        floor [f32, f64]
    }

    Trunc {
        trunc [f32, f64]
    }

    Nearest {
        nearest [f32, f64]
    }
}

macro_rules! scalar_unary_op_impls {
    ($($op_typ:ident { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
        $($($(unary_op_impl!($op_typ, $val_typ, $ins_func);)*)*)*
    }
}

scalar_unary_op_impls! {
    Not {
        bnot [bool]
    }

    ReverseBits {
        bitrev [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    ReverseBytes {
        bswap [i16, i32, i64, isize, u16, u32, u64, usize]
    }

    CountLeadingZeroBits {
        clz [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    CountLeadingOneBits {
        cls [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    CountTrailingZeroBits {
        ctz [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }
}

macro_rules! binary_op_impl {
    ($op_typ:ident, $val_typ:ty, $ins_func:ident) => {
        impl Fragment for $op_typ<$val_typ> {
            type Input = ($val_typ, $val_typ);

            type Output = $val_typ;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: (Value, Value)) -> Value {
                builder.ins().$ins_func(inputs.0, inputs.1)
            }
        }

        unsafe impl SafeFragment for $op_typ<$val_typ> {}
    };
}

macro_rules! simd_binary_op_impl {
    ($op_typ:ident, $val_typ:ty, $ins_func:ident) => {
        impl<const LANES: usize> Fragment for $op_typ<Simd<$val_typ, LANES>>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Input = (Simd<$val_typ, LANES>, Simd<$val_typ, LANES>);

            type Output = Simd<$val_typ, LANES>;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: (Value, Value)) -> Value {
                builder.ins().$ins_func(inputs.0, inputs.1)
            }
        }

        unsafe impl<const LANES: usize> SafeFragment for $op_typ<Simd<$val_typ, LANES>> where
            LaneCount<LANES>: SupportedLaneCount
        {
        }
    };
}

macro_rules! vectorized_binary_op_impls {
    ($($op_typ:ident { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
        $($($(
        binary_op_impl!($op_typ, $val_typ, $ins_func);
        simd_binary_op_impl!($op_typ, $val_typ, $ins_func);
        )*)*)*
    }
}

vectorized_binary_op_impls! {
    Add {
        iadd [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
        fadd [f32, f64]
    }

    Sub {
        isub [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
        fsub [f32, f64]
    }

    Mul {
        imul [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
        fmul [f32, f64]
    }

    Div {
        fdiv [f32, f64]
    }

    BitAnd {
        band [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    BitOr {
        bor [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    BitXor {
        bxor [i8, i16, i32, i64, isize, u8, u16, u32, u64, usize]
    }

    Min {
        smin [i8, i16, i32, i64, isize]
        umin [u8, u16, u32, u64, usize]
        fmin_pseudo [f32, f64]
    }

    Max {
        smax [i8, i16, i32, i64, isize]
        umax [u8, u16, u32, u64, usize]
        fmax_pseudo [f32, f64]
    }

    FMin {
        fmin [f32, f64]
    }

    FMax {
        fmax [f32, f64]
    }

    CopySign {
        fcopysign [f32, f64]
    }
}

macro_rules! scalar_binary_op_impls {
    ($($op_typ:ident { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
        $($($(binary_op_impl!($op_typ, $val_typ, $ins_func);)*)*)*
    }
}

scalar_binary_op_impls! {
    Div {
        sdiv [i8, i16, i32, i64, isize]
        udiv [u8, u16, u32, u64, usize]
    }

    Rem {
        srem [i8, i16, i32, i64, isize]
        urem [u8, u16, u32, u64, usize]
    }

    BitAnd {
        band [bool]
    }

    BitOr {
        bor [bool]
    }

    BitXor {
        bxor [bool]
    }
}

macro_rules! shift_op_impl {
    ($op_typ:ident, $lhs_typ:ty, $rhs_typ:ty, $ins_func:ident) => {
        impl Fragment for $op_typ<$lhs_typ, $rhs_typ> {
            type Input = ($lhs_typ, $rhs_typ);

            type Output = $lhs_typ;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: (Value, Value)) -> Value {
                builder.ins().$ins_func(inputs.0, inputs.1)
            }
        }

        unsafe impl SafeFragment for $op_typ<$lhs_typ, $rhs_typ> {}

        impl<const LANES: usize> Fragment for $op_typ<Simd<$lhs_typ, LANES>, $rhs_typ>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Input = (Simd<$lhs_typ, LANES>, $rhs_typ);

            type Output = Simd<$lhs_typ, LANES>;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: (Value, Value)) -> Value {
                builder.ins().$ins_func(inputs.0, inputs.1)
            }
        }

        unsafe impl<const LANES: usize> SafeFragment for $op_typ<Simd<$lhs_typ, LANES>, $rhs_typ>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {}
    };
}

macro_rules! shift_op_impls {
    ($($op_typ:ident { $($ins_func:ident [$($lhs_typ:ty),* $(,)?])* })*) => {
        $($($(
        shift_op_impl!($op_typ, $lhs_typ, u8, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, u16, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, u32, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, u64, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, usize, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, i8, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, i16, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, i32, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, i64, $ins_func);
        shift_op_impl!($op_typ, $lhs_typ, isize, $ins_func);
        )*)*)*
    }
}

shift_op_impls! {
    Shl {
        ishl [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize]
    }

    Shr {
        sshr [i8, i16, i32, i64, isize]
        ushr [u8, u16, u32, u64, usize]
    }

    RotateLeft {
        rotl [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize]
    }

    RotateRight {
        rotr [u8, u16, u32, u64, usize, i8, i16, i32, i64, isize]
    }
}
