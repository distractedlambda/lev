use std::simd::Simd;

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use super::Fragment;

macro_rules! op_decls {
    ($($name:ident),* $(,)?) => {
        $(
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name;
        )*
    }
}

op_decls!(
    Abs,
    Add,
    BitAnd,
    BitOr,
    BitXor,
    Ceil,
    CopySign,
    CountLeadingOneBits,
    CountLeadingZeroBits,
    CountOneBits,
    CountTrailingZeroBits,
    Div,
    FMax,
    FMin,
    Floor,
    Max,
    Min,
    Mul,
    Nearest,
    Neg,
    Not,
    Rem,
    ReverseBits,
    ReverseBytes,
    Sqrt,
    Sub,
    Trunc,
    Shl,
    Shr,
    RotateLeft,
    RotateRight,
);

macro_rules! unary_op_impl {
    ($op_typ:ty, $val_typ:ty, $ins_func:ident) => {
        unsafe impl Fragment<$val_typ> for $op_typ {
            type Output = $val_typ;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: Value) -> Value {
                builder.ins().$ins_func(inputs)
            }
        }
    };
}

macro_rules! vectorized_unary_op_impls {
    ($($op_typ:ty { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
        $($($(
        unary_op_impl!($op_typ, $val_typ, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 1>, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 2>, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 4>, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 8>, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 16>, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 32>, $ins_func);
        unary_op_impl!($op_typ, Simd<$val_typ, 64>, $ins_func);
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
    ($($op_typ:ty { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
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
    ($op_typ:ty, $val_typ:ty, $ins_func:ident) => {
        unsafe impl Fragment<($val_typ, $val_typ)> for $op_typ {
            type Output = $val_typ;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: (Value, Value)) -> Value {
                builder.ins().$ins_func(inputs.0, inputs.1)
            }
        }
    };
}

macro_rules! vectorized_binary_op_impls {
    ($($op_typ:ty { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
        $($($(
        binary_op_impl!($op_typ, $val_typ, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 1>, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 2>, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 4>, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 8>, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 16>, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 32>, $ins_func);
        binary_op_impl!($op_typ, Simd<$val_typ, 64>, $ins_func);
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
    ($($op_typ:ty { $($ins_func:ident [$($val_typ:ty),* $(,)?])* })*) => {
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
    ($op_typ:ty, $lhs_typ:ty, $rhs_typ:ty, $ins_func:ident) => {
        unsafe impl Fragment<($lhs_typ, $rhs_typ)> for $op_typ {
            type Output = $lhs_typ;

            fn emit_ir(&self, builder: &mut FunctionBuilder, inputs: (Value, Value)) -> Value {
                builder.ins().$ins_func(inputs.0, inputs.1)
            }
        }
    }
}

macro_rules! shift_op_impls_helper {
    ($op_typ:ty, $lhs_typ:ty, $ins_func:ident) => {
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
    }
}

macro_rules! shift_op_impls {
    ($($op_typ:ty { $($ins_func:ident [$($lhs_typ:ty),* $(,)?])* })*) => {
        $($($(
        shift_op_impls_helper!($op_typ, $lhs_typ, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 1>, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 2>, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 4>, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 8>, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 16>, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 32>, $ins_func);
        shift_op_impls_helper!($op_typ, Simd<$lhs_typ, 64>, $ins_func);
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
