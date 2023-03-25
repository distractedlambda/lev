use std::simd::Simd;

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use super::Fragment;

#[derive(Clone, Copy, Debug)]
pub struct Add;

#[derive(Clone, Copy, Debug)]
pub struct Sub;

#[derive(Clone, Copy, Debug)]
pub struct Mul;

#[derive(Clone, Copy, Debug)]
pub struct Div;

#[derive(Clone, Copy, Debug)]
pub struct Rem;

#[derive(Clone, Copy, Debug)]
pub struct BitAnd;

#[derive(Clone, Copy, Debug)]
pub struct BitOr;

#[derive(Clone, Copy, Debug)]
pub struct BitXor;

#[derive(Clone, Copy, Debug)]
pub struct Min;

#[derive(Clone, Copy, Debug)]
pub struct Max;

#[derive(Clone, Copy, Debug)]
pub struct FMin;

#[derive(Clone, Copy, Debug)]
pub struct FMax;

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
}
