use std::{
    marker::PhantomData,
    mem::{size_of, MaybeUninit},
    ptr::Pointee,
};

use cranelift_codegen::ir::{
    immediates::Offset32, types, Block, InstBuilder, MemFlags, Type, Value,
};
use cranelift_frontend::FunctionBuilder;
use memoffset::{offset_of, offset_of_tuple};

pub unsafe trait Fragment<Input: FragmentValue, Output: FragmentValue> {
    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: Input::IrValues,
    ) -> Output::IrValues;
}

unsafe impl<T: Fragment<Input, Output>, Input: FragmentValue, Output: FragmentValue, const N: usize> Fragment<[Input; N], [Output; N]> for [T; N] {
    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: Input::IrValues,
    ) -> Output::IrValues {
        self.each_ref()
            .zip(inputs)
            .map(|(f, i)| f.emit_ir(builder, i))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CommonInput<A, B>(A, B);

unsafe impl<A: Fragment<Input, AOutput>, B: Fragment<Input, BOutput>, Input: FragmentValue, AOutput: FragmentValue, BOutput: FragmentValue> Fragment<Input, (AOutput, BOutput)> for CommonInput<A, B> {
    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: Input::IrValues,
    ) -> (AOutput::IrValues, BOutput::IrValues) {
        let a = self.0.emit_ir(builder, inputs.clone());
        let b = self.1.emit_ir(builder, inputs);
        (a, b)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Compose<A, B>(A, B);

unsafe impl<A: Fragment, B: Fragment<Input = A::Output>> Fragment for Compose<A, B> {
    type Input = A::Input;

    type Output = B::Output;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        let a = self.0.emit_ir(builder, inputs);
        self.1.emit_ir(builder, a)
    }
}

#[derive(Debug)]
pub struct Identity<T>(PhantomData<T>);

impl<T> Clone for Identity<T> {
    fn clone(&self) -> Self {
        Identity(PhantomData)
    }
}

impl<T> Copy for Identity<T> {}

unsafe impl<T: FragmentValue> Fragment for Identity<T> {
    type Input = T;

    type Output = T;

    fn emit_ir(
        &self,
        _builder: &mut FunctionBuilder,
        inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        inputs
    }
}

macro_rules! impl_fragment_for_signed_int {
    ($([$typ:ty, $ir_typ:expr])*) => {
        $(
        unsafe impl Fragment for $typ {
            type Input = ();

            type Output = $typ;

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                _inputs: <Self::Input as FragmentValue>::IrValues,
            ) -> <Self::Output as FragmentValue>::IrValues {
                builder.ins().iconst($ir_typ, *self as i64)
            }
        }
        )*
    }
}

impl_fragment_for_signed_int! {
    [i8, types::I8]
    [i16, types::I16]
    [i32, types::I32]
    [i64, types::I64]
    [isize, ADDRESS_TYPE]
}

macro_rules! impl_fragment_for_unsigned_int {
    ($([$typ:ty, $ir_typ:expr])*) => {
        $(
        unsafe impl Fragment for $typ {
            type Input = ();

            type Output = $typ;

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                _inputs: <Self::Input as FragmentValue>::IrValues,
            ) -> <Self::Output as FragmentValue>::IrValues {
                builder.ins().iconst($ir_typ, *self as u64 as i64)
            }
        }
        )*
    }
}

impl_fragment_for_unsigned_int! {
    [u8, types::I8]
    [u16, types::I16]
    [u32, types::I32]
    [u64, types::I64]
    [usize, ADDRESS_TYPE]
}

unsafe impl Fragment for f32 {
    type Input = ();

    type Output = f32;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        _inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        builder.ins().f32const(*self)
    }
}

unsafe impl Fragment for f64 {
    type Input = ();

    type Output = f64;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        _inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        builder.ins().f64const(*self)
    }
}

unsafe impl Fragment for () {
    type Input = ();

    type Output = ();

    fn emit_ir(
        &self,
        _builder: &mut FunctionBuilder,
        _inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
    }
}

pub unsafe trait FragmentValue {
    type IrValues: Clone;

    fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues;

    fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>);

    fn emit_load(builder: &mut FunctionBuilder, address: Value, offset: Offset32)
        -> Self::IrValues;

    fn emit_store(
        builder: &mut FunctionBuilder,
        address: Value,
        values: Self::IrValues,
        offset: Offset32,
    );
}

unsafe impl FragmentValue for () {
    type IrValues = ();

    fn append_block_params(_builder: &mut FunctionBuilder, _block: Block) -> Self::IrValues {}

    fn unpack_ir_values(_values: Self::IrValues, _dst: &mut impl Extend<Value>) {}

    fn emit_load(
        _builder: &mut FunctionBuilder,
        _address: Value,
        _offset: Offset32,
    ) -> Self::IrValues {
    }

    fn emit_store(
        _builder: &mut FunctionBuilder,
        _address: Value,
        _values: Self::IrValues,
        _offset: Offset32,
    ) {
    }
}

macro_rules! impl_scalar_fragment_value {
    ($([$typ:ty, $ir_typ:expr])*) => {
        $(
        unsafe impl FragmentValue for $typ {
            type IrValues = Value;

            fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues {
                builder.append_block_param(block, $ir_typ)
            }

            fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>) {
                dst.extend_one(values)
            }

            fn emit_load(
                builder: &mut FunctionBuilder,
                address: Value,
                offset: Offset32,
            ) -> Self::IrValues {
                builder
                    .ins()
                    .load($ir_typ, MemFlags::trusted(), address, offset)
            }

            fn emit_store(
                builder: &mut FunctionBuilder,
                address: Value,
                values: Self::IrValues,
                offset: Offset32,
            ) {
                builder
                    .ins()
                    .store(MemFlags::trusted(), values, address, offset);
            }
        }
        )*
    }
}

impl_scalar_fragment_value! {
    [i8, types::I8]
    [i16, types::I16]
    [i32, types::I32]
    [i64, types::I64]
    [isize, ADDRESS_TYPE]
    [u8, types::I8]
    [u16, types::I16]
    [u32, types::I32]
    [u64, types::I64]
    [usize, ADDRESS_TYPE]
    [f32, types::F32]
    [f64, types::F64]
}

macro_rules! tuple_impl {
    ($([$typ_param:ident, $idx:tt])*) => {
        unsafe impl<$($typ_param: FragmentValue,)*> FragmentValue for ($($typ_param,)*) {
            type IrValues = ($($typ_param::IrValues,)*);

            fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues {
                ($($typ_param::append_block_params(builder, block),)*)
            }

            fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>) {
                $($typ_param::unpack_ir_values(values.$idx, dst);)*
            }

            fn emit_load(
                builder: &mut FunctionBuilder,
                address: Value,
                offset: Offset32,
            ) -> Self::IrValues {
                ($($typ_param::emit_load(builder, address, combine_offset(offset, offset_of_tuple!(Self, $idx))),)*)
            }

            fn emit_store(
                builder: &mut FunctionBuilder,
                address: Value,
                values: Self::IrValues,
                offset: Offset32,
            ) {
                $($typ_param::emit_store(builder, address, values.$idx, combine_offset(offset, offset_of_tuple!(Self, $idx)));)*
            }
        }

        unsafe impl<$($typ_param: Fragment,)*> Fragment for ($($typ_param,)*) {
            type Input = ($($typ_param::Input,)*);

            type Output = ($($typ_param::Output,)*);

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                inputs: <Self::Input as FragmentValue>::IrValues,
            ) -> <Self::Output as FragmentValue>::IrValues {
                ($(self.$idx.emit_ir(builder, inputs.$idx),)*)
            }
        }
    }
}

macro_rules! tuple_impls_rec {
    (($($collected:tt)*) [$typ_param:ident, $idx:tt] $($rest:tt)*) => {
        tuple_impl!($($collected)* [$typ_param, $idx]);
        tuple_impls_rec!(($($collected)* [$typ_param, $idx]) $($rest)*);
    };

    (($($collected:tt)*)) => {};
}

macro_rules! tuple_impls {
    ($($params:tt)*) => {
        tuple_impls_rec!(() $($params)*);
    }
}

tuple_impls! {
    [A, 0]
    [B, 1]
    [C, 2]
    [D, 3]
    [E, 4]
    [F, 5]
    [G, 6]
    [H, 7]
    [I, 8]
    [J, 9]
    [K, 10]
    [L, 11]
    [M, 12]
    [N, 13]
    [O, 14]
    [P, 15]
    [Q, 16]
    [R, 17]
    [S, 18]
    [T, 19]
    [U, 20]
    [V, 21]
    [W, 22]
    [X, 23]
    [Y, 24]
    [Z, 25]
}

unsafe impl<T: FragmentValue, const N: usize> FragmentValue for [T; N] {
    type IrValues = [T::IrValues; N];

    fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues {
        new_array_by(|_| T::append_block_params(builder, block))
    }

    fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>) {
        for v in values {
            T::unpack_ir_values(v, dst);
        }
    }

    fn emit_load(
        builder: &mut FunctionBuilder,
        address: Value,
        offset: Offset32,
    ) -> Self::IrValues {
        new_array_by(|i| {
            T::emit_load(
                builder,
                address,
                combine_offset(offset, i.checked_mul(size_of::<T>()).unwrap()),
            )
        })
    }

    fn emit_store(
        builder: &mut FunctionBuilder,
        address: Value,
        values: Self::IrValues,
        offset: Offset32,
    ) {
        for (i, v) in values.into_iter().enumerate() {
            T::emit_store(
                builder,
                address,
                v,
                combine_offset(offset, i.checked_mul(size_of::<T>()).unwrap()),
            )
        }
    }
}

macro_rules! impl_pointer_fragment_value {
    ($([$typ_var:ident, $typ:ty])*) => {
        $(
        unsafe impl<$typ_var: ?Sized> FragmentValue for $typ
        where
            <$typ_var as Pointee>::Metadata: FragmentValue,
        {
            type IrValues = PointerIrValues<$typ_var>;

            fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues {
                append_pointer_block_params::<$typ_var>(builder, block)
            }

            fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>) {
                unpack_pointer_ir_values::<$typ_var>(values, dst)
            }

            fn emit_load(
                builder: &mut FunctionBuilder,
                address: Value,
                offset: Offset32,
            ) -> Self::IrValues {
                emit_pointer_load::<$typ_var>(builder, address, offset)
            }

            fn emit_store(
                builder: &mut FunctionBuilder,
                address: Value,
                values: Self::IrValues,
                offset: Offset32,
            ) {
                emit_pointer_store::<$typ_var>(builder, address, values, offset)
            }
        }
        )*
    }
}

impl_pointer_fragment_value! {
    [T, &T]
    [T, &mut T]
    [T, *const T]
    [T, *mut T]
}

#[cfg(target_pointer_width = "32")]
const ADDRESS_TYPE: Type = types::I32;

#[cfg(target_pointer_width = "64")]
const ADDRESS_TYPE: Type = types::I64;

/// Copied from core::ptr::metadata sources
#[repr(C)]
struct PtrComponents<T: ?Sized> {
    data_address: *const (),
    metadata: <T as Pointee>::Metadata,
}

#[inline]
fn combine_offset(base: Offset32, adjustment: usize) -> Offset32 {
    base.try_add_i64(adjustment.try_into().unwrap()).unwrap()
}

fn new_array_by<T, const N: usize>(mut f: impl FnMut(usize) -> T) -> [T; N] {
    unsafe {
        let mut result = MaybeUninit::uninit_array();

        for i in 0..N {
            // TODO: drop things on panic
            result[i].write(f(i));
        }

        MaybeUninit::array_assume_init(result)
    }
}

type PointerIrValues<T> = (Value, <<T as Pointee>::Metadata as FragmentValue>::IrValues);

fn append_pointer_block_params<T: ?Sized>(
    builder: &mut FunctionBuilder,
    block: Block,
) -> PointerIrValues<T>
where
    <T as Pointee>::Metadata: FragmentValue,
{
    let address = usize::append_block_params(builder, block);
    let metadata = <<T as Pointee>::Metadata as FragmentValue>::append_block_params(builder, block);
    (address, metadata)
}

fn unpack_pointer_ir_values<T: ?Sized>(values: PointerIrValues<T>, dst: &mut impl Extend<Value>)
where
    <T as Pointee>::Metadata: FragmentValue,
{
    usize::unpack_ir_values(values.0, dst);
    <<T as Pointee>::Metadata as FragmentValue>::unpack_ir_values(values.1, dst);
}

fn emit_pointer_load<T: ?Sized>(
    builder: &mut FunctionBuilder,
    address: Value,
    offset: Offset32,
) -> PointerIrValues<T>
where
    <T as Pointee>::Metadata: FragmentValue,
{
    let address = usize::emit_load(
        builder,
        address,
        combine_offset(offset, offset_of!(PtrComponents<T>, data_address)),
    );

    let metadata = <<T as Pointee>::Metadata as FragmentValue>::emit_load(
        builder,
        address,
        combine_offset(offset, offset_of!(PtrComponents<T>, metadata)),
    );

    (address, metadata)
}

fn emit_pointer_store<T: ?Sized>(
    builder: &mut FunctionBuilder,
    address: Value,
    values: PointerIrValues<T>,
    offset: Offset32,
) where
    <T as Pointee>::Metadata: FragmentValue,
{
    usize::emit_store(
        builder,
        address,
        values.0,
        combine_offset(offset, offset_of!(PtrComponents<T>, data_address)),
    );

    <<T as Pointee>::Metadata as FragmentValue>::emit_store(
        builder,
        address,
        values.1,
        combine_offset(offset, offset_of!(PtrComponents<T>, metadata)),
    );
}