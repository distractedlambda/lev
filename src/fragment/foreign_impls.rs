use std::{
    mem::{size_of, MaybeUninit},
    ops::Deref,
    pin::Pin,
    ptr::Pointee,
    rc::Rc,
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
    sync::Arc,
};

use cranelift_codegen::ir::{
    immediates::{Imm64, Offset32},
    types, Block, InstBuilder, MemFlags, Type, Value,
};
use cranelift_frontend::FunctionBuilder;
use memoffset::{offset_of, offset_of_tuple};

use super::{Fragment, FragmentValue, PrimitiveValue, SafeFragment, ADDRESS_TYPE};

unsafe impl<T: SimdElement + PrimitiveValue, const LANES: usize> PrimitiveValue for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn ir_type() -> Type {
        T::ir_type().by(LANES.try_into().unwrap()).unwrap()
    }
}

impl Fragment for () {
    type Input = ();

    type Output = ();

    fn emit_ir(&self, _builder: &mut FunctionBuilder, _inputs: ()) {}
}

unsafe impl SafeFragment for () {}

impl Fragment for bool {
    type Input = ();

    type Output = bool;

    fn emit_ir(&self, builder: &mut FunctionBuilder, _inputs: ()) -> Value {
        builder
            .ins()
            .iconst(types::I8, if *self { Imm64::new(1) } else { Imm64::new(0) })
    }
}

unsafe impl SafeFragment for bool {}

impl Fragment for f32 {
    type Input = ();

    type Output = f32;

    fn emit_ir(&self, builder: &mut FunctionBuilder, _inputs: ()) -> Value {
        builder.ins().f32const(*self)
    }
}

unsafe impl SafeFragment for f32 {}

impl Fragment for f64 {
    type Input = ();

    type Output = f64;

    fn emit_ir(&self, builder: &mut FunctionBuilder, _inputs: ()) -> Value {
        builder.ins().f64const(*self)
    }
}

unsafe impl SafeFragment for f64 {}

impl<T: Fragment + ?Sized> Fragment for &T {
    type Input = T::Input;

    type Output = T::Output;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        (*self).emit_ir(builder, inputs)
    }
}

unsafe impl<T: SafeFragment + ?Sized> SafeFragment for &T {}

impl<T: Deref> Fragment for Pin<T>
where
    T::Target: Fragment,
{
    type Input = <T::Target as Fragment>::Input;

    type Output = <T::Target as Fragment>::Output;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        self.deref().emit_ir(builder, inputs)
    }
}

unsafe impl<T: Deref> SafeFragment for Pin<T> where T::Target: SafeFragment {}

impl<T: Fragment, const N: usize> Fragment for [T; N] {
    type Input = [T::Input; N];

    type Output = [T::Output; N];

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: [<T::Input as FragmentValue>::IrValues; N],
    ) -> [<T::Output as FragmentValue>::IrValues; N] {
        self.each_ref()
            .zip(inputs)
            .map(|(f, i)| f.emit_ir(builder, i))
    }
}

unsafe impl<T: SafeFragment, const N: usize> SafeFragment for [T; N] {}

impl<T: PrimitiveValue + SimdElement, const LANES: usize> Fragment for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Input = ();

    type Output = Self;

    fn emit_ir(&self, builder: &mut FunctionBuilder, _inputs: ()) -> Value {
        let first = self[0].emit_ir(builder, ());

        let mut vector = builder
            .ins()
            .scalar_to_vector(<Self as PrimitiveValue>::ir_type(), first);

        for i in 1..LANES {
            let element = self[i].emit_ir(builder, ());

            vector = builder
                .ins()
                .insertlane(vector, element, u8::try_from(i).unwrap());
        }

        vector
    }
}

unsafe impl<T: PrimitiveValue + SimdElement, const LANES: usize> SafeFragment for Simd<T, LANES> where
    LaneCount<LANES>: SupportedLaneCount
{
}

unsafe impl FragmentValue for () {
    type IrValues = ();

    fn append_block_params(_builder: &mut FunctionBuilder, _block: Block) -> Self::IrValues {}

    fn unpack_ir_values(_values: Self::IrValues, _dst: &mut impl Extend<Value>) {}

    fn emit_load(
        _builder: &mut FunctionBuilder,
        _flags: MemFlags,
        _address: Value,
        _offset: Offset32,
    ) {
    }

    fn emit_store(
        _builder: &mut FunctionBuilder,
        _flags: MemFlags,
        _address: Value,
        _values: Self::IrValues,
        _offset: Offset32,
    ) {
    }
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
        flags: MemFlags,
        address: Value,
        offset: Offset32,
    ) -> Self::IrValues {
        new_array_by(|i| {
            T::emit_load(
                builder,
                flags,
                address,
                combine_offset(offset, i.checked_mul(size_of::<T>()).unwrap()),
            )
        })
    }

    fn emit_store(
        builder: &mut FunctionBuilder,
        flags: MemFlags,
        address: Value,
        values: Self::IrValues,
        offset: Offset32,
    ) {
        for (i, v) in values.into_iter().enumerate() {
            T::emit_store(
                builder,
                flags,
                address,
                v,
                combine_offset(offset, i.checked_mul(size_of::<T>()).unwrap()),
            )
        }
    }
}

unsafe impl<T: PrimitiveValue + SimdElement, const LANES: usize> FragmentValue for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type IrValues = Value;

    fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues {
        builder.append_block_param(block, Self::ir_type())
    }

    fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>) {
        dst.extend_one(values)
    }

    fn emit_load(
        builder: &mut FunctionBuilder,
        flags: MemFlags,
        address: Value,
        offset: Offset32,
    ) -> Self::IrValues {
        builder.ins().load(Self::ir_type(), flags, address, offset)
    }

    fn emit_store(
        builder: &mut FunctionBuilder,
        flags: MemFlags,
        address: Value,
        values: Self::IrValues,
        offset: Offset32,
    ) {
        builder.ins().store(flags, values, address, offset);
    }
}

macro_rules! deref_fragment_impls {
    ($($container:ident),* $(,)?) => {
        $(
        impl<T: Fragment + ?Sized> Fragment for $container<T> {
            type Input = T::Input;

            type Output = T::Output;

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                inputs: <Self::Input as FragmentValue>::IrValues,
            ) -> <Self::Output as FragmentValue>::IrValues {
                self.deref().emit_ir(builder, inputs)
            }
        }

        unsafe impl<T: SafeFragment + ?Sized> SafeFragment for $container<T> {}
        )*
    }
}

deref_fragment_impls!(Arc, Box, Rc);

macro_rules! signed_int_fragment_impls {
    ($($typ:ty),* $(,)?) => {
        $(
        impl Fragment for $typ {
            type Input = ();

            type Output = $typ;

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                _inputs: (),
            ) -> Value {
                builder.ins().iconst(<$typ as PrimitiveValue>::ir_type(), *self as i64)
            }
        }

        unsafe impl SafeFragment for $typ {}
        )*
    }
}

signed_int_fragment_impls!(i8, i16, i32, i64, isize);

macro_rules! unsigned_int_fragment_impls {
    ($($typ:ty),* $(,)?) => {
        $(
        impl Fragment for $typ {
            type Input = ();

            type Output = $typ;

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                _inputs: (),
            ) -> Value {
                builder.ins().iconst(<$typ as PrimitiveValue>::ir_type(), *self as u64 as i64)
            }
        }

        unsafe impl SafeFragment for $typ {}
        )*
    }
}

unsigned_int_fragment_impls!(u8, u16, u32, u64, usize);

macro_rules! primitive_value_impls {
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
                flags: MemFlags,
                address: Value,
                offset: Offset32,
            ) -> Self::IrValues {
                builder.ins().load($ir_typ, flags, address, offset)
            }

            fn emit_store(
                builder: &mut FunctionBuilder,
                flags: MemFlags,
                address: Value,
                values: Self::IrValues,
                offset: Offset32,
            ) {
                builder.ins().store(flags, values, address, offset);
            }
        }

        unsafe impl PrimitiveValue for $typ {
            fn ir_type() -> Type {
                $ir_typ
            }
        }
        )*
    }
}

primitive_value_impls! {
    [bool, types::I8]
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
                flags: MemFlags,
                address: Value,
                offset: Offset32,
            ) -> Self::IrValues {
                ($($typ_param::emit_load(builder, flags, address, combine_offset(offset, offset_of_tuple!(Self, $idx))),)*)
            }

            fn emit_store(
                builder: &mut FunctionBuilder,
                flags: MemFlags,
                address: Value,
                values: Self::IrValues,
                offset: Offset32,
            ) {
                $($typ_param::emit_store(builder, flags, address, values.$idx, combine_offset(offset, offset_of_tuple!(Self, $idx)));)*
            }
        }

        impl<$($typ_param: Fragment,)*> Fragment for ($($typ_param,)*) {
            type Input = ($($typ_param::Input,)*);

            type Output = ($($typ_param::Output,)*);

            fn emit_ir(
                &self,
                builder: &mut FunctionBuilder,
                inputs: ($(<$typ_param::Input as FragmentValue>::IrValues,)*),
            ) -> ($(<$typ_param::Output as FragmentValue>::IrValues,)*) {
                ($(self.$idx.emit_ir(builder, inputs.$idx),)*)
            }
        }

        unsafe impl<$($typ_param: SafeFragment,)*> SafeFragment for ($($typ_param,)*) {}
    }
}

macro_rules! tuple_impls_rec {
    (($($collected:tt)*) $first:tt $($rest:tt)*) => {
        tuple_impl!($($collected)* $first);
        tuple_impls_rec!(($($collected)* $first) $($rest)*);
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

macro_rules! pointer_value_impls {
    ($([$typ_var:ident, $typ:ty])*) => {
        $(
        unsafe impl<$typ_var: ?Sized + 'static> FragmentValue for $typ
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
                flags: MemFlags,
                address: Value,
                offset: Offset32,
            ) -> Self::IrValues {
                emit_pointer_load::<$typ_var>(builder, flags, address, offset)
            }

            fn emit_store(
                builder: &mut FunctionBuilder,
                flags: MemFlags,
                address: Value,
                values: Self::IrValues,
                offset: Offset32,
            ) {
                emit_pointer_store::<$typ_var>(builder, flags, address, values, offset)
            }
        }
        )*
    }
}

pointer_value_impls! {
    [T, &'static T]
    [T, *const T]
    [T, *mut T]
}

fn combine_offset(base: Offset32, adjustment: usize) -> Offset32 {
    base.try_add_i64(adjustment.try_into().unwrap()).unwrap()
}

/// Copied from core::ptr::metadata sources
#[repr(C)]
struct PtrComponents<T: ?Sized> {
    data_address: *const (),
    metadata: <T as Pointee>::Metadata,
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
    flags: MemFlags,
    address: Value,
    offset: Offset32,
) -> PointerIrValues<T>
where
    <T as Pointee>::Metadata: FragmentValue,
{
    let address = usize::emit_load(
        builder,
        flags,
        address,
        combine_offset(offset, offset_of!(PtrComponents<T>, data_address)),
    );

    let metadata = <<T as Pointee>::Metadata as FragmentValue>::emit_load(
        builder,
        flags,
        address,
        combine_offset(offset, offset_of!(PtrComponents<T>, metadata)),
    );

    (address, metadata)
}

fn emit_pointer_store<T: ?Sized>(
    builder: &mut FunctionBuilder,
    flags: MemFlags,
    address: Value,
    values: PointerIrValues<T>,
    offset: Offset32,
) where
    <T as Pointee>::Metadata: FragmentValue,
{
    usize::emit_store(
        builder,
        flags,
        address,
        values.0,
        combine_offset(offset, offset_of!(PtrComponents<T>, data_address)),
    );

    <<T as Pointee>::Metadata as FragmentValue>::emit_store(
        builder,
        flags,
        address,
        values.1,
        combine_offset(offset, offset_of!(PtrComponents<T>, metadata)),
    );
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
