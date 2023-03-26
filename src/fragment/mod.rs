mod foreign_impls;
mod ops;

use cranelift_codegen::ir::{immediates::Offset32, types, Block, MemFlags, Type, Value};
use cranelift_frontend::FunctionBuilder;

pub unsafe trait Fragment<Input: FragmentValue> {
    type Output: FragmentValue;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: Input::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues;
}

pub unsafe trait FragmentValue: Copy + 'static {
    type IrValues: Clone;

    fn append_block_params(builder: &mut FunctionBuilder, block: Block) -> Self::IrValues;

    fn unpack_ir_values(values: Self::IrValues, dst: &mut impl Extend<Value>);

    fn emit_load(
        builder: &mut FunctionBuilder,
        flags: MemFlags,
        address: Value,
        offset: Offset32,
    ) -> Self::IrValues;

    fn emit_store(
        builder: &mut FunctionBuilder,
        flags: MemFlags,
        address: Value,
        values: Self::IrValues,
        offset: Offset32,
    );
}

pub unsafe trait PrimitiveValue: Fragment<(), Output = Self> + FragmentValue<IrValues = Value> {
    fn ir_type() -> Type;
}

#[derive(Clone, Copy, Debug)]
pub struct CommonInput<A, B>(A, B);

unsafe impl<A: Fragment<Input>, B: Fragment<Input>, Input: FragmentValue> Fragment<Input>
    for CommonInput<A, B>
{
    type Output = (A::Output, B::Output);

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: Input::IrValues,
    ) -> (
        <A::Output as FragmentValue>::IrValues,
        <B::Output as FragmentValue>::IrValues,
    ) {
        let a = self.0.emit_ir(builder, inputs.clone());
        let b = self.1.emit_ir(builder, inputs);
        (a, b)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Compose<A, B>(A, B);

unsafe impl<A: Fragment<Input>, B: Fragment<A::Output>, Input: FragmentValue> Fragment<Input>
    for Compose<A, B>
{
    type Output = B::Output;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: Input::IrValues,
    ) -> <B::Output as FragmentValue>::IrValues {
        let a = self.0.emit_ir(builder, inputs);
        self.1.emit_ir(builder, a)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Identity;

unsafe impl<T: FragmentValue> Fragment<T> for Identity {
    type Output = T;

    fn emit_ir(&self, _builder: &mut FunctionBuilder, inputs: T::IrValues) -> T::IrValues {
        inputs
    }
}

#[cfg(target_pointer_width = "32")]
const ADDRESS_TYPE: Type = types::I32;

#[cfg(target_pointer_width = "64")]
const ADDRESS_TYPE: Type = types::I64;
