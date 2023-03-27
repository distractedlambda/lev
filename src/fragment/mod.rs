mod foreign_impls;
mod jit;
mod ops;

use cranelift_codegen::ir::{immediates::Offset32, types, Block, MemFlags, Type, Value};
use cranelift_frontend::FunctionBuilder;
pub use jit::*;
pub use ops::*;

pub trait Fragment {
    type Input: FragmentValue;

    type Output: FragmentValue;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues;
}

pub unsafe trait SafeFragment: Fragment {}

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

pub unsafe trait PrimitiveValue:
    SafeFragment<Input = (), Output = Self> + FragmentValue<IrValues = Value>
{
    fn ir_type() -> Type;
}

#[derive(Clone, Copy, Debug)]
pub struct CommonInput<A, B>(A, B);

impl<A: Fragment, B: Fragment<Input = A::Input>> Fragment for CommonInput<A, B> {
    type Input = A::Input;

    type Output = (A::Output, B::Output);

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <A::Input as FragmentValue>::IrValues,
    ) -> (
        <A::Output as FragmentValue>::IrValues,
        <B::Output as FragmentValue>::IrValues,
    ) {
        let a = self.0.emit_ir(builder, inputs.clone());
        let b = self.1.emit_ir(builder, inputs);
        (a, b)
    }
}

unsafe impl<A: SafeFragment, B: SafeFragment<Input = A::Input>> SafeFragment for CommonInput<A, B> {}

#[derive(Clone, Copy, Debug)]
pub struct Compose<A, B>(pub A, pub B);

impl<A: Fragment, B: Fragment<Input = A::Output>> Fragment for Compose<A, B> {
    type Input = A::Input;

    type Output = B::Output;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <A::Input as FragmentValue>::IrValues,
    ) -> <B::Output as FragmentValue>::IrValues {
        let a = self.0.emit_ir(builder, inputs);
        self.1.emit_ir(builder, a)
    }
}

unsafe impl<A: SafeFragment, B: SafeFragment<Input = A::Output>> SafeFragment for Compose<A, B> {}

#[cfg(target_pointer_width = "32")]
const ADDRESS_TYPE: Type = types::I32;

#[cfg(target_pointer_width = "64")]
const ADDRESS_TYPE: Type = types::I64;
