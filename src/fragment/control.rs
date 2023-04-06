use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use smallvec::SmallVec;

use crate::fragment::{Fragment, FragmentValue};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct If<C, T, F> {
    condition: C,
    if_true: T,
    if_false: F,
}

impl<C, T, F> Fragment for If<C, T, F>
where
    C: Fragment<Output = bool>,
    T: Fragment<Input = C::Input>,
    F: Fragment<Input = C::Input, Output = T::Output>,
{
    type Input = C::Input;

    type Output = T::Output;

    fn emit_ir(
        &self,
        builder: &mut FunctionBuilder,
        inputs: <Self::Input as FragmentValue>::IrValues,
    ) -> <Self::Output as FragmentValue>::IrValues {
        let true_block = builder.create_block();
        let false_block = builder.create_block();
        let join_block = builder.create_block();

        let join_params = T::Output::append_block_params(builder, join_block);
        let mut unpacked_join_args = SmallVec::<[_; 16]>::new();

        let condition = self.condition.emit_ir(builder, inputs.clone());
        builder
            .ins()
            .brif(condition, true_block, &[], false_block, &[]);

        builder.seal_block(true_block);
        builder.seal_block(false_block);

        builder.switch_to_block(true_block);
        let true_result = self.if_true.emit_ir(builder, inputs.clone());
        T::Output::unpack_ir_values(true_result, &mut unpacked_join_args);
        builder.ins().jump(join_block, &unpacked_join_args);
        unpacked_join_args.clear();

        builder.switch_to_block(false_block);
        let false_result = self.if_false.emit_ir(builder, inputs);
        T::Output::unpack_ir_values(false_result, &mut unpacked_join_args);
        builder.ins().jump(join_block, &unpacked_join_args);

        builder.seal_block(join_block);
        builder.switch_to_block(join_block);

        join_params
    }
}
