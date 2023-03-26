use std::mem::{transmute, MaybeUninit};

use cranelift_codegen::{
    ir::{immediates::Offset32, AbiParam, InstBuilder, MemFlags, Value},
    Context,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module, ModuleResult};

use crate::fragment::{Fragment, FragmentValue, SafeFragment, ADDRESS_TYPE};

pub struct FragmentCompiler {
    module: JITModule,
    frontend_context: FunctionBuilderContext,
    backend_context: Context,
}

#[derive(Clone)]
pub struct CompiledFragment<Input, Output, const SAFE: bool>(
    unsafe extern "C" fn(&Input, &mut MaybeUninit<Output>),
);

pub type CompiledUnsafeFragment<Input, Output> = CompiledFragment<Input, Output, false>;

pub type CompiledSafeFragment<Input, Output> = CompiledFragment<Input, Output, true>;

impl FragmentCompiler {
    pub fn new() -> ModuleResult<Self> {
        Ok(Self {
            module: JITModule::new(JITBuilder::with_flags(
                &[("opt_level", "best")],
                default_libcall_names(),
            )?),
            frontend_context: FunctionBuilderContext::new(),
            backend_context: Context::new(),
        })
    }

    pub fn compile_unsafe<F: Fragment<Input>, Input: FragmentValue>(
        &mut self,
        fragment: &F,
    ) -> ModuleResult<CompiledUnsafeFragment<Input, F::Output>> {
        self.compile_with_safety(fragment)
    }

    pub fn compile<F: SafeFragment<Input>, Input: FragmentValue>(
        &mut self,
        fragment: &F,
    ) -> ModuleResult<CompiledSafeFragment<Input, F::Output>> {
        self.compile_with_safety(fragment)
    }

    fn compile_with_safety<F: Fragment<Input>, Input: FragmentValue, const SAFE: bool>(
        &mut self,
        fragment: &F,
    ) -> ModuleResult<CompiledFragment<Input, F::Output, SAFE>> {
        Ok(CompiledFragment(unsafe {
            transmute(self.codegen(fragment)?)
        }))
    }

    fn codegen<F: Fragment<Input>, Input: FragmentValue>(
        &mut self,
        fragment: &F,
    ) -> ModuleResult<*const u8> {
        let (mut builder, input_address, output_address) = self.begin_codegen();

        let inputs = Input::emit_load(
            &mut builder,
            MemFlags::trusted().with_readonly(),
            input_address,
            Offset32::new(0),
        );

        let outputs = fragment.emit_ir(&mut builder, inputs);

        F::Output::emit_store(
            &mut builder,
            MemFlags::trusted(),
            output_address,
            outputs,
            Offset32::new(0),
        );

        builder.ins().return_(&[]);
        builder.finalize();

        self.finish_codegen()
    }

    fn begin_codegen(&mut self) -> (FunctionBuilder, Value, Value) {
        self.module.clear_context(&mut self.backend_context);

        self.backend_context
            .func
            .signature
            .params
            .extend_from_slice(&[AbiParam::new(ADDRESS_TYPE), AbiParam::new(ADDRESS_TYPE)]);

        let mut builder =
            FunctionBuilder::new(&mut self.backend_context.func, &mut self.frontend_context);

        let entry_block = builder.create_block();
        builder.seal_block(entry_block);
        let input_address = builder.append_block_param(entry_block, ADDRESS_TYPE);
        let output_address = builder.append_block_param(entry_block, ADDRESS_TYPE);
        builder.switch_to_block(entry_block);

        (builder, input_address, output_address)
    }

    fn finish_codegen(&mut self) -> ModuleResult<*const u8> {
        let id = self
            .module
            .declare_anonymous_function(&self.backend_context.func.signature)?;
        self.module.define_function(id, &mut self.backend_context)?;
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(id))
    }
}

impl<Input, Output, const SAFE: bool> CompiledFragment<Input, Output, SAFE> {
    unsafe fn call_unsafe(&self, input: &Input) -> Output {
        let mut output = MaybeUninit::uninit();
        (self.0)(input, &mut output);
        output.assume_init()
    }
}

impl<Input, Output> CompiledFragment<Input, Output, true> {
    fn call(&self, input: &Input) -> Output {
        unsafe { self.call_unsafe(input) }
    }
}