use std::mem::MaybeUninit;

use cranelift_codegen::{ir::Signature, Context};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module, ModuleResult};

use crate::fragment::{Fragment, FragmentValue};

pub struct FragmentCompiler {
    jit_builder: JITBuilder,
    frontend_context: FunctionBuilderContext,
    backend_context: Context,
}

pub struct CompiledFragment<Input, Output, const SAFE: bool> {
    module: JITModule,
    entry: unsafe extern "C" fn(&Input, &mut MaybeUninit<Output>),
}

pub type CompiledUnsafeFragment<Input, Output> = CompiledFragment<Input, Output, false>;

pub type CompiledSafeFragment<Input, Output> = CompiledFragment<Input, Output, true>;

impl FragmentCompiler {
    pub fn new() -> ModuleResult<Self> {
        Ok(Self {
            jit_builder: JITBuilder::with_flags(&[("opt_level", "best")], default_libcall_names())?,
            frontend_context: FunctionBuilderContext::new(),
            backend_context: Context::new(),
        })
    }

    // fn populate_signature<Input: FragmentValue, Output: FragmentValue>()

    fn emit_ir<F: Fragment<Input>, Input: FragmentValue>(
        &mut self,
        module: &JITModule,
        fragment: &F,
    ) -> Result<(), ()> {
        module.clear_context(&mut self.backend_context);
        Ok(())
    }
}

impl<Input, Output, const SAFE: bool> CompiledFragment<Input, Output, SAFE> {
    unsafe fn call_unsafe(&self, input: &Input) -> Output {
        let mut output = MaybeUninit::uninit();
        (self.entry)(input, &mut output);
        output.assume_init()
    }
}

impl<Input, Output> CompiledFragment<Input, Output, true> {
    fn call(&self, input: &Input) -> Output {
        unsafe { self.call_unsafe(input) }
    }
}
