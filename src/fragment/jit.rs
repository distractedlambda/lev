use std::{mem::{transmute, MaybeUninit}, ops::Deref, ptr::{null_mut, NonNull}, slice};
use std::ops::DerefMut;

use anyhow::anyhow;
use cranelift_codegen::{
    ir::{immediates::Offset32, AbiParam, InstBuilder, MemFlags, Value},
    settings::Configurable,
    Context,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Module};
use libc::{
    mmap, pthread_jit_write_protect_np, vm_page_size, MAP_JIT, MAP_PRIVATE, PROT_EXEC, PROT_READ,
    PROT_WRITE,
};
use parking_lot::Mutex;

use crate::{
    fragment::{Fragment, FragmentValue, SafeFragment, ADDRESS_TYPE},
    free_region_set::FreeRegionSet,
};

static FREE_JIT_MEMORY: Mutex<FreeRegionSet<usize>> = Mutex::new(FreeRegionSet::new());

#[derive(Debug)]
struct JitMemory {
    base: NonNull<u8>,
    len: usize,
}

impl JitMemory {
    pub fn alloc(len: usize) -> Self {
        let (base, alloc_len) = 'blk: {
            if let Some(region) = FREE_JIT_MEMORY.lock().remove_best_fit(len) {
                let base = NonNull::new(region.start as *mut u8).unwrap();
                let alloc_len = region.end - region.start;
                break 'blk (base, alloc_len);
            }

            #[cfg(target_os = "macos")]
            unsafe {
                let min_pages = len.div_ceil(vm_page_size);
                let alloc_len = min_pages * vm_page_size;
                let base = mmap(
                    null_mut(),
                    alloc_len,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_JIT,
                    -1,
                    0,
                );
                let base = NonNull::new(base as *mut u8).unwrap();
                break 'blk (base, alloc_len);
            }

            #[cfg(not(target_os = "macos"))]
            todo!()
        };

        if len != alloc_len {
            let base_addr = base.as_ptr() as usize;
            FREE_JIT_MEMORY
                .lock()
                .insert(base_addr + len..base_addr + alloc_len);
        }

        Self { base, len }
    }

    pub unsafe fn mutate(&mut self) -> MutableJitMemory {
        #[cfg(target_os = "macos")]
        pthread_jit_write_protect_np(0);

        MutableJitMemory(self)
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        let base = self.base.as_ptr() as usize;
        let region = base..base + self.len;
        FREE_JIT_MEMORY.lock().insert(region);
    }
}

struct MutableJitMemory<'a>(&'a mut JitMemory);

impl<'a> Deref for MutableJitMemory<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe {
            slice::from_raw_parts(self.0.base.as_ptr(), self.0.len)
        }
    }
}

impl<'a> DerefMut for MutableJitMemory<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(self.0.base.as_ptr())
        }
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn sys_icache_invalidate(start: *mut u8, len: usize);
}

impl<'a> Drop for MutableJitMemory<'a> {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(self.0.base.as_ptr(), self.0.len);
        }

        #[cfg(not(target_os = "macos"))]
        todo!()
    }
}

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
    pub fn new() -> anyhow::Result<Self> {
        let mut flags = cranelift_codegen::settings::builder();

        flags.set("use_colocated_libcalls", "false")?;

        Ok(Self {
            module: JITModule::new(JITBuilder::with_isa(
                cranelift_native::builder()
                    .map_err(|msg| anyhow!("{}", msg))?
                    .finish(cranelift_codegen::settings::Flags::new(flags))?,
                default_libcall_names(),
            )),
            frontend_context: FunctionBuilderContext::new(),
            backend_context: Context::new(),
        })
    }

    pub fn compile_unsafe<F: Fragment>(
        &mut self,
        fragment: &F,
    ) -> anyhow::Result<CompiledUnsafeFragment<F::Input, F::Output>> {
        self.compile_with_safety(fragment)
    }

    pub fn compile<F: SafeFragment>(
        &mut self,
        fragment: &F,
    ) -> anyhow::Result<CompiledSafeFragment<F::Input, F::Output>> {
        self.compile_with_safety(fragment)
    }

    fn compile_with_safety<F: Fragment, const SAFE: bool>(
        &mut self,
        fragment: &F,
    ) -> anyhow::Result<CompiledFragment<F::Input, F::Output, SAFE>> {
        Ok(CompiledFragment(unsafe {
            transmute(self.codegen(fragment)?)
        }))
    }

    fn codegen<F: Fragment>(&mut self, fragment: &F) -> anyhow::Result<*const u8> {
        let (mut builder, input_address, output_address) = self.begin_codegen();

        let inputs = F::Input::emit_load(
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

    fn finish_codegen(&mut self) -> anyhow::Result<*const u8> {
        let id = self
            .module
            .declare_anonymous_function(&self.backend_context.func.signature)?;
        self.module.define_function(id, &mut self.backend_context)?;
        self.module.finalize_definitions()?;
        Ok(self.module.get_finalized_function(id))
    }
}

impl<Input, Output, const SAFE: bool> CompiledFragment<Input, Output, SAFE> {
    pub unsafe fn call_unsafe(&self, input: &Input) -> Output {
        let mut output = MaybeUninit::uninit();
        (self.0)(input, &mut output);
        output.assume_init()
    }
}

impl<Input, Output> CompiledFragment<Input, Output, true> {
    pub fn call(&self, input: &Input) -> Output {
        unsafe { self.call_unsafe(input) }
    }
}
