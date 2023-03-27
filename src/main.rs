#![feature(array_methods)]
#![feature(array_zip)]
#![feature(extend_one)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(portable_simd)]
#![feature(ptr_metadata)]

use crate::fragment::{add, Compose, FragmentCompiler, mul, neg};

mod fragment;

fn main() -> anyhow::Result<()> {
    let mut compiler = FragmentCompiler::new()?;
    let f = compiler.compile(&Compose(mul::<isize>(), neg::<isize>()))?;
    println!("-(42 * 2) = {}", f.call(&(42, 2)));
    Ok(())
}
