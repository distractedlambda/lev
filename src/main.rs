#![allow(incomplete_features)]
#![feature(array_methods)]
#![feature(array_zip)]
#![feature(generic_const_exprs)]
#![feature(int_roundings)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]

mod aabb;
mod constant;
mod simd;
mod vector;
mod vectorize;

fn main() {
    println!("Hello, world!");
}
