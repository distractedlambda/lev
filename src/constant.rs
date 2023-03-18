use sealed::sealed;

#[derive(Clone, Copy, Debug)]
pub struct BoolConstant<const VALUE: bool>;

#[derive(Clone, Copy, Debug)]
pub struct U8Constant<const VALUE: u8>;

#[derive(Clone, Copy, Debug)]
pub struct U16Constant<const VALUE: u16>;

#[derive(Clone, Copy, Debug)]
pub struct U32Constant<const VALUE: u32>;

#[derive(Clone, Copy, Debug)]
pub struct U64Constant<const VALUE: u64>;

#[derive(Clone, Copy, Debug)]
pub struct UsizeConstant<const VALUE: usize>;

#[derive(Clone, Copy, Debug)]
pub struct I8Constant<const VALUE: i8>;

#[derive(Clone, Copy, Debug)]
pub struct I16Constant<const VALUE: i16>;

#[derive(Clone, Copy, Debug)]
pub struct I32Constant<const VALUE: i32>;

#[derive(Clone, Copy, Debug)]
pub struct I64Constant<const VALUE: i64>;

#[derive(Clone, Copy, Debug)]
pub struct IsizeConstant<const VALUE: isize>;

#[sealed]
pub trait ConstantTrue {}

#[sealed]
pub trait ConstantFalse {}

#[sealed]
impl ConstantTrue for BoolConstant<true> {}

#[sealed]
impl ConstantFalse for BoolConstant<false> {}
