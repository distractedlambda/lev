use std::{
    borrow::{Borrow, BorrowMut},
    iter::{Product, Sum},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr,
        ShrAssign, Sub, SubAssign,
    },
    simd::{
        LaneCount, Simd, SimdElement, SimdFloat, SimdOrd, SimdPartialEq, SimdPartialOrd,
        SupportedLaneCount,
    },
    vec::IntoIter,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Vector<T, const LEN: usize>([T; LEN]);

impl<T, const LEN: usize> Vector<T, LEN> {
    #[inline]
    pub fn each_ref(&self) -> Vector<&T, LEN> {
        Vector(self.0.each_ref())
    }

    #[inline]
    pub fn each_mut(&mut self) -> Vector<&mut T, LEN> {
        Vector(self.0.each_mut())
    }

    #[inline]
    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Vector<U, LEN> {
        Vector(self.0.map(f))
    }

    #[inline]
    pub fn map2<U, V>(self, other: Vector<U, LEN>, mut f: impl FnMut(T, U) -> V) -> Vector<V, LEN> {
        self.zip(other).map(|(a, b)| f(a, b))
    }

    #[inline]
    pub fn map3<U, V, W>(
        self,
        other1: Vector<U, LEN>,
        other2: Vector<V, LEN>,
        mut f: impl FnMut(T, U, V) -> W,
    ) -> Vector<W, LEN> {
        self.map2(other1.zip(other2), |a, (b, c)| f(a, b, c))
    }

    #[inline]
    pub fn zip<U>(self, other: Vector<U, LEN>) -> Vector<(T, U), LEN> {
        Vector(self.0.zip(other.0))
    }

    #[inline]
    pub fn reduce(self, f: impl FnMut(T, T) -> T) -> T {
        self.into_iter().reduce(f).unwrap()
    }

    #[inline]
    pub fn each_from<U>(other: Vector<U, LEN>) -> Self
    where
        T: From<U>,
    {
        other.map(T::from)
    }

    #[inline]
    pub fn each_into<U>(self) -> Vector<U, LEN>
    where
        T: Into<U>,
    {
        self.map(T::into)
    }

    #[inline]
    pub fn iter(&self) -> <&[T; LEN] as IntoIterator>::IntoIter {
        self.0.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> <&mut [T; LEN] as IntoIterator>::IntoIter {
        self.0.iter_mut()
    }

    #[inline]
    pub fn sum<U: Sum<T>>(self) -> U {
        self.into_iter().sum()
    }

    #[inline]
    pub fn product<U: Product<T>>(self) -> U {
        self.into_iter().product()
    }

    #[inline]
    pub fn dot<U, V: Sum<<T as Mul<U>>::Output>>(self, other: Vector<U, LEN>) -> V
    where
        T: Mul<U>,
    {
        (self * other).sum()
    }

    #[inline]
    pub fn norm_squared<U: Sum<<T as Mul>::Output>>(self) -> U
    where
        T: Mul + Clone,
    {
        self.clone().dot(self)
    }

    #[inline]
    pub fn each_eq<U>(&self, other: &Vector<U, LEN>) -> Vector<bool, LEN>
    where
        T: PartialEq<U>,
    {
        self.each_ref().map2(other.each_ref(), T::eq)
    }

    #[inline]
    pub fn each_ne<U>(&self, other: &Vector<U, LEN>) -> Vector<bool, LEN>
    where
        T: PartialEq<U>,
    {
        self.each_ref().map2(other.each_ref(), T::ne)
    }

    #[inline]
    pub fn each_lt<U>(&self, other: &Vector<U, LEN>) -> Vector<bool, LEN>
    where
        T: PartialOrd<U>,
    {
        self.each_ref().map2(other.each_ref(), T::lt)
    }

    #[inline]
    pub fn each_le<U>(&self, other: &Vector<U, LEN>) -> Vector<bool, LEN>
    where
        T: PartialOrd<U>,
    {
        self.each_ref().map2(other.each_ref(), T::le)
    }

    #[inline]
    pub fn each_gt<U>(&self, other: &Vector<U, LEN>) -> Vector<bool, LEN>
    where
        T: PartialOrd<U>,
    {
        self.each_ref().map2(other.each_ref(), T::gt)
    }

    #[inline]
    pub fn each_ge<U>(&self, other: &Vector<U, LEN>) -> Vector<bool, LEN>
    where
        T: PartialOrd<U>,
    {
        self.each_ref().map2(other.each_ref(), T::ge)
    }
}

impl<T: Ord, const LEN: usize> Vector<T, LEN> {
    #[inline]
    pub fn each_min(self, other: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map2(other, T::min)
    }

    #[inline]
    pub fn each_max(self, other: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map2(other, T::max)
    }

    #[inline]
    pub fn each_clamp(self, min: Vector<T, LEN>, max: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map3(min, max, T::clamp)
    }
}

impl<T: SimdPartialEq, const LEN: usize> Vector<T, LEN> {
    #[inline]
    pub fn each_simd_eq(self, other: Vector<T, LEN>) -> Vector<T::Mask, LEN> {
        self.map2(other, T::simd_eq)
    }

    #[inline]
    pub fn each_simd_ne(self, other: Vector<T, LEN>) -> Vector<T::Mask, LEN> {
        self.map2(other, T::simd_ne)
    }
}

impl<T: SimdPartialOrd, const LEN: usize> Vector<T, LEN> {
    #[inline]
    pub fn each_simd_lt(self, other: Vector<T, LEN>) -> Vector<T::Mask, LEN> {
        self.map2(other, T::simd_lt)
    }

    #[inline]
    pub fn each_simd_le(self, other: Vector<T, LEN>) -> Vector<T::Mask, LEN> {
        self.map2(other, T::simd_le)
    }

    #[inline]
    pub fn each_simd_gt(self, other: Vector<T, LEN>) -> Vector<T::Mask, LEN> {
        self.map2(other, T::simd_gt)
    }

    #[inline]
    pub fn each_simd_ge(self, other: Vector<T, LEN>) -> Vector<T::Mask, LEN> {
        self.map2(other, T::simd_ge)
    }
}

impl<T: SimdOrd, const LEN: usize> Vector<T, LEN> {
    #[inline]
    pub fn each_simd_min(self, other: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map2(other, T::simd_min)
    }

    #[inline]
    pub fn each_simd_max(self, other: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map2(other, T::simd_max)
    }

    #[inline]
    pub fn each_simd_clamp(self, min: Vector<T, LEN>, max: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map3(min, max, T::simd_clamp)
    }
}

impl<T: SimdFloat, const LEN: usize> Vector<T, LEN> {
    #[inline]
    pub fn each_simd_to_bits(self) -> Vector<T::Bits, LEN> {
        self.map(T::to_bits)
    }

    #[inline]
    pub fn each_simd_fmin(self, other: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map2(other, T::simd_min)
    }

    #[inline]
    pub fn each_simd_fmax(self, other: Vector<T, LEN>) -> Vector<T, LEN> {
        self.map2(other, T::simd_max)
    }
}

impl<T: SimdPartialEq, const LEN: usize> SimdPartialEq for Vector<T, LEN>
where
    T::Mask: BitAnd<Output = T::Mask>,
{
    type Mask = T::Mask;

    #[inline]
    fn simd_eq(self, other: Self) -> Self::Mask {
        self.map2(other, T::simd_eq).reduce(BitAnd::bitand)
    }

    #[inline]
    fn simd_ne(self, other: Self) -> Self::Mask {
        self.map2(other, T::simd_ne).reduce(BitAnd::bitand)
    }
}

impl<T: Clone, const LEN: usize> From<T> for Vector<T, LEN> {
    #[inline]
    fn from(value: T) -> Self {
        Vector([(); LEN].map(|_| value.clone()))
    }
}

impl<T, const LEN: usize> From<[T; LEN]> for Vector<T, LEN> {
    #[inline]
    fn from(value: [T; LEN]) -> Self {
        Self(value)
    }
}

impl<T, const LEN: usize> From<Vector<T, LEN>> for [T; LEN] {
    #[inline]
    fn from(value: Vector<T, LEN>) -> Self {
        value.0
    }
}

impl<T, const LEN: usize> AsRef<[T; LEN]> for Vector<T, LEN> {
    #[inline]
    fn as_ref(&self) -> &[T; LEN] {
        &self.0
    }
}

impl<T, const LEN: usize> AsMut<[T; LEN]> for Vector<T, LEN> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; LEN] {
        &mut self.0
    }
}

impl<T, const LEN: usize> AsRef<[T]> for Vector<T, LEN> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T, const LEN: usize> AsMut<[T]> for Vector<T, LEN> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T, const LEN: usize> Borrow<[T; LEN]> for Vector<T, LEN> {
    #[inline]
    fn borrow(&self) -> &[T; LEN] {
        &self.0
    }
}

impl<T, const LEN: usize> BorrowMut<[T; LEN]> for Vector<T, LEN> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T; LEN] {
        &mut self.0
    }
}

impl<T, const LEN: usize> Borrow<[T]> for Vector<T, LEN> {
    #[inline]
    fn borrow(&self) -> &[T] {
        &self.0
    }
}

impl<T, const LEN: usize> BorrowMut<[T]> for Vector<T, LEN> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T, const LEN: usize> IntoIterator for Vector<T, LEN> {
    type Item = T;

    type IntoIter = <[T; LEN] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, const LEN: usize> IntoIterator for &'a Vector<T, LEN> {
    type Item = &'a T;

    type IntoIter = <&'a [T; LEN] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T, const LEN: usize> IntoIterator for &'a mut Vector<T, LEN> {
    type Item = &'a mut T;

    type IntoIter = <&'a mut [T; LEN] as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T, I, const LEN: usize> Index<I> for Vector<T, LEN>
where
    [T; LEN]: Index<I>,
{
    type Output = <[T; LEN] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I, const LEN: usize> IndexMut<I> for Vector<T, LEN>
where
    [T; LEN]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Not, const LEN: usize> Not for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn not(self) -> Self::Output {
        Vector(self.0.map(T::not))
    }
}

impl<T: Neg, const LEN: usize> Neg for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector(self.0.map(T::neg))
    }
}

impl<T: Add<U>, U, const LEN: usize> Add<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn add(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l + r))
    }
}

impl<T: Sub<U>, U, const LEN: usize> Sub<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn sub(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l - r))
    }
}

impl<T: Mul<U>, U, const LEN: usize> Mul<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn mul(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l * r))
    }
}

impl<T: Div<U>, U, const LEN: usize> Div<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn div(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l / r))
    }
}

impl<T: Rem<U>, U, const LEN: usize> Rem<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn rem(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l % r))
    }
}

impl<T: BitAnd<U>, U, const LEN: usize> BitAnd<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn bitand(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l & r))
    }
}

impl<T: BitOr<U>, U, const LEN: usize> BitOr<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn bitor(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l | r))
    }
}

impl<T: BitXor<U>, U, const LEN: usize> BitXor<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn bitxor(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l ^ r))
    }
}

impl<T: Shl<U>, U, const LEN: usize> Shl<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn shl(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l << r))
    }
}

impl<T: Shr<U>, U, const LEN: usize> Shr<Vector<U, LEN>> for Vector<T, LEN> {
    type Output = Vector<T::Output, LEN>;

    #[inline]
    fn shr(self, rhs: Vector<U, LEN>) -> Self::Output {
        Vector(self.0.zip(rhs.0).map(|(l, r)| l >> r))
    }
}

impl<T: AddAssign<U>, U, const LEN: usize> AddAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn add_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l += r
        }
    }
}

impl<T: SubAssign<U>, U, const LEN: usize> SubAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l -= r
        }
    }
}

impl<T: MulAssign<U>, U, const LEN: usize> MulAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn mul_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l *= r
        }
    }
}

impl<T: DivAssign<U>, U, const LEN: usize> DivAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn div_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l /= r
        }
    }
}

impl<T: RemAssign<U>, U, const LEN: usize> RemAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn rem_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l %= r
        }
    }
}

impl<T: BitAndAssign<U>, U, const LEN: usize> BitAndAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn bitand_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l &= r
        }
    }
}

impl<T: BitOrAssign<U>, U, const LEN: usize> BitOrAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn bitor_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l |= r
        }
    }
}

impl<T: BitXorAssign<U>, U, const LEN: usize> BitXorAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l ^= r
        }
    }
}

impl<T: ShlAssign<U>, U, const LEN: usize> ShlAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn shl_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l <<= r
        }
    }
}

impl<T: ShrAssign<U>, U, const LEN: usize> ShrAssign<Vector<U, LEN>> for Vector<T, LEN> {
    #[inline]
    fn shr_assign(&mut self, rhs: Vector<U, LEN>) {
        for (l, r) in self.0.each_mut().zip(rhs.0) {
            *l >>= r
        }
    }
}
