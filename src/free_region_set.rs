use std::{
    collections::BTreeMap,
    ops::{Add, Bound, Range, Sub},
};

use num::traits::bounds::LowerBounded;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct LenStart<T>(T, T);

#[derive(Clone, Debug)]
pub struct FreeRegionSet<T> {
    by_start: BTreeMap<T, T>,
    by_len: BTreeMap<LenStart<T>, ()>,
}

impl<T> FreeRegionSet<T> {
    pub const fn new() -> Self {
        Self {
            by_start: BTreeMap::new(),
            by_len: BTreeMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.by_start.clear();
        self.by_len.clear();
    }

    pub fn insert(&mut self, mut region: Range<T>)
    where
        T: Copy + Sub<Output = T> + Ord,
    {
        let mut cursor = self
            .by_start
            .upper_bound_mut(Bound::Excluded(&region.start));

        match cursor.key_value() {
            Some((&start_before, &end_before)) if end_before == region.start => {
                region.start = start_before;
                *cursor.value_mut().unwrap() = region.end;
                let len_before = end_before - start_before;
                self.by_len.remove(&LenStart(len_before, start_before));
            }

            _ => {
                cursor.insert_after(region.start, region.end);
                cursor.move_next();
            }
        }

        match cursor.peek_next() {
            Some((&start_after, &mut end_after)) if start_after == region.end => {
                region.end = end_after;
                *cursor.value_mut().unwrap() = region.end;
                let len_after = end_after - start_after;
                self.by_len.remove(&LenStart(len_after, start_after));
                cursor.move_next();
                cursor.remove_current_and_move_back();
            }

            _ => (),
        }

        self.by_len
            .insert(LenStart(region.end - region.start, region.start), ());
    }

    pub fn remove_best_fit(&mut self, len: T) -> Option<Range<T>>
    where
        T: Ord + LowerBounded,
        for<'a> &'a T: Add<T, Output = T>,
    {
        let key = LenStart(len, T::min_value());
        let (LenStart(len, start), ()) = self
            .by_len
            .lower_bound_mut(Bound::Included(&key))
            .remove_current()?;
        let _ = self.by_start.remove(&start);
        let end = &start + len;
        Some(start..end)
    }
}

impl<T> Default for FreeRegionSet<T> {
    fn default() -> Self {
        Self::new()
    }
}
