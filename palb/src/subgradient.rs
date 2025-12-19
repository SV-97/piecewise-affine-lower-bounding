/// This is adapted from the rust standard library at https://doc.rust-lang.org/src/core/iter/traits/iterator.rs.html#2211-2214
/// to avoid relying on nightly.
/// Consequently this function is licensed according to MIT / Apache 2.0
///
/// See tracking issue [62543] to avoid this reimplementation in the future.
fn partition_in_place<'a, S, T: 'a, P>(
    some_iter: &mut S,
    #[allow(clippy::toplevel_ref_arg)] ref mut predicate: P,
) -> usize
where
    S: Sized + DoubleEndedIterator<Item = &'a mut T>,
    P: FnMut(&T) -> bool,
{
    // FIXME: should we worry about the count overflowing? The only way to have more than
    // `usize::MAX` mutable references is with ZSTs, which aren't useful to partition...

    // These closure "factory" functions exist to avoid genericity in `Self`.

    #[inline]
    fn is_false<'a, T>(
        predicate: &'a mut impl FnMut(&T) -> bool,
        true_count: &'a mut usize,
    ) -> impl FnMut(&&mut T) -> bool + 'a {
        move |x| {
            let p = predicate(&**x);
            *true_count += p as usize;
            !p
        }
    }

    #[inline]
    fn is_true<T>(predicate: &mut impl FnMut(&T) -> bool) -> impl FnMut(&&mut T) -> bool + '_ {
        move |x| predicate(&**x)
    }

    // Repeatedly find the first `false` and swap it with the last `true`.
    let mut true_count = 0;
    while let Some(head) = some_iter.find(is_false(predicate, &mut true_count)) {
        if let Some(tail) = some_iter.rfind(is_true(predicate)) {
            std::mem::swap(head, tail);
            true_count += 1;
        } else {
            break;
        }
    }
    true_count
}

/// Partitions a slice in place such that all elements for which the predicate is true precede those where
/// it's false. The left slice contains the "true" values, the right one the "false" ones.
pub fn partition_slice<T, P>(slice: &mut [T], predicate: P) -> (&mut [T], &mut [T])
where
    P: FnMut(&T) -> bool,
{
    let n_part_1 = partition_in_place(&mut slice.iter_mut(), predicate);
    slice.split_at_mut(n_part_1)
}

/// Three-way partition using Dutch National Flag algorithm
/// Partitions slice into [< pivot, == pivot, > pivot] in a single O(n) pass
/// Returns (less_than, equal_to, greater_than) slices
// FYI: this turned out to be slower than two back-to-back two-way partitions in practice.
#[allow(dead_code)]
pub fn three_way_partition<T, F>(slice: &mut [T], mut classify: F) -> (&mut [T], &mut [T], &mut [T])
where
    F: FnMut(&T) -> std::cmp::Ordering,
{
    if slice.is_empty() {
        return (&mut [], &mut [], &mut []);
    }

    let mut low = 0; // End of "less than" section
    let mut high = slice.len(); // Start of "greater than" section
    let mut i = 0; // Current position

    while i < high {
        match classify(&slice[i]) {
            std::cmp::Ordering::Less => {
                // Move to "less than" section
                slice.swap(low, i);
                low += 1;
                i += 1;
            }
            std::cmp::Ordering::Equal => {
                // Keep in "equal" section, just advance
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                // Move to "greater than" section
                high -= 1;
                slice.swap(i, high);
                // Don't increment i - we need to classify the swapped element
            }
        }
    }

    // Split the slice into the three sections
    let (left_and_middle, right) = slice.split_at_mut(high);
    let (left, middle) = left_and_middle.split_at_mut(low);

    (left, middle, right)
}
