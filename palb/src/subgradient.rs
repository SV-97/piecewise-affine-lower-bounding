#[inline]
fn partition_in_place<T, P>(data: &mut [T], mut predicate: P) -> usize
where
    P: FnMut(&T) -> bool,
{
    let mut l = 0;
    let mut r = data.len();

    loop {
        while l < r && predicate(&data[l]) {
            l += 1;
        }

        while l < r && !predicate(&data[r - 1]) {
            r -= 1;
        }

        if l >= r {
            break;
        }

        data.swap(l, r - 1);

        l += 1;
        r -= 1;
    }
    l
}

/// Partitions a slice in place such that all elements for which the predicate is true precede those where
/// it's false. The left slice contains the "true" values, the right one the "false" ones.
pub fn partition_slice<T, P>(slice: &mut [T], predicate: P) -> (&mut [T], &mut [T])
where
    P: FnMut(&T) -> bool,
{
    let n_part_1 = partition_in_place(slice, predicate);
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
