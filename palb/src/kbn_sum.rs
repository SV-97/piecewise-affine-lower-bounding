//! Implements the Kahan-Babuška-Neumaier copensated summation algorithm,
//! cf. https://en.wikipedia.org/wiki/Kahan_summation_algorithm.
use crate::Floating;
use num_traits::Zero;

/// Implements the Kahan-Babuška-Neumaier summation algorithm, cf. https://en.wikipedia.org/wiki/Kahan_summation_algorithm.
pub fn kahan_babushka_neumaier_sum<I>(iter: I) -> Floating
where
    I: Iterator<Item = Floating>,
{
    let mut sum = Floating::zero();
    let mut c = Floating::zero();

    for x in iter {
        let t = sum + x;
        c += if sum.abs() >= x.abs() {
            (sum - t) + x
        } else {
            (x - t) + sum
        };
        sum = t;
    }
    sum + c
}

/// An extension trait for iterators to add Kahan-Babuška-Neumaier summation.
pub trait KbnSumIteratorExt: Iterator {
    /// Consumes the iterator and computes the sum of its elements using the
    /// Kahan-Babuška-Neumaier algorithm for high-precision summation.
    ///
    /// This method is only available for iterators that yield `Floating` items.
    fn kbn_sum(self) -> Self::Item;
}

/// Implements the `KbnSumIteratorExt` trait for any iterator whose items
/// are of the type `Floating`.
impl<I> KbnSumIteratorExt for I
where
    I: Iterator<Item = Floating>,
{
    /// The core implementation of the KBN summation algorithm.
    fn kbn_sum(self) -> Self::Item {
        kahan_babushka_neumaier_sum(self)
    }
}
