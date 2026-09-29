use std::{fmt, ops::Neg};

use num_traits::Signed;

/// An interval containing its boundary points, i.e. a set {x : a <= x <= b} for some a,b : T
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct ClosedInterval<T> {
    // Invariant: bounds must be (nonstrictly) ordered in ascending order
    bounds: [T; 2],
}

impl<T> AsRef<[T; 2]> for ClosedInterval<T> {
    #[inline]
    fn as_ref(&self) -> &[T; 2] {
        &self.bounds
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Pos = 1,
    Zero = 0,
    Neg = -1,
}

impl Neg for Sign {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Self::Neg => Self::Pos,
            Self::Pos => Self::Neg,
            Self::Zero => Self::Zero,
        }
    }
}

#[allow(unused)]
pub enum TieBreak {
    MinSlope,
    MaxSlope,
    Any,
}

impl<T> ClosedInterval<T>
where
    T: Signed + std::fmt::Debug,
{
    /// We say that [a,b] has a uniform sign of 0 if it contains 0,
    /// uniform positive sign if all its values are positive,
    /// and uniform negative sign if all its values are negative.
    #[inline]
    pub fn uniform_sign(&self) -> Sign {
        let g_min = &self.bounds[0];
        let g_max = &self.bounds[1];

        // Setting TieBreak to MinSlope or MaxSlope causes palb to determine minimal and maximal slopes --- ish.
        // While this worked fine in our testing, it's clear that there are edge cases that need some additional handling.
        // For instance: if the two starting points both are stationary then we really need to first enlarge / shift the
        // interval to ensure that it contains the max / min slopes.
        // So consider all `tie_break` values except for `Any` to be highly experimental at this point,
        // and more of a proof of concept.
        let tie_break = TieBreak::Any;
        match tie_break {
            TieBreak::Any => {
                // Strict check: stops anywhere on the plateau
                if g_min.is_positive() && !g_min.is_zero() {
                    Sign::Pos
                } else if g_max.is_negative() && !g_max.is_zero() {
                    Sign::Neg
                } else {
                    Sign::Zero
                }
            }
            TieBreak::MinSlope => {
                // Biased left: treats 0 as positive to force decreasing slope.
                // Stops ONLY when g_min is strictly negative (left kink).
                if g_min.is_positive() || g_min.is_zero() {
                    Sign::Pos
                } else if g_max.is_negative() && !g_max.is_zero() {
                    Sign::Neg
                } else {
                    Sign::Zero
                }
            }
            TieBreak::MaxSlope => {
                // Biased right: treats 0 as negative to force increasing slope.
                // Stops ONLY when g_max is strictly positive (right kink).
                if g_min.is_positive() && !g_min.is_zero() {
                    Sign::Pos
                } else if g_max.is_negative() || g_max.is_zero() {
                    Sign::Neg
                } else {
                    Sign::Zero
                }
            }
        }
    }
}

impl<T: fmt::Display> fmt::Display for ClosedInterval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.bounds[0], self.bounds[1])
    }
}

impl<T: Ord> ClosedInterval<T> {
    #[inline]
    pub fn new(mut bounds: [T; 2]) -> Self {
        bounds.sort();
        Self { bounds }
    }

    #[inline]
    pub fn max(&self) -> T
    where
        T: Copy,
    {
        self.bounds[1]
    }

    #[inline]
    pub fn min(&self) -> T
    where
        T: Copy,
    {
        self.bounds[0]
    }
}
