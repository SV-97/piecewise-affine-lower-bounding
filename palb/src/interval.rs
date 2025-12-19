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

impl<T> ClosedInterval<T>
where
    T: Signed,
{
    /// We say that [a,b] has a uniform sign of 0 if it contains 0,
    /// uniform positive sign if all its values are positive,
    /// and uniform negative sign if all its values are negative.
    #[inline]
    pub fn uniform_sign(&self) -> Sign {
        if self.bounds[0].is_positive() {
            Sign::Pos
        } else if self.bounds[1].is_negative() {
            Sign::Neg
        } else {
            Sign::Zero
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
