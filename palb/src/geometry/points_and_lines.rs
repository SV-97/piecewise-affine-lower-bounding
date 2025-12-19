//! # Define some types for primal / dual lines and points and their dualization

use num_traits::Zero;

use crate::Floating;

/// A point in the plane given by (x,y) coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimalPoint {
    pub coords: (Floating, Floating),
}

impl PrimalPoint {
    #[inline(always)]
    pub fn new(x: Floating, y: Floating) -> Self {
        Self { coords: (x, y) }
    }

    #[inline(always)]
    pub fn x(self) -> Floating {
        self.coords.0
    }

    #[inline(always)]
    pub fn y(self) -> Floating {
        self.coords.1
    }
}

/// A line in dual space given by slope and intercept
///
/// Every point (x₀, y₀) in space is in bijection with the set of lines passing through it.
/// These lines γ(x) = (x, mx+t) all satisfy y₀ = mx₀ + t and hence t = -x₀m + y₀.
/// This second equation shows that the lines through the point correspond to a line
/// in the dual space (parametrized by m and t). This struct represents such a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DualLine {
    pub coords: (Floating, Floating),
}

/// A point in dual space (dual to a line in primal space)
// derived Ord instance implements lexicographic order which works for us.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DualPoint {
    pub coords: (Floating, Floating),
}

/// A line in primal (ordinary) space given by slope and intercept
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrimalLine {
    pub coords: (Floating, Floating),
}

/// A dualization relationship
pub trait Dual {
    /// Dualizing twice should get you back where you started
    type DualType: Dual<DualType = Self>;
    fn dual(self) -> Self::DualType;
}

impl Dual for PrimalPoint {
    type DualType = DualLine;
    #[inline(always)]
    fn dual(self) -> Self::DualType {
        DualLine {
            coords: (-self.coords.0, self.coords.1),
        }
    }
}

impl Dual for DualLine {
    type DualType = PrimalPoint;
    #[inline(always)]
    fn dual(self) -> Self::DualType {
        PrimalPoint {
            coords: (-self.coords.0, self.coords.1),
        }
    }
}

impl Dual for PrimalLine {
    type DualType = DualPoint;
    #[inline(always)]
    fn dual(self) -> Self::DualType {
        // Note that there's no inversion here: Primal Lines *are* Dual Points
        DualPoint {
            coords: (self.coords.0, self.coords.1),
        }
    }
}

impl Dual for DualPoint {
    type DualType = PrimalLine;
    #[inline(always)]
    fn dual(self) -> Self::DualType {
        PrimalLine {
            coords: (self.coords.0, self.coords.1),
        }
    }
}

impl DualLine {
    #[inline(always)]
    pub fn slope(self) -> Floating {
        self.coords.0
    }

    #[inline(always)]
    pub fn intercept(self) -> Floating {
        self.coords.1
    }

    /// Determine the intercept for a given slope on this dual line.
    ///
    /// If self has slope -x and intercept y this returns the value of the
    /// function t(m) = -xm + y at the given slope m.
    #[inline(always)]
    pub fn eval_at(self, at: Floating) -> Floating {
        self.slope() * at + self.intercept()
    }

    #[inline]
    pub fn try_intersect(self, other: Self) -> Option<DualPoint> {
        // order inputs by slope to normalize output: intersect(self, other) should equal intersect(other, self)
        // to guarantee this we flip the points around
        if self.coords.0 > other.coords.0 {
            other.try_intersect(self)
        } else {
            // TODO: maybe rewrite without combinators or check up front: benchmark different versions.
            let naive_slope =
                (self.intercept() - other.intercept()) / (other.slope() - self.slope());
            let slope = Some(naive_slope).and_then(
                #[inline]
                |slope| {
                    // It's possible that we divided by 0 above, instead of checking that up front we
                    // just throw out NaNs and Infs here.
                    // There's three cases:
                    if slope.is_nan() {
                        // if we divide 0/0 we get a NaN. This corresponds to both lines having equal
                        // slope and intercept i.e. the coincide. Such lines in particular intersect at 0.
                        Some(Floating::zero())
                    } else if slope.is_finite() {
                        // if we get a finite value everything is fine
                        Some(slope)
                    } else {
                        // if we divide x/0 with x≠0 we get Inf or -Inf. This corresponds to both lines having equal
                        // slope but distinct intercept. Such lines don't intersect.
                        None
                    }
                },
            )?;
            let intercept = self.eval_at(slope);
            Some(DualPoint {
                coords: (slope, intercept),
            })
        }
    }

    #[inline]
    pub fn intersect_unchecked(self, other: Self) -> DualPoint {
        // order inputs by slope to normalize output: intersect(self, other) should equal intersect(other, self)
        // to guarantee this we flip the points around.
        // I don't think we need this since any pair of lines should only have its intersection calculated exactly once?
        // if self.coords.0 > other.coords.0 {
        //     other.intersect_unchecked(self)
        // } else {
        let naive_slope = (self.intercept() - other.intercept()) / (other.slope() - self.slope());
        let intercept = self.eval_at(naive_slope);
        DualPoint {
            coords: (naive_slope, intercept),
        }
        // }
    }

    /// Prefer using [intersect_unchecked] even if you only need the slope: if you don't need the
    /// intercept it'll get optimized out.
    #[inline]
    pub fn slope_of_intersect_unchecked(self, other: Self) -> Floating {
        // order inputs by slope to normalize output: intersect(self, other) should equal intersect(other, self)
        // to guarantee this we flip the points around.
        // I don't think we need this since any pair of lines should only have its intersection calculated exactly once?
        if self.coords.0 > other.coords.0 {
            other.slope_of_intersect_unchecked(self)
        } else {
            (self.intercept() - other.intercept()) / (other.slope() - self.slope())
        }
    }

    /*
    /// A (hopefully) numerically somewhat stable way to compute the point of intersection
    #[inline]
    pub fn slope_of_intersect_unchecked_stable(self, other: Self) -> Floating {
        // order inputs by slope to normalize output: intersect(self, other) should equal intersect(other, self)
        // to guarantee this we flip the points around.
        // I don't think we need this since any pair of lines should only have its intersection calculated exactly once?
        if self.coords.0 > other.coords.0 {
            other.slope_of_intersect_unchecked(self)
        } else {
            // We solve the linear system
            //   (1    -m1)  (y)    = (t1)
            //   (1    -m2)  (x)      (t2)
            // for x using a givens rotation followed by division.
            let elim: GivensRotation = GivensRotation {
                cos: Floating::from(std::f32::consts::FRAC_1_SQRT_2),
                sin: -Floating::from(std::f32::consts::FRAC_1_SQRT_2),
            };
            let (_new_a12, new_a22) = elim.apply_to_pair(-self.slope(), -other.slope());
            let (_new_rhs1, new_rhs2) = elim.apply_to_pair(self.intercept(), other.intercept());
            new_rhs2 / new_a22
        }
    }
    */
}

impl PrimalLine {
    #[inline(always)]
    pub fn slope(self) -> Floating {
        self.coords.0
    }

    #[inline(always)]
    pub fn intercept(self) -> Floating {
        self.coords.1
    }

    /// Determine the y-coordinate for a point (x,y) on this line given x.
    ///
    /// If self has slope m and intercept t this returns the value of the
    /// function y(x) = mx + t at the given value x.
    #[inline(always)]
    pub fn eval_at(self, at: Floating) -> Floating {
        self.slope() * at + self.intercept()
    }
}

/// We tried doing a simple 2x2 solve using Givens rotations to compute the intersection of two lines.
/// This turned out to be quite slow so we switched to our current implementation instead.
/// This is currently unused.
#[derive(Copy, Clone, Debug)]
struct GivensRotation {
    pub cos: Floating,
    pub sin: Floating,
}

impl GivensRotation {
    #[allow(unused)]
    #[inline]
    pub fn to_eliminate(killer: Floating, victim: Floating) -> Self {
        // TODO: could look into continuity optimized implementations like the one described at:
        // https://en.wikipedia.org/wiki/Givens_rotation
        // https://www.netlib.org/lapack/lawnspdf/lawn150.pdf
        let r = killer.hypot(*victim);
        Self {
            cos: killer / r,
            sin: -victim / r,
        }
    }

    // Apply the Givens rotation to a pair of values, returning the rotated values.
    #[inline]
    pub fn apply_to_pair(&self, x: Floating, y: Floating) -> (Floating, Floating) {
        let new_x = self.cos * x - self.sin * y;
        let new_y = self.sin * x + self.cos * y;
        (new_x, new_y)
    }
}

#[allow(unused)]
fn solve_2_by_2(arr: [[Floating; 2]; 2], rhs: [Floating; 2]) -> [Floating; 2] {
    // Extract the elements of the matrix and RHS
    let [[a11, a12], [a21, a22]] = arr;
    let [b1, b2] = rhs;

    // Step 1: Create a Givens rotation to eliminate the a21 element
    let rotation = GivensRotation::to_eliminate(a11, a21);

    // Step 2: Apply the Givens rotation to the first column of the matrix and the RHS
    let (new_a11, _) = rotation.apply_to_pair(a11, a21);
    let (new_a12, new_a22) = rotation.apply_to_pair(a12, a22);
    let (new_b1, new_b2) = rotation.apply_to_pair(b1, b2);

    // Step 3: Solve the upper triangular system
    // At this point:
    // new_a11 * x + new_a12 * y = new_b1
    //              new_a22 * y = new_b2
    let y = new_b2 / new_a22;
    let x = (new_b1 - new_a12 * y) / new_a11;

    // Return the solution
    [x, y]
}
