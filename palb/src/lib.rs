//! PALB is an exact, robust, high-performance solver for the Least-Absolute-Deviations-Line (LAD) problem, i.e. one dimensional affine linear L1 regression.
//! This is the Rust core; be aware that there is also a Python API (`palb_py`).
use std::borrow::Cow;

pub use geometry::{Dual, DualLine, PrimalLine, PrimalPoint};
use interval::{ClosedInterval, Sign};
use num_traits::{One, Signed, Zero};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use subgradient::partition_slice;
mod geometry;
mod interval;
mod kbn_sum;
mod subgradient;

use take_until::TakeUntilExt;

use crate::kbn_sum::KbnSumIteratorExt;

/// A simple wrapper around `f64` that specifies a total, and hence not IEEE754-compatible, order.
pub type Floating = OrderedFloat<f64>;

/// The value of the objective function for a given line.
pub fn objective_value(line: PrimalLine, points: &[PrimalPoint]) -> Floating {
    points
        .iter()
        .copied()
        .map(|p| Floating::abs(&(line.eval_at(p.x()) - p.y())))
        .kbn_sum()
}

/// A state at one of the two interval boundaries maintained by the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AlgState {
    pub slope: Floating,
    pub median_line: DualLine,
    pub subgrad: ClosedInterval<Floating>,
    /// Value of the objective function (if it is known)
    pub obj_val: Option<Floating>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectiveType {
    Minimize,
    Maximize,
}

/// Solves the continuous knapsack problem via greedy selection.
/// In this `beta` references the transformed formulation of the problem (with variables beta in the paper).
/// So this solves
/// ```text
/// min_{\beta \in [0,1]^|I_0|} sum_i beta_i * key(value_i)
/// s.t. sum_i beta_i = capacity
/// ```
/// if `objective_type == ObjectiveType::Minimize`, otherwise it solves the analogous maximization problem.
fn solve_cont_knapsack_beta<T>(
    values: &mut [T],
    capacity: Floating,
    objective_type: ObjectiveType,
    mut key: impl FnMut(&T) -> Floating, // Closure to extract the value
) -> Floating {
    let n = values.len();
    let n_f = Floating::from(n as f64);

    if capacity.is_negative() || capacity > n_f {
        panic!("Capacity must be between 0.0 and {n_f}, but got {capacity}.");
    }
    if n == 0 || capacity == 0.0 {
        return Floating::zero();
    } else if n == 1 {
        // we already know that capacity <= n = 1 here
        return capacity * key(&values[0]);
    } else if capacity == n_f {
        return values.iter().map(key).kbn_sum();
    }

    // The number of items that will be fully "filled" (beta_i = 1.0).
    // this line has issues when the capacity is *giant* (i.e. outside the range representable by usize).
    // We don't think that we'll ever encounter such a case.
    let num_full_items = capacity.floor() as usize;

    let fractional_part = capacity - Floating::from(num_full_items as f64);

    // The pivot is the item at index `num_full_items`.
    // `select_nth_unstable_by` partitions the slice in-place around this index.
    let pivot_idx = num_full_items;
    match objective_type {
        ObjectiveType::Maximize => {
            // To maximize we sort in descending order (highest values first).
            values.select_nth_unstable_by(pivot_idx, |a: &T, b: &T| key(a).cmp(&key(b)).reverse());
        }
        ObjectiveType::Minimize => {
            // To maximize we sort in descending order (highest values first).
            values.select_nth_unstable_by(pivot_idx, |a: &T, b: &T| key(a).cmp(&key(b)));
        }
    }

    // `values` has now been partitioned.
    // The `num_full_items` largest values are in the slice `&values[..pivot_idx]`.
    // These items all have beta_i = 1.
    let sum_full_items = values[..pivot_idx].iter().map(&mut key).kbn_sum();

    // The pivot item is at `values[pivot_idx]`.
    // It is assigned the fractional part.
    let pivot_contribution = key(&values[pivot_idx]) * fractional_part;

    // All other items (with values smaller than the pivot) are assigned beta_i = 0.0
    // and thus contribute nothing to the sum.
    sum_full_items + pivot_contribution
}

/// A wrapper that computes either the min or max of `sum(alpha_i * x_i)`.
/// It returns the single calculated bound of the variable sum.
/// In this `alpha` references the untransformed formulation of the problem (with variables alpha in the paper).
/// So this solves
/// ```text
/// min_{\alpha \in [-1,1]^|I_0|} sum_i alpha_i * key(value_i)
/// s.t. sum_i alpha_i = -B
/// where B = |n_above| - |n_below|
/// ```
/// if `objective_type == ObjectiveType::Minimize`, otherwise it solves the analogous maximization problem.
fn solve_cont_knapsack_alpha<T>(
    items: &mut [T], // The items for the I_0 set
    n_below: u32,
    n_above: u32,
    objective: ObjectiveType,
    mut key: impl FnMut(&T) -> Floating,
) -> Floating {
    let n_equal = items.len() as u32;
    if n_equal == 0 {
        return Floating::zero();
    }

    // Target sum for the alpha coefficients: C_alpha = |I_-| - |I_+|
    let c_alpha = n_above as i64 - n_below as i64;

    // Transform to the beta-knapsack capacity: C_beta = (|I_0| + C_alpha) / 2
    let c_beta = Floating::from((n_equal as i64 + c_alpha) as f64) * 0.5;

    let sum_xs_equal = items.iter().map(&mut key).kbn_sum();

    match objective {
        ObjectiveType::Maximize => {
            // V_max = max(sum(beta_i * x_i))
            let v_max = solve_cont_knapsack_beta(items, c_beta, ObjectiveType::Maximize, key);
            // Return max_alpha_sum
            Floating::from(2.0) * v_max - sum_xs_equal
        }
        ObjectiveType::Minimize => {
            // V_min = min(sum(beta_i * x_i))
            let v_min = solve_cont_knapsack_beta(items, c_beta, ObjectiveType::Minimize, key);
            // Return min_alpha_sum
            Floating::from(2.0) * v_min - sum_xs_equal
        }
    }
}

/// Computes one bound (min or max) of the exact subdifferential interval.
fn compute_subgrad_bound<const N: usize>(
    lines: &mut [(DualLine, Floating)],
    median_idx: usize,
    objectives: [ObjectiveType; N],
) -> (DualLine, [Floating; N]) {
    // 1. Find the median value and partition the slice.
    let (median_line, median_value) = *lines
        .select_nth_unstable_by_key(median_idx, |(_, val)| *val)
        .1;

    // 2. Further partition into I<, I>, and I0.
    let (s_base, equal_to_median, n_below, n_above) = {
        let eps = Floating::from(1.0e-15);

        let (strictly_below, equal_and_above) =
            partition_slice(lines, |(_, val)| *val < median_value - eps);
        let (equal, strictly_above) =
            partition_slice(equal_and_above, |(_, val)| *val <= median_value + eps);
        /* Slower
        let (strictly_below, equal, strictly_above) = three_way_partition(lines, |(_, val)| {
            let diff = *val - median_value;
            if diff < -eps {
                std::cmp::Ordering::Less
            } else if diff > eps {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
         */

        // difference of the sums below and above the median
        let s_base = strictly_below
            .iter()
            .map(|(p, _)| p.dual().x())
            .chain(strictly_above.iter().map(|(p, _)| -p.dual().x()))
            .kbn_sum();

        (
            s_base,
            equal,
            strictly_below.len() as u32,
            strictly_above.len() as u32,
        )
    };

    let bounds = objectives.map(|objective| {
        let knapsack_bound = solve_cont_knapsack_alpha(
            equal_to_median,
            n_below,
            n_above,
            objective, // Pass the objective down
            |item: &(DualLine, Floating)| item.0.dual().x(),
        );
        s_base + knapsack_bound
    });
    (median_line, bounds)
}

/// Computes the partial subgradient of the objective function.
fn partial_subgrad(
    median_value: Floating,
    lines: &mut [(DualLine, Floating)],
) -> ClosedInterval<Floating> {
    let eps = Floating::from(1.0e-15);
    let (strictly_below_median, equal_to_median, strictly_above_median) = {
        let reference = median_value;
        /*
        let (lt, eq, gt) = three_way_partition(lines, |(_, val)| {
            let diff = *val - reference;
            if diff < -eps {
                std::cmp::Ordering::Less
            } else if diff > eps {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        */
        let (lt, geq) = partition_slice(lines, |(_, val)| *val < reference - eps);
        let (eq, gt) = partition_slice(geq, |(_, val)| *val <= reference + eps);
        (lt, eq, gt)
    };
    /*
               let n_below = strictly_below_median.len() as u32;
               let n_equal = equal_to_median.len() as u32;
               let n_above = strictly_above_median.len() as u32;
    */

    let sum_xs_equal = equal_to_median
        .iter()
        .map(|(p, _)| p.dual().x().abs())
        .kbn_sum();
    let sum_xs_below_minus_above = strictly_below_median
        .iter()
        .map(|(p, _)| p.dual().x())
        .chain(strictly_above_median.iter().map(|(p, _)| -p.dual().x()))
        .kbn_sum();

    ClosedInterval::new([
        sum_xs_below_minus_above - sum_xs_equal,
        sum_xs_below_minus_above + sum_xs_equal,
    ])
}

impl AlgState {
    /// Create a new AlgState at the given slope (for the L1 problem on the given lines) using the provided value buffer.
    /// `use_exact_subgrad` determines whether to use the exact subgradient or a partial subgradient that gives a superset of the actual one.
    pub fn new_with_val_buf(
        slope: Floating,
        lines: &mut [(DualLine, Floating)],
        use_exact_subgrad: bool,
    ) -> AlgState {
        lines.iter_mut().for_each(|(line, val)| {
            *val = line.eval_at(slope);
        });

        let n = lines.len();
        let (median_line, subgrad) = if use_exact_subgrad {
            if n % 2 == 1 {
                // ODD Case: Unique Median
                let median_idx = n / 2;

                // In the odd case, the partition for min and max is the same.
                // We can compute both bounds at once.
                let (median_line, bounds) = compute_subgrad_bound(
                    lines,
                    median_idx,
                    [ObjectiveType::Minimize, ObjectiveType::Maximize],
                );

                (median_line, ClosedInterval::new(bounds))
            } else {
                // EVEN Case: Lower and Upper Median
                let upper_median_idx = n / 2;
                let lower_median_idx = upper_median_idx - 1;

                let (median_line, [s_max]) =
                    compute_subgrad_bound(lines, upper_median_idx, [ObjectiveType::Maximize]);
                let (_, [s_min]) =
                    compute_subgrad_bound(lines, lower_median_idx, [ObjectiveType::Minimize]);
                (median_line, ClosedInterval::new([s_min, s_max]))
                /*
                let low_subgrad = compute_subgrad_bound(
                    lines,
                    lower_median_idx,
                    [ObjectiveType::Minimize, ObjectiveType::Maximize],
                );
                let high_subgrad = compute_subgrad_bound(
                    lines,
                    upper_median_idx,
                    [ObjectiveType::Minimize, ObjectiveType::Maximize],
                );
                */
                /*let (median_line, [s_min, s_max]) = compute_subgrad_bound(
                    lines,
                    lower_median_idx,
                    [ObjectiveType::Minimize, ObjectiveType::Maximize],
                );*/
                // (median_line, ClosedInterval::new([s_min, s_max]))
            }
        } else {
            let median_idx = n / 2;
            let (median_line, median_value) = *lines
                .select_nth_unstable_by_key(median_idx, |(_, val)| *val)
                .1;
            let partial_subgrad = partial_subgrad(median_value, lines);

            (median_line, partial_subgrad)
        };

        // let subgrad = subgrad_info.evaluate().0;
        AlgState {
            median_line,
            // subgrad_info,
            subgrad,
            slope,
            obj_val: None,
        }
    }

    /// Determine the L1 line estimate indicated by this state by evaluating its associated median line.
    #[inline]
    pub fn line_estimate(&self) -> PrimalLine {
        PrimalLine {
            coords: (self.slope, self.median_line.eval_at(self.slope)),
        }
    }

    /// Get the objective value of the line estimate indicated by this state, using a cached value if available.
    /// Costs O(n) if the cached value is not available.
    pub fn get_or_compute_obj_val_cached(&mut self, points: &[PrimalPoint]) -> Floating {
        let line_estimate = self.line_estimate();
        *self
            .obj_val
            .get_or_insert_with(|| objective_value(line_estimate, points))
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum L1LineObsStateType {
    NonStationary,
    Stationary,
}

impl L1LineObsState {
    pub fn is_stationary(&self) -> bool {
        match self.state_type {
            L1LineObsStateType::Stationary => true,
            L1LineObsStateType::NonStationary => false,
        }
    }
}

/// An observable (i.e. returned by the method) algorithm state for a single slope.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct L1LineObsState {
    /// The primal slope m of this state.
    pub slope: Floating,
    /// A dual line with median value at the slope m.
    pub median_line: DualLine,
    /// The L1 line estimate indicated by the median line.
    pub line_estimate: PrimalLine,
    /// Whether or not this state is already stationary.
    pub state_type: L1LineObsStateType,
    /// Value of the objective function (if it is known)
    pub obj_val: Option<Floating>,
}

impl L1LineObsState {
    pub fn get_or_compute_obj_val_noncached(&self, points: &[PrimalPoint]) -> Floating {
        self.obj_val
            .unwrap_or_else(|| objective_value(self.line_estimate, points))
    }
}

/// See [L1LineObsState].
/// This groups two of those for the two interval boundaries managed by the algorithm.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct PalpObsState {
    pub options: [L1LineObsState; 2],
    pub info: SolverInfo,
}

impl From<AlgState> for L1LineObsState {
    fn from(state: AlgState) -> Self {
        L1LineObsState {
            slope: state.slope,
            median_line: state.median_line,
            line_estimate: state.line_estimate(),
            // subgrad_info: state.subgrad_info,
            state_type: L1LineObsStateType::NonStationary,
            obj_val: state.obj_val,
        }
    }
}

/// Some auxiliary information about the solver.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub struct SolverInfo {
    /// Number of iterations performed by the solver.
    pub num_iters: usize,
    /// Number of expansion steps performed by the solver.
    pub num_expansion: usize,
    /// Number of subdivision steps performed by the solver.
    pub num_subdiv: usize,
}

/// The main struct implementing the actual solver logic (via its [Iterator] instance).
#[derive(Debug, Clone)]
pub struct PalpGen<'a, Delta = DoubleIntervalSize> {
    // state: Option<AlgState>,
    points: Cow<'a, [PrimalPoint]>, // this should probably be a &mut [PrimalPoint] instead
    // lines: Vec<DualLine>,
    line_val_buf: Vec<(DualLine, Floating)>,
    /// Whether we're currently "subdividing" or "expanding"
    subdividing: bool,
    options: [AlgState; 2],
    info: SolverInfo,
    fuse_blown: bool,
    stepsize_rule: Delta,
    use_exact_subgrad: bool,
}

/// How certain you are about the initial guess of the solution.
/// Zero uncertainty means that the initial guess is exact --- in which case there's no point in calling the solver.
/// The uncertainty is given relative to the size of the initial guess, for details please see the associated paper.
pub struct Uncertainty(pub Floating);

impl Default for Uncertainty {
    fn default() -> Self {
        Self(Floating::from(0.01))
    }
}

/// A trait for implementing different stepsize rules.
/// In the notation of the paper this is \delta_k / (µ |m_0|) where µ is the uncertainty and m_0 the initial guess.
pub trait StepsizeRule {
    fn stepsize(&mut self, num_iters: usize) -> Floating;
}

/// A stepsize rule that doubles the interval size with each iteration.
pub struct DoubleIntervalSize;

impl StepsizeRule for DoubleIntervalSize {
    #[inline(always)]
    fn stepsize(&mut self, num_iters: usize) -> Floating {
        Floating::from(1 << num_iters)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A supporting line to the objective function. Given in the form:
/// ```text
/// L(x) = grad * (x - eval_point) + obj_val
/// ```
///
struct AffineSupport {
    /// The point at which the objective function was evaluated.
    pub eval_point: Floating,
    /// The objective function function value at the given point.
    pub obj_val: Floating,
    /// The slope of the supporting line (a subgradient of the objective function at the given point).
    pub grad: Floating,
}

impl AffineSupport {
    /// Computes the point of intersection of two lines in a numerically stable way.
    ///
    /// This works by computing the intersection of the lines in shifted coordinates and then translating it back to improve numerical stability.
    pub fn intersect_stable(self, other: &AffineSupport) -> Floating {
        let a = self;
        let b = other;
        let midpoint = (a.eval_point + b.eval_point) * Floating::from(0.5);

        // shifted coordinates
        let a_shifted = a.eval_point - midpoint;
        let b_shifted = b.eval_point - midpoint;

        // intersection in shifted coords
        let denom = a.grad - b.grad;
        let numer = [
            b.obj_val,
            -a.obj_val,
            a.grad * a_shifted,
            -b.grad * b_shifted,
        ]
        .into_iter()
        .kbn_sum(); //(fb - fa) + (sa * a_shifted - sb * b_shifted);
        let x_p = numer / denom;

        // shift back
        midpoint + x_p
    }
}

impl<'a, Delta: StepsizeRule> PalpGen<'a, Delta> {
    pub fn new(
        some_primal_slope: Floating,
        points: Cow<'a, [PrimalPoint]>,
        uncertainty: Uncertainty,
        stepsize_rule: Delta,
    ) -> Self {
        let mut line_val_buf = Vec::with_capacity(points.len());
        line_val_buf.extend(
            points
                .iter()
                .copied()
                .map(PrimalPoint::dual)
                .map(|l| (l, Floating::zero())),
        );

        let use_exact_subgrad = true;

        // determine two slopes whose accompanying subgradients hopefully have different (uniform) signs
        let options = [
            some_primal_slope * (Floating::one() + uncertainty.0),
            some_primal_slope * (Floating::one() - uncertainty.0),
        ]
        .map(|slope| AlgState::new_with_val_buf(slope, &mut line_val_buf, use_exact_subgrad));
        Self {
            points,
            line_val_buf,
            subdividing: false,
            options,
            fuse_blown: false,
            info: SolverInfo::default(),
            stepsize_rule,
            use_exact_subgrad,
        }
    }
}

impl<Delta: StepsizeRule> PalpGen<'_, Delta> {
    #[inline]
    fn finalize_with_a_optimal(&mut self) -> PalpObsState {
        let [a, b] = self.options;
        let options = [
            L1LineObsState {
                state_type: L1LineObsStateType::Stationary,
                ..L1LineObsState::from(a)
            },
            L1LineObsState::from(b),
        ];
        self.fuse_blown = true;
        PalpObsState {
            options,
            info: self.info,
        }
    }

    #[inline]
    fn finalize_with_b_optimal(&mut self) -> PalpObsState {
        let [a, b] = self.options;
        let options = [
            L1LineObsState::from(a),
            L1LineObsState {
                state_type: L1LineObsStateType::Stationary,
                ..L1LineObsState::from(b)
            },
        ];
        self.fuse_blown = true;
        PalpObsState {
            options,
            info: self.info,
        }
    }

    #[inline]
    fn subdivide(&mut self) -> PalpObsState {
        self.info.num_subdiv += 1;
        if !self.subdividing {
            self.subdividing = true;
        }

        self.options.sort_by(|s, t| s.slope.cmp(&t.slope));
        let [mut a, mut b] = self.options;

        // Compute objective value for both options (we cache these values in the AlgState of either option)
        let fa = a.get_or_compute_obj_val_cached(&self.points);
        let fb = b.get_or_compute_obj_val_cached(&self.points);

        let next_slope = {
            let support_at_a = AffineSupport {
                obj_val: fa,
                eval_point: a.slope,
                grad: a.subgrad.max(),
            };
            let support_at_b = AffineSupport {
                obj_val: fb,
                eval_point: b.slope,
                grad: b.subgrad.min(),
            };
            let intersection_slope = support_at_a.intersect_stable(&support_at_b);

            // Note: we tested multiple other eps rules here (e.g. to allow steps closer to the boundary later on or stuff like that),
            // in particular also linear interpolation and smoothstep. But we found that this simple rule works best.
            let eps = (b.slope - a.slope).abs() * Floating::from(0.01);

            // check that the proposed slope is "sufficiently far inside the interior" of the current interval
            if intersection_slope <= a.slope || (intersection_slope - a.slope).abs() < eps {
                a.slope + eps
            } else if intersection_slope >= b.slope || (b.slope - intersection_slope).abs() < eps {
                b.slope - eps
            } else {
                intersection_slope
            }
        };

        // let next_state = AlgState::new(next_slope, &mut self.lines, &self.points);
        let next_state =
            AlgState::new_with_val_buf(next_slope, &mut self.line_val_buf, self.use_exact_subgrad);

        let sign_next = next_state.subgrad.uniform_sign();
        if sign_next == Sign::Zero || sign_next == a.subgrad.uniform_sign() {
            self.options[0] = next_state;
        } else if sign_next == b.subgrad.uniform_sign() {
            self.options[1] = next_state;
        } else {
            unreachable!()
        }
        let options = self.options.map(L1LineObsState::from);
        PalpObsState {
            options,
            info: self.info,
        }
    }

    #[inline]
    fn expand(&mut self) -> PalpObsState {
        self.info.num_expansion += 1;
        let direction = -self.options[0].subgrad.uniform_sign();

        let [a, b] = self.options;
        self.options = match direction {
            Sign::Pos => {
                let new_b = AlgState::new_with_val_buf(
                    b.slope + self.stepsize_rule.stepsize(self.info.num_iters),
                    &mut self.line_val_buf,
                    self.use_exact_subgrad,
                );
                [b, new_b]
            }
            Sign::Neg => {
                let new_a = AlgState::new_with_val_buf(
                    a.slope - self.stepsize_rule.stepsize(self.info.num_iters),
                    &mut self.line_val_buf,
                    self.use_exact_subgrad,
                );
                [new_a, a]
            }
            Sign::Zero => unreachable!(),
        };
        let options = self.options.map(L1LineObsState::from);
        PalpObsState {
            options,
            info: self.info,
        }
    }
}

impl<Delta: StepsizeRule> Iterator for PalpGen<'_, Delta> {
    type Item = PalpObsState;
    /// Advances the main algorithm to the next state by taking an expansion or subdivision step.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.fuse_blown {
            return None;
        }
        self.info.num_iters += 1;
        let [ref a, ref b] = self.options;

        match (a.subgrad.uniform_sign(), b.subgrad.uniform_sign()) {
            (Sign::Zero, _) => {
                if self.use_exact_subgrad {
                    Some(self.finalize_with_a_optimal())
                } else {
                    self.use_exact_subgrad = true;
                    self.options = self.options.map(|state| {
                        AlgState::new_with_val_buf(state.slope, &mut self.line_val_buf, true)
                    });
                    self.next()
                }
            }
            (_, Sign::Zero) => {
                if self.use_exact_subgrad {
                    Some(self.finalize_with_b_optimal())
                } else {
                    self.use_exact_subgrad = true;
                    self.options = self.options.map(|state| {
                        AlgState::new_with_val_buf(state.slope, &mut self.line_val_buf, true)
                    });
                    self.next()
                }
            }
            _ if self.info.num_iters == 1 => {
                // first step should always return the "starting guess". This isn't really needed, but it's "nice".
                Some(PalpObsState {
                    options: [L1LineObsState::from(*a), L1LineObsState::from(*b)],
                    info: self.info,
                })
            }
            (Sign::Pos, Sign::Neg) | (Sign::Neg, Sign::Pos) => {
                if !self.subdividing {
                    self.subdividing = true;
                }
                Some(self.subdivide())
            }
            (Sign::Pos, Sign::Pos) | (Sign::Neg, Sign::Neg) => Some(self.expand()),
        }
    }
}

/// Compute the least-absolute-deviations line for a given collection of points using the Piecewise Affine Lower Bounding (PALB) method.
pub fn l1line(points: &[PrimalPoint]) -> Option<PrimalLine> {
    l1line_with_info::<true>(points).map(|sol| sol.optimal_line)
}

/// A solution to the least-absolute-deviations line problem.
/// Consists of the optimal line, the associated objective value, and some solver statistics (like the number of iterations).
pub struct Solution {
    pub optimal_line: PrimalLine,
    pub objective_value: Floating,
    pub info: SolverInfo,
}

#[inline]
/// Apply an affine coordinate transformation to the given points to improve numerical stability. Maps all points into the square [-1,1]².
/// The returned closure implements the inverse transformation.
/// Allocation could be avoided at this point by overwriting the given points.
fn get_transform(
    points: &[PrimalPoint],
) -> Option<(Vec<PrimalPoint>, impl Fn(Solution) -> Solution)> {
    let n = points.len();
    if n < 2 {
        None
    } else {
        let n_float = Floating::from(points.len() as f64);
        let x_mean = points.iter().map(|p| p.x()).sum::<Floating>() / n_float;
        let y_mean = points.iter().map(|p| p.y()).sum::<Floating>() / n_float;

        let translated_points: Vec<_> = points
            .iter()
            .map(|p| PrimalPoint {
                coords: (p.x() - x_mean, p.y() - y_mean),
            })
            .collect();

        let x_scaling = translated_points
            .iter()
            .map(|p| p.x().abs())
            .max()
            .unwrap_or(Floating::from(1.0));
        let y_scaling = translated_points
            .iter()
            .map(|p| p.y().abs())
            .max()
            .unwrap_or(Floating::from(1.0));

        // handle degenerate cases
        let x_scaling = if x_scaling == Floating::zero() {
            Floating::from(1.0)
        } else {
            x_scaling
        };
        let y_scaling = if y_scaling == Floating::zero() {
            Floating::from(1.0)
        } else {
            y_scaling
        };

        let scaled_points: Vec<_> = translated_points
            .iter()
            .map(|p| PrimalPoint {
                coords: (p.x() / x_scaling, p.y() / y_scaling),
            })
            .collect();

        let inverse_transform = move |mut solution: Solution| {
            let (slope_scaled, intercept_scaled) = solution.optimal_line.coords;

            let slope = slope_scaled * (y_scaling / x_scaling);
            let intercept = intercept_scaled * y_scaling + y_mean - slope * x_mean;

            solution.optimal_line.coords = (slope, intercept);
            solution.objective_value *= y_scaling;

            solution
        };

        Some((scaled_points, inverse_transform))
    }
}

/// Calculates the slope of the L2 regression line (ordinary least squares).
///
/// Returns `None` if the slope is undefined, which occurs if:
/// 1. There are fewer than 2 points.
/// 2. All points have the same x-coordinate (a vertical line).
///
/// TODO: Better implementation. Maybe using Welford's algorithm --- maybe just using kbn_sum instead of sum.
pub fn least_squares_slope(points: &[PrimalPoint]) -> Option<Floating> {
    let n = points.len();
    if n < 2 {
        return None;
    }

    let n_float = Floating::from(n as f64);
    let mean_x = points.iter().map(|p| p.x()).sum::<Floating>() / n_float;
    let mean_y = points.iter().map(|p| p.y()).sum::<Floating>() / n_float;

    let mut numerator = Floating::zero();
    let mut denominator = Floating::zero();

    for p in points {
        let dx = p.x() - mean_x;
        let dy = p.y() - mean_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    if denominator.is_zero() {
        return None;
    }

    Some(numerator / denominator)
}

/// Compute the least-absolute-deviations line for a given collection of points using the Piecewise Affine Lower Bounding (PALB) method.
/// Also return some informations about the solver like the number of iterations it took etc.
pub fn l1line_with_info<const NORMALIZE_INPUT: bool>(points: &[PrimalPoint]) -> Option<Solution> {
    let (points, inv_transform): (Cow<[PrimalPoint]>, Option<_>) =
        if NORMALIZE_INPUT && points.len() > 1 {
            let (owned_points, inv_transform) = get_transform(points).expect("Internal error");
            (Cow::Owned(owned_points), Some(inv_transform))
        } else {
            (Cow::Borrowed(points), None)
        };

    let starting_slope = match points.as_ref() {
        [] => {
            return None;
        }
        [p] => {
            return Some(Solution {
                optimal_line: PrimalLine {
                    coords: (Floating::zero(), p.y()),
                },
                objective_value: Floating::zero(),
                info: SolverInfo {
                    num_iters: 0,
                    num_expansion: 0,
                    num_subdiv: 0,
                },
            });
        }
        points @ [p1, .., p2] if points.len() <= 100 => (p1.y() - p2.y()) / (p1.x() - p2.x()),
        points => {
            //let mut rng = rand::rng();
            //let mut points = points.to_owned();
            //let (ten_points, _rest) = points.partial_shuffle(&mut rng, 10);
            //let ten_points = first_and_last_5(points).unwrap();
            //l1line(&ten_points).unwrap().slope()
            least_squares_slope(points).unwrap_or_else(|| {
                let mut rng = rand::rng();
                let mut points = points.to_owned();
                let (sample_of_points, _rest) = points.partial_shuffle(&mut rng, 100);
                l1line(sample_of_points).unwrap().slope()
            })
        } /*points if points.len() <= 10_000 => {
              let mut rng = rand::rng();
              let mut points = points.to_owned();
              let (sample, _rest) = points.partial_shuffle(&mut rng, 100);
              // let twenty_points = first_and_last_10(points).unwrap();
              l1line(&sample).unwrap().slope()
          }
          points => {
              //let hundred_points = first_and_last_50(points).unwrap();
              //l1line(&hundred_points).unwrap().slope()
              let mut rng = rand::rng();
              let mut points = points.to_owned();
              let (sample, _rest) = points.partial_shuffle(&mut rng, 1000);
              l1line(&sample).unwrap().slope()
          }*/
    };

    let max_steps = 15 * (points.len().ilog10() as usize) + 300;
    PalpGen::new(
        starting_slope,
        points.clone(),
        Uncertainty::default(),
        DoubleIntervalSize,
    )
    .take_until(|obs_state| {
        (obs_state.options[0].slope - obs_state.options[1].slope).abs() < Floating::from(1e-15)
    }) // stop iteration once the interval gets *tiny* (if that ever happens)
    .take(max_steps) // at most this many iterations, then we bail out
    .last()
    .map(|obs_state| {
        (
            obs_state
                .options
                .into_iter()
                .min_by_key(|state| state.get_or_compute_obj_val_noncached(points.as_ref()))
                .unwrap(),
            obs_state.info,
        )
    })
    .map(|(state, info)| Solution {
        optimal_line: state.line_estimate,
        objective_value: state
            .obj_val
            .unwrap_or_else(|| state.get_or_compute_obj_val_noncached(points.as_ref())),
        info,
    })
    .map(|sol| {
        if let Some(transf) = inv_transform {
            (transf)(sol)
        } else {
            sol
        }
    })
}

#[cfg(test)]
mod tests_bisect {
    use crate::l1line;
    use crate::objective_value;

    use super::{Floating, PrimalLine, PrimalPoint};
    use approx::{assert_abs_diff_eq, relative_eq};
    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    /// Generates `n` random points with x uniformly distributed in [0,1] and
    /// y = 3.0 * x - 2.0 + noise, where noise is uniformly distributed in [-0.2, 0.2].
    /// The random number generator is seeded for reproducibility.
    pub fn generate_random_points(
        n_samples: usize,
        seed: u64,
        ground_truth: PrimalLine,
    ) -> Vec<PrimalPoint> {
        let mut rng = StdRng::seed_from_u64(seed); // Create a seeded random number generator
        let mut points = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let x = rng.random_range(0.0..=1.0);
            let noise = rng.random_range(-0.2..=0.2);
            let y = ground_truth.eval_at(Floating::from(x)) + noise;
            points.push(PrimalPoint {
                coords: (Floating::from(x), y),
            });
        }

        points
    }

    fn assert_solution_likely_correct(
        solution: PrimalLine,
        ground_truth: PrimalLine,
        points: &[PrimalPoint],
    ) {
        let our_solution_objective = objective_value(solution, &points);
        let ground_truth_objective = objective_value(ground_truth, &points);
        dbg!(
            Floating::from(
                (our_solution_objective - ground_truth_objective) / our_solution_objective
            )
            .abs()
        );
        assert!(
            dbg!(our_solution_objective) <= dbg!(ground_truth_objective)
                || relative_eq!(
                    Floating::from(our_solution_objective).into_inner(),
                    Floating::from(ground_truth_objective).into_inner(),
                    max_relative = 5.0e-2
                )
        );
        assert_abs_diff_eq!(
            Floating::from(solution.slope()).into_inner(),
            Floating::from(ground_truth.slope()).into_inner(),
            epsilon = 5.0e-1,
        );
        assert_abs_diff_eq!(
            Floating::from(solution.intercept()).into_inner(),
            Floating::from(ground_truth.intercept()).into_inner(),
            epsilon = 5.0e-1,
        );
    }

    #[test]
    fn works_small() {
        //INIT.call_once(|| pretty_env_logger::init());
        let ground_truth = PrimalLine {
            coords: (Floating::from(-3.0), Floating::from(-2.0)),
        };
        let points = generate_random_points(10, 0, ground_truth);
        dbg!(points.iter().map(|p| p.x()).collect_vec());
        dbg!(points.iter().map(|p| p.y()).collect_vec());
        let res = l1line(&points).unwrap();
        assert_solution_likely_correct(res, ground_truth, &points);
    }

    #[test]
    fn works_med() {
        //INIT.call_once(|| pretty_env_logger::init());
        let ground_truth = PrimalLine {
            coords: (Floating::from(-3.0), Floating::from(-2.0)),
        };
        let points = generate_random_points(100, 0, ground_truth);
        let res = l1line(&points).unwrap();
        assert_solution_likely_correct(res, ground_truth, &points);
    }

    #[test]
    fn works_random() {
        // INIT.call_once(|| pretty_env_logger::init());

        let normal_xs = Normal::new(0.0, 100.0).unwrap();
        let normal_ys = Normal::new(0.0, 100.0).unwrap();

        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let ground_truth = PrimalLine {
                coords: (
                    Floating::from(normal_xs.sample(&mut rng)),
                    Floating::from(normal_ys.sample(&mut rng)),
                ),
            };

            let points = generate_random_points(100, seed + 1, ground_truth);

            eprintln!("points {} = [", seed);
            for p in &points {
                eprintln!("  [{},{}],", p.x(), p.y());
            }
            eprintln!("]");

            let res = l1line(&points).unwrap();
            assert_solution_likely_correct(res, ground_truth, &points);
        }
    }

    #[test]
    fn works_particular1() {
        //INIT.call_once(|| pretty_env_logger::init());
        let ground_truth = PrimalLine {
            coords: (
                Floating::from(-97.7302306198001),
                Floating::from(-197.27940523542932),
            ),
        };
        let points = generate_random_points(10, 2 + 1, ground_truth);

        eprintln!("points = [");
        for p in &points {
            eprintln!("  [{},{}],", p.x(), p.y());
        }
        eprintln!("]");

        let res = l1line(&points).unwrap();
        eprintln!("res = {:?}", &res);
        assert_solution_likely_correct(res, ground_truth, &points);
    }

    #[test]
    fn works_particular2() {
        //INIT.call_once(|| pretty_env_logger::init());
        let ground_truth = PrimalLine {
            coords: (
                Floating::from(71.28130103834549),
                Floating::from(85.83314468179),
            ),
        };
        let points = generate_random_points(30, 2 + 1, ground_truth);
        let res = l1line(&points).unwrap();
        assert_solution_likely_correct(res, ground_truth, &points);
    }
}
