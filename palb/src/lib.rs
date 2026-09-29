//! PALB is an exact, robust, high-performance solver for the Least-Absolute-Deviations-Line (LAD) problem, i.e. one dimensional affine linear L1 regression.
//! This is the Rust core; be aware that there is also a Python API (`palb_py`).
pub use geometry::{Dual, DualLine, PrimalLine, PrimalPoint};
use interval::{ClosedInterval, Sign};
use itertools::Itertools;
use num_traits::{One, Signed, Zero};
use ordered_float::OrderedFloat;
use rand::{SeedableRng, rngs::ChaCha8Rng};

use subgradient::partition_slice;
mod geometry;
mod interval;
mod kbn_sum;
mod subgradient;

use take_until::TakeUntilExt;

pub use crate::kbn_sum::KbnSumIteratorExt;

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
#[inline]
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
pub struct PalbObsState {
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
#[derive(Debug)]
pub struct PalbGen<'a, Buf, Delta = DoubleIntervalSize>
where
    Buf: AsMut<[(DualLine, Floating)]>,
{
    // state: Option<AlgState>,
    points: &'a mut [PrimalPoint],
    // lines: Vec<DualLine>,
    line_val_buf: Buf,
    /// Whether we're currently "subdividing" or "expanding"
    subdividing: bool,
    options: [AlgState; 2],
    info: SolverInfo,
    fuse_blown: bool,
    stepsize_rule: Delta,
    use_exact_subgrad: bool,
    uncertainty: Uncertainty,
    initial_slope: Floating,
}

/// How certain you are about the initial guess of the solution.
/// Zero uncertainty means that the initial guess is exact --- in which case there's no point in calling the solver.
/// The uncertainty is given relative to the size of the initial guess, for details please see the associated paper.
#[derive(Debug, Clone, Copy)]
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
        // f64 exactly represents powers of 2 up to 2^1023.
        // Cap at 1023 to avoid hitting f64::INFINITY.
        let val = 2.0_f64.powi((num_iters as i32).min(f64::MAX_EXP - 1));
        Floating::from(val)
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
    #[inline]
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

impl<'a, Delta: StepsizeRule> PalbGen<'a, Vec<(DualLine, Floating)>, Delta> {
    pub fn new(
        some_primal_slope: Floating,
        points: &'a mut [PrimalPoint],
        uncertainty: Uncertainty,
        stepsize_rule: Delta,
    ) -> Self {
        let line_val_buf = vec![Default::default(); points.len()];
        Self::new_with_val_buf(
            some_primal_slope,
            points,
            line_val_buf,
            uncertainty,
            stepsize_rule,
        )
        .unwrap()
    }
}

impl<'a, Buf: AsMut<[(DualLine, Floating)]>, Delta: StepsizeRule> PalbGen<'a, Buf, Delta> {
    pub fn new_with_val_buf(
        some_primal_slope: Floating,
        points: &'a mut [PrimalPoint],
        mut line_val_buf: Buf,
        uncertainty: Uncertainty,
        stepsize_rule: Delta,
    ) -> Option<Self> {
        let buf_slice = line_val_buf.as_mut();

        if buf_slice.len() < points.len() {
            // Buffer is too small
            return None;
        }

        for (buf_slot, p) in buf_slice.iter_mut().zip(points.iter()) {
            *buf_slot = (p.dual(), Floating::zero());
        }

        let use_exact_subgrad = true;

        // determine two slopes whose accompanying subgradients hopefully have different (uniform) signs
        let some_primal_slope = if some_primal_slope.is_zero() {
            // choosing the uncertainty µ´ at this point results in one initial slope being zero and the other being 2µ.
            uncertainty.0
        } else {
            some_primal_slope
        };
        let options = [
            some_primal_slope * (Floating::one() + uncertainty.0),
            some_primal_slope * (Floating::one() - uncertainty.0),
        ]
        .map(|slope| AlgState::new_with_val_buf(slope, buf_slice, use_exact_subgrad));
        Some(Self {
            points,
            line_val_buf,
            subdividing: false,
            options,
            fuse_blown: false,
            info: SolverInfo::default(),
            stepsize_rule,
            use_exact_subgrad,
            uncertainty,
            initial_slope: some_primal_slope.abs(),
        })
    }
}

impl<Buf: AsMut<[(DualLine, Floating)]>, Delta: StepsizeRule> PalbGen<'_, Buf, Delta> {
    #[inline]
    fn finalize_with_a_optimal(&mut self) -> PalbObsState {
        let [a, b] = self.options;
        let options = [
            L1LineObsState {
                state_type: L1LineObsStateType::Stationary,
                ..L1LineObsState::from(a)
            },
            L1LineObsState::from(b),
        ];
        self.fuse_blown = true;
        PalbObsState {
            options,
            info: self.info,
        }
    }

    #[inline]
    fn finalize_with_b_optimal(&mut self) -> PalbObsState {
        let [a, b] = self.options;
        let options = [
            L1LineObsState::from(a),
            L1LineObsState {
                state_type: L1LineObsStateType::Stationary,
                ..L1LineObsState::from(b)
            },
        ];
        self.fuse_blown = true;
        PalbObsState {
            options,
            info: self.info,
        }
    }

    #[inline]
    fn subdivide(&mut self) -> PalbObsState {
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
        let next_state = AlgState::new_with_val_buf(
            next_slope,
            self.line_val_buf.as_mut(),
            self.use_exact_subgrad,
        );

        let sign_next = next_state.subgrad.uniform_sign();
        if sign_next == Sign::Zero || sign_next == a.subgrad.uniform_sign() {
            self.options[0] = next_state;
        } else if sign_next == b.subgrad.uniform_sign() {
            self.options[1] = next_state;
        } else {
            unreachable!()
        }
        let options = self.options.map(L1LineObsState::from);
        PalbObsState {
            options,
            info: self.info,
        }
    }

    #[inline]
    fn expand(&mut self) -> PalbObsState {
        self.info.num_expansion += 1;
        let direction = -self.options[0].subgrad.uniform_sign();

        let [a, b] = self.options;
        self.options = match direction {
            Sign::Pos => {
                let new_b = AlgState::new_with_val_buf(
                    b.slope
                        + self.uncertainty.0
                            * self.initial_slope
                            * self.stepsize_rule.stepsize(self.info.num_iters),
                    self.line_val_buf.as_mut(),
                    self.use_exact_subgrad,
                );
                [b, new_b]
            }
            Sign::Neg => {
                let new_a = AlgState::new_with_val_buf(
                    a.slope
                        - self.uncertainty.0
                            * self.initial_slope
                            * self.stepsize_rule.stepsize(self.info.num_iters),
                    self.line_val_buf.as_mut(),
                    self.use_exact_subgrad,
                );
                [new_a, a]
            }
            Sign::Zero => unreachable!(),
        };
        let options = self.options.map(L1LineObsState::from);
        PalbObsState {
            options,
            info: self.info,
        }
    }
}

impl<Buf: AsMut<[(DualLine, Floating)]>, Delta: StepsizeRule> Iterator for PalbGen<'_, Buf, Delta> {
    type Item = PalbObsState;
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
                        AlgState::new_with_val_buf(state.slope, self.line_val_buf.as_mut(), true)
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
                        AlgState::new_with_val_buf(state.slope, self.line_val_buf.as_mut(), true)
                    });
                    self.next()
                }
            }
            _ if self.info.num_iters == 1 => {
                // first step should always return the "starting guess". This isn't really needed, but it's "nice".
                Some(PalbObsState {
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
pub fn l1line(points: &mut [PrimalPoint]) -> Option<PrimalLine> {
    l1line_with_info::<true>(points).map(|sol| sol.optimal_line)
}

/// Compute the least-absolute-deviations line for a given collection of points using the Piecewise Affine Lower Bounding (PALB) method
/// given some starting slope and uncertainty.
/// Also return some informations about the solver like the number of iterations it took etc.
pub fn l1line_with_initial_guess<const NORMALIZE_INPUT: bool>(
    points: &mut [PrimalPoint],
    mut initial_slope: Option<Floating>,
    uncertainty: Option<Uncertainty>,
) -> Option<Solution> {
    let inv_transform: Option<_> = if NORMALIZE_INPUT && points.len() > 1 {
        let (inv_transform, aff) = get_transform(points).expect("Internal error");
        if let Some(ref mut slope) = initial_slope {
            *slope = *slope * aff.scaling.0 / aff.scaling.1;
        }
        Some(inv_transform)
    } else {
        None
    };
    let transform_back_or_dont = |sol: Solution| {
        if let Some(transf) = inv_transform {
            (transf)(sol)
        } else {
            sol
        }
    };
    match trivial_solution_or_slope(points) {
        None => None,
        Some(TrivialSolutionOrSlope::ProblemTrivial(sol)) => Some(transform_back_or_dont(sol)),
        Some(TrivialSolutionOrSlope::Slope(default_starting_slope)) => {
            let max_steps = 15 * (points.len().ilog10() as usize) + 300;
            PalbGen::new(
                initial_slope.unwrap_or(default_starting_slope),
                points,
                uncertainty.unwrap_or_default(),
                DoubleIntervalSize,
            )
            .take_until(|obs_state| {
                (obs_state.options[0].slope - obs_state.options[1].slope).abs()
                    < Floating::from(1e-15)
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
            .map(transform_back_or_dont)
        }
    }
}

pub struct Solution {
    pub optimal_line: PrimalLine,
    pub objective_value: Floating,
    pub info: SolverInfo,
}

struct AffineTransform {
    scaling: (Floating, Floating),
    #[allow(unused)]
    translation: (Floating, Floating),
}

#[inline]
/// Apply an affine coordinate transformation to the given points in-place to improve numerical stability.
/// Returns the inverse transform on solutions, as well as the affine transform which was applied to the data.
fn get_transform(
    points: &mut [PrimalPoint],
) -> Option<(impl Fn(Solution) -> Solution + 'static, AffineTransform)> {
    let n = points.len();
    if n < 2 {
        None
    } else {
        let n_float = Floating::from(points.len() as f64);
        let z = Floating::from(0.0);
        #[inline(always)]
        fn app2<T1, T2, S1, S2>(
            (x, y): (T1, S1),
            f: impl FnOnce(T1) -> T2,
            g: impl FnOnce(S1) -> S2,
        ) -> (T2, S2) {
            (f(x), g(y))
        }
        let (x_mean, y_mean) = app2(
            points.iter().fold((z, z), |(x_mean, y_mean), p| {
                (x_mean + p.x(), y_mean + p.y())
            }),
            |x| x / n_float,
            |y| y / n_float,
        );

        let one = Floating::from(1.0);
        let (x_scaling, y_scaling) = app2(
            points
                .iter()
                .map(|p| (p.x() - x_mean, p.y() - y_mean))
                .fold((z, z), |(x_scaling, y_scaling), (x, y)| {
                    (x_scaling.max(x.abs()), y_scaling.max(y.abs()))
                }),
            |x_scaling| if x_scaling.is_zero() { one } else { x_scaling },
            |y_scaling| if y_scaling.is_zero() { one } else { y_scaling },
        );

        // Apply both translation and scaling in a single in-place mutation pass
        for p in points.iter_mut() {
            p.coords = ((p.x() - x_mean) / x_scaling, (p.y() - y_mean) / y_scaling);
        }

        let inverse_transform = move |mut solution: Solution| {
            let (slope_scaled, intercept_scaled) = solution.optimal_line.coords;

            let slope = slope_scaled * (y_scaling / x_scaling);
            let intercept = intercept_scaled * y_scaling + y_mean - slope * x_mean;

            solution.optimal_line.coords = (slope, intercept);
            solution.objective_value *= y_scaling;

            solution
        };

        Some((
            inverse_transform,
            AffineTransform {
                scaling: (x_scaling, y_scaling),
                translation: (-x_mean, -y_mean),
            },
        ))
    }
}

pub enum LeastSquaresSlopeResult {
    NoPoints,
    VerticalLine,
    SlopeUnstable {
        numerator: Floating,
        denominator: Floating,
    },
    Slope(Floating),
}

impl LeastSquaresSlopeResult {
    /// Converts the result to a canonical form, returning `None` for ill-posed problems
    /// and a slope of zero for vertical lines or single points.
    pub fn canonicalize(self) -> Option<Floating> {
        match self {
            LeastSquaresSlopeResult::NoPoints => None,
            LeastSquaresSlopeResult::VerticalLine
            | LeastSquaresSlopeResult::SlopeUnstable { .. } => Some(Floating::zero()),
            LeastSquaresSlopeResult::Slope(s) => Some(s),
        }
    }
}

/// Calculates the slope of the L2 regression line (ordinary least squares).
///
/// Returns [NoPoints] if there are no points,
/// VerticalLine if all points have exactly the same x-coordinate (in particular if there is just one point),
/// SlopeUnstable(s) if the x-coordinates are within epsilon of each other (i.e. the line is nearly vertical),
/// and Slope(s) otherwise.
pub fn least_squares_slope(points: &[PrimalPoint], epsilon: Floating) -> LeastSquaresSlopeResult {
    let n = points.len();
    if n == 0 {
        return LeastSquaresSlopeResult::NoPoints;
    } else if n == 1 {
        // if there's just one point then all xs are equal i.e. we have a "vertical" line
        return LeastSquaresSlopeResult::VerticalLine;
    }

    let n_float = Floating::from(n as f64);
    let mean_x = points.iter().map(|p| p.x()).kbn_sum() / n_float;
    let mean_y = points.iter().map(|p| p.y()).kbn_sum() / n_float;

    // Numerator:   sum((x_i - mean_x) * (y_i - mean_y))
    // Denominator: sum((x_i - mean_x)^2)
    let numerator = points
        .iter()
        .map(|p| (p.x() - mean_x) * (p.y() - mean_y))
        .kbn_sum();
    let denominator = points
        .iter()
        .map(|p| {
            let dx = p.x() - mean_x;
            dx * dx
        })
        .kbn_sum();

    if denominator.is_zero() {
        // All xs coincide with the mean, in particular they are equal.
        return LeastSquaresSlopeResult::VerticalLine;
    } else if denominator.abs() < epsilon {
        // Same as above modulo some epsilon
        return LeastSquaresSlopeResult::SlopeUnstable {
            numerator,
            denominator,
        };
    } else {
        LeastSquaresSlopeResult::Slope(numerator / denominator)
    }
}

enum TrivialSolutionOrSlope {
    ProblemTrivial(Solution),
    Slope(Floating),
}

// Returns None if the problem is not well-posed (there are no points)
// If all points have the same x-value, the slope is initialized to zero
// (the optimal line then has a median of the y-values as intercept)
fn trivial_solution_or_slope(points: &[PrimalPoint]) -> Option<TrivialSolutionOrSlope> {
    match points.as_ref() {
        [] => None,
        [p] => {
            let sol = Solution {
                optimal_line: PrimalLine {
                    coords: (Floating::zero(), p.y()),
                },
                objective_value: Floating::zero(),
                info: SolverInfo {
                    num_iters: 0,
                    num_expansion: 0,
                    num_subdiv: 0,
                },
            };
            Some(TrivialSolutionOrSlope::ProblemTrivial(sol))
        }
        points @ [p1, ..] if points.len() <= 100 => {
            // try to find a point with a different x-coordinate than p1
            if let Some(p2) = points.iter().rev().find(|p| p.x() != p1.x()) {
                let slope = (p1.y() - p2.y()) / (p1.x() - p2.x());
                Some(TrivialSolutionOrSlope::Slope(slope))
            } else {
                // Degenerate case: all points share the exact same x-coordinate.
                // One possible optimal line has a slope of zero and any median of the y-values as intercept;
                // we hence return this as the optimal line.

                // we make a copy of the y-values so that we can sort them without modifying the original
                let mut ys = points.iter().copied().map(PrimalPoint::y).collect_vec();
                let mid = ys.len() / 2;
                let (_, &mut median, _) = ys.select_nth_unstable(mid);
                let optimal_line = PrimalLine {
                    coords: (Floating::zero(), median),
                };
                let sol = Solution {
                    optimal_line,
                    objective_value: objective_value(optimal_line, points),
                    info: SolverInfo::default(),
                };
                Some(TrivialSolutionOrSlope::ProblemTrivial(sol))
            }
        }
        points => {
            let slope = match least_squares_slope(points, Floating::from(1e-10)) {
                LeastSquaresSlopeResult::NoPoints => return None,
                LeastSquaresSlopeResult::VerticalLine => Floating::zero(),
                LeastSquaresSlopeResult::SlopeUnstable { .. } => {
                    // we set some arbitrary seed for reproducibility
                    let seed: [u8; 32] = [142; 32];
                    let mut rng = ChaCha8Rng::from_seed(seed);
                    // sample indices (to avoid cloning all points)
                    let sample_indices = rand::seq::index::sample(&mut rng, points.len(), 100);
                    let mut sample_of_points =
                        sample_indices.into_iter().map(|i| points[i]).collect_vec();
                    l1line(&mut sample_of_points).unwrap().slope()
                }
                LeastSquaresSlopeResult::Slope(s) => s,
            };
            Some(TrivialSolutionOrSlope::Slope(slope))
        }
    }
}

/// Compute the least-absolute-deviations line for a given collection of points using the Piecewise Affine Lower Bounding (PALB) method.
/// Also return some informations about the solver like the number of iterations it took etc.
pub fn l1line_with_info<const NORMALIZE_INPUT: bool>(
    points: &mut [PrimalPoint],
) -> Option<Solution> {
    let inv_transform: Option<_> = if NORMALIZE_INPUT && points.len() > 1 {
        let inv_transform = get_transform(points).expect("Internal error");
        Some(inv_transform)
    } else {
        None
    };
    let transform_back_or_dont = |sol: Solution| {
        if let Some((transf, _)) = inv_transform {
            (transf)(sol)
        } else {
            sol
        }
    };
    match trivial_solution_or_slope(points) {
        None => None,
        Some(TrivialSolutionOrSlope::ProblemTrivial(sol)) => Some(transform_back_or_dont(sol)),
        Some(TrivialSolutionOrSlope::Slope(starting_slope)) => {
            let max_steps = 15 * (points.len().ilog10() as usize) + 300;
            PalbGen::new(
                starting_slope,
                points,
                Uncertainty::default(),
                DoubleIntervalSize,
            )
            .take_until(|obs_state| {
                (obs_state.options[0].slope - obs_state.options[1].slope).abs()
                    < Floating::from(1e-15)
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
            .map(transform_back_or_dont)
        }
    }
}

/// A small wrapper around [PalbGen] that allows efficiently processing a sequence of
/// points in a sliding window fashing (including warmstarting from one window to the next).
/// Note that the sliding windows are more of an example at this point: the implementation of this struct
/// and in particular its Iterator implementation
/// can be considered as a prototypical example of other warmstarting strategies for more general problems.
pub struct WindowedPalb<'a, const NORMALIZE_INPUT: bool> {
    points: &'a mut [PrimalPoint],
    window_size: usize,
    current_start: usize,
    max_steps: usize,

    starting_slope: Floating,
    window_points: Vec<PrimalPoint>,
    line_val_buf: Vec<(DualLine, Floating)>,

    inv_transform: Option<Box<dyn Fn(Solution) -> Solution + 'static>>,
}

impl<'a, const NORMALIZE_INPUT: bool> WindowedPalb<'a, NORMALIZE_INPUT> {
    pub fn new(
        points: &'a mut [PrimalPoint],
        window_size: usize,
        starting_slope: Option<Floating>,
    ) -> Option<Self> {
        if points.len() < window_size || window_size < 2 {
            // in this case no solutions exist or they are all trivial.
            // Handling the trivial case in the following gets a bit annoying so we just bail out
            return None;
        }
        let inv_transform: Option<_> = if NORMALIZE_INPUT && points.len() > 1 {
            let (inv_transform, _) = get_transform(points).expect("Internal error");
            let it: Box<dyn Fn(Solution) -> Solution + 'static> = Box::new(inv_transform);
            Some(it)
        } else {
            None
        };
        let max_steps = 15 * (window_size.ilog10() as usize) + 300;
        let starting_slope = starting_slope.unwrap_or_else(|| {
            match trivial_solution_or_slope(&points[..window_size]) {
                Some(TrivialSolutionOrSlope::Slope(starting_slope)) => starting_slope,
                Some(TrivialSolutionOrSlope::ProblemTrivial(_)) | None => unreachable!(),
            }
        });
        Some(Self {
            points,
            window_size,
            max_steps,
            starting_slope,
            current_start: 0,
            window_points: vec![Default::default(); window_size],
            line_val_buf: vec![Default::default(); window_size],
            inv_transform,
        })
    }
}

impl<'a, const NORMALIZE_INPUT: bool> Iterator for WindowedPalb<'a, NORMALIZE_INPUT> {
    type Item = Solution;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_start + self.window_size > self.points.len() {
            return None;
        }
        // make a solver, then run the solver to completion.
        self.window_points.clear();
        self.window_points.extend_from_slice(
            &self.points[self.current_start..self.current_start + self.window_size],
        );
        // update for next iteration
        self.current_start += 1;
        let solver = PalbGen::new_with_val_buf(
            self.starting_slope,
            &mut self.window_points,
            &mut self.line_val_buf,
            Uncertainty::default(),
            DoubleIntervalSize,
        )
        .expect("Created buffer was too small");

        // we now do the same thing as in the non-windowed case
        let sol = solver
            .take_until(|obs_state| {
                (obs_state.options[0].slope - obs_state.options[1].slope).abs()
                    < Floating::from(1e-15)
            }) // stop iteration once the interval gets *tiny* (if that ever happens)
            .take(self.max_steps) // at most this many iterations, then we bail out
            .last()
            .map(|obs_state| {
                (
                    obs_state
                        .options
                        .into_iter()
                        .min_by_key(|state| {
                            state.get_or_compute_obj_val_noncached(self.window_points.as_ref())
                        })
                        .unwrap(),
                    obs_state.info,
                )
            })
            .map(|(state, info)| Solution {
                optimal_line: state.line_estimate,
                objective_value: state.obj_val.unwrap_or_else(|| {
                    state.get_or_compute_obj_val_noncached(self.window_points.as_ref())
                }),
                info,
            });
        match sol {
            Some(sol) => {
                self.starting_slope = sol.optimal_line.slope();
                if let Some(transf) = &self.inv_transform {
                    Some((transf)(sol))
                } else {
                    Some(sol)
                }
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests_bisect {

    use crate::l1line;
    use crate::l1line_with_initial_guess;
    use crate::objective_value;

    use super::{Floating, PrimalLine, PrimalPoint};
    use approx::{assert_abs_diff_eq, relative_eq};
    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
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
        let mut points2 = points.clone();
        let res = l1line(&mut points2).unwrap();
        assert_solution_likely_correct(res, ground_truth, &points);
    }

    #[test]
    fn works_med() {
        //INIT.call_once(|| pretty_env_logger::init());
        let ground_truth = PrimalLine {
            coords: (Floating::from(-3.0), Floating::from(-2.0)),
        };
        let points = generate_random_points(100, 0, ground_truth);
        let mut points2 = points.clone();
        let res = l1line(&mut points2).unwrap();
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

            let mut points2 = points.clone();
            let res = l1line(&mut points2).unwrap();
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

        let mut points2 = points.clone();
        let res = l1line(&mut points2).unwrap();
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
        let mut points2 = points.clone();
        let res = l1line(&mut points2).unwrap();
        assert_solution_likely_correct(res, ground_truth, &points);
    }

    #[test]
    fn works_particular3() {
        let z = Floating::from(0.0);
        let o = Floating::from(1.0);
        let points = vec![
            PrimalPoint::new(-o, z),
            PrimalPoint::new(o, o + o),
            PrimalPoint::new(o, -(o + o)),
        ];
        let mut points2 = points.clone();
        for initial_slope in [-o, z, o] {
            let res =
                l1line_with_initial_guess::<true>(&mut points2, Some(initial_slope), None).unwrap();
            println!("res = {:?}", &res.optimal_line.slope());
            println!("\n");
        }
        // panic!()
    }

    #[test]
    fn works_particular4() {
        let z = Floating::from(0.0);
        let o = Floating::from(1.0);
        let points = vec![
            PrimalPoint::new(z, z),
            PrimalPoint::new(o, z),
            PrimalPoint::new(Floating::from(1e-12), o),
        ];
        let mut points2 = points.clone();
        let res = l1line(&mut points2).unwrap();
        println!("res = {:?}", res);
        // std::hint::black_box(res);
        // panic!()
    }
}
