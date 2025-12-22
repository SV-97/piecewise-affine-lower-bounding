use std::borrow::Cow;

use palb::{self, DoubleIntervalSize, Floating, PrimalPoint, Uncertainty};
use take_until::TakeUntilExt;

/// Generates some sample points in the plane
fn sample_data() -> Vec<PrimalPoint> {
    let xs = [
        -1.,
        -0.77777778,
        -0.55555556,
        -0.33333333,
        -0.11111111,
        0.11111111,
        0.33333333,
        0.55555556,
        0.77777778,
        1.,
    ];
    let ys = [
        2.54318089,
        0.72247295,
        0.36083089,
        1.24960558,
        -0.99193818,
        0.14858973,
        -2.13678337,
        -0.94156457,
        -4.3048653,
        -6.3074208,
    ];
    xs.into_iter()
        .zip(ys.into_iter())
        .map(|(x, y)| PrimalPoint {
            coords: (Floating::from(x), Floating::from(y)),
        })
        .collect()
}

fn main() {
    let points = sample_data();
    // Alternatively to the higher level l1line function (and variants) you can also use
    // the lower-level iterator interface if you need / want more control.
    // Note that this does *not* automatically normalize the data in the same way the higher level functions do.

    // compute the least squares solution as initial guess (you can do whatever you want here. Just don't start from zero)
    let initial_guess = palb::least_squares_slope(&points).unwrap();
    // construct the new generator starting at that initial guess using a default uncertainty for this initial guess
    let res = palb::PalpGen::new(
        initial_guess,
        Cow::Owned(points),
        Uncertainty::default(),
        DoubleIntervalSize, // stepsize rule
    )
    // you can use some early stopping condition here, we'll just accept everything
    .take_until(|_obs_state| false)
    // we want to do at most 30 iterations
    .take(30)
    // and we want the output to be the final state
    .last()
    // we know this doesn't fail :)
    .unwrap();

    if res.options[0].is_stationary() {
        println!(
            "Final state is stationary. Solution is {:?}",
            res.options[0].line_estimate
        );
    } else {
        println!(
            "Final state is not stationary. Last estimate is {:?}",
            res.options[0].line_estimate
        );
    }
}
