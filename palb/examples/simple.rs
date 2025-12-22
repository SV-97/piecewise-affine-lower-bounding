use palb::{self, Floating, PrimalPoint};

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
    let res = palb::l1line(&points).unwrap();
    println!("The result is {:?}", res);
}
