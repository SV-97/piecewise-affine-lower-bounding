use numpy::{borrow::PyReadonlyArray2, PyReadonlyArray1};
use palb;
use palb::Floating;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass]
#[doc = "Result of an L1 regression.\n\n\
Contains the slope, intercept and the objective value of the fitted line."]
pub struct RegressionResult {
    #[pyo3(get)]
    pub slope: f64,
    #[pyo3(get)]
    pub intercept: f64,
    #[pyo3(get)]
    pub objective_value: f64,
}

#[pymethods]
impl RegressionResult {
    #[new]
    #[doc = "Create a RegressionResult object.\n\n\
`slope`: float — slope of the line\n\
`intercept`: float — intercept of the line\n\
`objective_value`: float — sum |yi − (slope·xi + intercept)|"]
    fn new(slope: f64, intercept: f64, objective_value: f64) -> Self {
        Self {
            slope,
            intercept,
            objective_value,
        }
    }

    #[doc = "Return the (slope, intercept) pair as a tuple."]
    fn to_slope_intercept(&self) -> (f64, f64) {
        (self.slope, self.intercept)
    }

    fn __repr__(&self) -> String {
        format!(
            "RegressionResult(slope={}, intercept={}, objective_value={})",
            self.slope, self.intercept, self.objective_value
        )
    }
}

impl RegressionResult {
    fn from_solution(res: &palb::Solution) -> Self {
        Self {
            slope: *res.optimal_line.slope(),
            intercept: *res.optimal_line.intercept(),
            objective_value: *res.objective_value,
        }
    }
}

/// Fit a least-absolute deviation (L1) line to a set of N points given as a 2D array of shape (N, 2).
#[pyfunction]
#[pyo3(signature = (points, normalize_input=true))]
fn l1line<'py>(
    points: PyReadonlyArray2<'py, f64>,
    normalize_input: bool,
) -> PyResult<RegressionResult> {
    let arr = points.as_array();
    if arr.ncols() != 2 {
        if arr.nrows() == 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "Input array must have 2 columns. Your array has 2 rows, perhaps you meant to transpose it?",
            ));
        } else {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Input array must have exactly 2 columns, yours has shape {:?}",
                arr.shape()
            )));
        }
    } else {
        let wrapped_points = arr
            .rows()
            .into_iter()
            .map(|row| palb::PrimalPoint::new(Floating::from(row[0]), Floating::from(row[1])))
            .collect::<Vec<_>>();
        (if normalize_input {
            palb::l1line_with_info::<true>
        } else {
            palb::l1line_with_info::<false>
        })(&wrapped_points)
        .map(|res| RegressionResult::from_solution(&res))
        .ok_or_else(|| PyErr::new::<PyValueError, _>("Failed to compute L1 line"))
    }
}

/// Similar to [l1line] but takes two separate arrays for the x and y values.
#[pyfunction]
#[pyo3(signature = (xs, ys, normalize_input=true))]
pub fn l1line_xy<'py>(
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    normalize_input: bool,
) -> PyResult<RegressionResult> {
    let x_arr = xs.as_array();
    let y_arr = ys.as_array();

    if x_arr.len() != y_arr.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "Input arrays must have the same number of elements",
        ));
    }
    let wrapped_points = x_arr
        .into_iter()
        .zip(y_arr.into_iter())
        .map(|(x, y)| palb::PrimalPoint::new(Floating::from(*x), Floating::from(*y)))
        .collect::<Vec<_>>();
    (if normalize_input {
        palb::l1line_with_info::<true>
    } else {
        palb::l1line_with_info::<false>
    })(&wrapped_points)
    .map(|res| RegressionResult::from_solution(&res))
    .ok_or_else(|| PyErr::new::<PyValueError, _>("Failed to compute L1 line"))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RegressionResult>()?;
    m.add_function(wrap_pyfunction!(l1line, m)?)?;
    m.add_function(wrap_pyfunction!(l1line_xy, m)?)?;
    Ok(())
}
