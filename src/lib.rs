use pyo3::prelude::*;

mod qbacktrade;
mod qstock;

#[pymodule]
fn polars_quant(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<qbacktrade::Backtrade>()?;
    m.add_function(wrap_pyfunction!(qstock::history, m)?)?;
    Ok(())
}
