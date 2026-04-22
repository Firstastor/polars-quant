use pyo3::prelude::*;

mod backtest;
mod talib;

#[pymodule]
fn polars_quant(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    backtest::register_submodule(m)?;
    talib::register_submodule(m)?;
    Ok(())
}
