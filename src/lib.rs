use pyo3::prelude::*;

mod talib;

#[pymodule]
fn polars_quant(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    talib::register_submodule(m)?;
    Ok(())
}
