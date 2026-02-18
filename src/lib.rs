use pyo3::prelude::*;

#[pymodule]
fn polars_quant(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
