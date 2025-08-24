use pyo3::prelude::*;

mod qbacktrade;
mod qstock;
mod qtalib;

#[pymodule]
fn polars_quant(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<qbacktrade::Backtrade>()?;
    m.add_function(wrap_pyfunction!(qstock::history, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::bband, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::dema, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::ema, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::kama, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::ma, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::mama, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::mavp, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::sma, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::t3, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::tema, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::trima, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::wma, m)?)?;
    Ok(())
}
