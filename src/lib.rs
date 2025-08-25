use pyo3::prelude::*;

mod qbacktrade;
mod qstock;
mod qtalib;

#[pymodule]
fn polars_quant(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<qbacktrade::Backtrade>()?;
    m.add_function(wrap_pyfunction!(qstock::info, m)?)?;
    m.add_function(wrap_pyfunction!(qstock::info_save, m)?)?;
    m.add_function(wrap_pyfunction!(qstock::history, m)?)?;
    m.add_function(wrap_pyfunction!(qstock::history_save, m)?)?;

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

    m.add_function(wrap_pyfunction!(qtalib::adx, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::adxr, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::apo, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::aroon, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::aroonosc, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::bop, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::cci, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::cmo, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::dx, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::macd, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::mfi, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::mom, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::ppo, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::roc, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::rsi, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::stoch, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::stochf, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::stochrsi, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::trix, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::ultosc, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::willr, m)?)?;

    m.add_function(wrap_pyfunction!(qtalib::ad, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::adosc, m)?)?;
    m.add_function(wrap_pyfunction!(qtalib::obv, m)?)?;
    Ok(())
}

