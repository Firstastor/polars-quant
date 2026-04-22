use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::talib::overlap::calc_ema;

// ====================================================================
// Volatility Indicators - 波动率指标 (Alphabetical Order)
// ====================================================================

#[derive(Deserialize)]
struct AtrKwargs {
    timeperiod: Option<usize>,
}

#[polars_expr(output_type=Float64)]
pub fn atr(inputs: &[Series], kwargs: AtrKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let trange = calc_trange(high, low, close);
    Ok(calc_ema(&trange, 2 * timeperiod - 1).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn natr(inputs: &[Series], kwargs: AtrKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let trange = calc_trange(high, low, close);
    let atr = calc_ema(&trange, 2 * timeperiod - 1);
    Ok((&atr / close * 100).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn trange(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    Ok(calc_trange(high, low, close).into_series())
}

// ====================================================================
// Calculation Helpers
// ====================================================================

fn calc_trange(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
) -> Float64Chunked {
    let pre_close = close.shift(1 as i64);
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("trange".into(), high.len());

    izip!(high, low, &pre_close).for_each(|(h, l, pc)| match (h, l, pc) {
        (Some(h), Some(l), Some(pc)) => {
            let tr = (h - l).max((h - pc).abs()).max((l - pc).abs());
            builder.append_value(tr);
        }
        _ => builder.append_null(),
    });

    builder.finish()
}
