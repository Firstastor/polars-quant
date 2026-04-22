use crate::talib::overlap::calc_rma;
use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// ====================================================================
// Volatility Indicators - 波动率指标 (Alphabetical Order)
// ====================================================================

#[polars_expr(output_type=Float64)]
pub fn atr(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let mut trange_filled: Vec<f64> = Vec::with_capacity(high.len());
    let mut prev_close: Option<f64> = None;

    for (h_opt, l_opt, c_opt) in izip!(high.into_iter(), low.into_iter(), close.into_iter()) {
        if prev_close.is_none() {
            let val = match (h_opt, l_opt) {
                (Some(h), Some(l)) => Some(h - l),
                _ => None,
            };
            trange_filled.push(val.unwrap_or(0.0));
        } else {
            let val = match (h_opt, l_opt, prev_close) {
                (Some(h), Some(l), Some(pc)) => {
                    Some((h - l).max((h - pc).abs()).max((l - pc).abs()))
                }
                _ => None,
            };
            trange_filled.push(val.unwrap_or(0.0));
        }
        prev_close = c_opt;
    }

    let rma_res = calc_rma(&trange_filled, timeperiod);

    let n = rma_res.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("atr".into(), n);
    for v in rma_res {
        if let Some(x) = v {
            builder.append_value(x);
        } else {
            builder.append_null();
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn natr(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let mut trange_filled: Vec<f64> = Vec::with_capacity(high.len());
    let mut prev_close: Option<f64> = None;

    for (h_opt, l_opt, c_opt) in izip!(high.into_iter(), low.into_iter(), close.into_iter()) {
        if prev_close.is_none() {
            let val = match (h_opt, l_opt) {
                (Some(h), Some(l)) => Some(h - l),
                _ => None,
            };
            trange_filled.push(val.unwrap_or(0.0));
        } else {
            let val = match (h_opt, l_opt, prev_close) {
                (Some(h), Some(l), Some(pc)) => {
                    Some((h - l).max((h - pc).abs()).max((l - pc).abs()))
                }
                _ => None,
            };
            trange_filled.push(val.unwrap_or(0.0));
        }
        prev_close = c_opt;
    }

    let atr = calc_rma(&trange_filled, timeperiod);

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("natr".into(), atr.len());
    for (atr_opt, c_opt) in izip!(atr.into_iter(), close.into_iter()) {
        match (atr_opt, c_opt) {
            (Some(a), Some(c)) if c != 0.0 => builder.append_value((a / c) * 100.0),
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn trange(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let n = high.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("trange".into(), n);
    let mut prev_close: Option<f64> = None;

    for (h_opt, l_opt, c_opt) in izip!(high.into_iter(), low.into_iter(), close.into_iter()) {
        let val = if prev_close.is_none() {
            match (h_opt, l_opt) {
                (Some(h), Some(l)) => Some(h - l),
                _ => None,
            }
        } else {
            match (h_opt, l_opt, prev_close) {
                (Some(h), Some(l), Some(pc)) => {
                    Some((h - l).max((h - pc).abs()).max((l - pc).abs()))
                }
                _ => None,
            }
        };

        if let Some(v) = val {
            builder.append_value(v);
        } else {
            builder.append_null();
        }

        prev_close = c_opt;
    }

    Ok(builder.finish().into_series())
}
