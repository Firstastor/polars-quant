use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::talib::overlap::calc_ema;

// ====================================================================
// Volume Indicators - 成交量指标 (Alphabetical Order)
// ====================================================================

#[derive(Deserialize)]
struct AdoscKwargs {
    fastperiod: Option<usize>,
    slowperiod: Option<usize>,
}

#[polars_expr(output_type=Float64)]
pub fn ad(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;
    let volume = inputs[3].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;
    let volume = volume.f64()?;

    Ok(calc_ad(high, low, close, volume).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn adosc(inputs: &[Series], kwargs: AdoscKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;
    let volume = inputs[3].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;
    let volume = volume.f64()?;

    let ad = calc_ad(high, low, close, volume);
    
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adl".into(), ad.len());
    let mut sum = 0.0f64;
    
    ad.downcast_iter().for_each(|ad_array| {
        ad_array.iter().for_each(|ad_value| {
            if let Some(ad_value) = ad_value {
                sum += ad_value;
                builder.append_value(sum);
            } else {
                builder.append_null();
            }
        });
    });
    
    let adl = builder.finish();
    let fastperiod = kwargs.fastperiod.unwrap_or(3);
    let slowperiod = kwargs.slowperiod.unwrap_or(10);
    
    Ok((calc_ema(&adl, fastperiod) - calc_ema(&adl, slowperiod)).into_series())
    
}

#[polars_expr(output_type=Float64)]
pub fn obv(inputs: &[Series]) -> PolarsResult<Series> {
    let close = inputs[0].cast(&DataType::Float64)?;
    let volume = inputs[1].cast(&DataType::Float64)?;

    let close = close.f64()?;
    let volume = volume.f64()?;
    
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("obv".into(), close.len());
    let close_diff = &close.shift(1 as i64) - close;
    let mut sum = 0.0f64;
    
    izip!(&close_diff, volume).for_each(|(c_diff, v)| match (c_diff, v) {
        (Some(c_diff), Some(v)) => {
            if c_diff > 0.0 {
                sum += v;
            } else if c_diff < 0.0 {
                sum -= v;
            }
            builder.append_value(sum);
        }
        _ => builder.append_null(),
    });
    
    Ok(builder.finish().into_series())
}

// ====================================================================
// Calculation Helpers
// ====================================================================

fn calc_ad(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    volume: &Float64Chunked,
) -> Float64Chunked {
    let n = high.len();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ad".into(), n);

    let mut sum = 0.0f64;

    izip!(high, low, close, volume).for_each(|(h, l, c, v)| match (h, l, c, v) {
        (Some(h), Some(l), Some(c), Some(v)) => {
            let diff = h - l;
            if diff == 0.0 {
                builder.append_value(0.0);
            } else {
                sum += (2.0 * c - l - h) / diff * v;
                builder.append_value(sum);
            }
        }
        _ => builder.append_null(),
    });

    builder.finish()
}
