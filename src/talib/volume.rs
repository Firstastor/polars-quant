use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

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

    let fastperiod = kwargs.fastperiod.unwrap_or(3);
    let slowperiod = kwargs.slowperiod.unwrap_or(10);

    Ok(calc_adosc(high, low, close, volume, fastperiod, slowperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn obv(inputs: &[Series]) -> PolarsResult<Series> {
    let close = inputs[0].cast(&DataType::Float64)?;
    let volume = inputs[1].cast(&DataType::Float64)?;

    let close = close.f64()?;
    let volume = volume.f64()?;

    let n = close.len();
    if volume.len() != n || close.null_count() + volume.null_count() > 0 {
        return Ok(Float64Chunked::full_null("obv".into(), n).into_series());
    }

    let close = close.rechunk();
    let volume = volume.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("obv".into(), close.len());

    let mut value_iter = izip!(close.no_null_iter(), volume.no_null_iter());

    let (mut c_pre, mut sum) = value_iter.next().unwrap();
    builder.append_value(sum);

    for (c, v) in value_iter {
        if c > c_pre {
            sum += v;
        } else if c < c_pre {
            sum -= v;
        }
        c_pre = c;

        builder.append_value(sum);
    }

    Ok(builder.finish().into_series())
}

fn calc_ad(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    volume: &Float64Chunked,
) -> Float64Chunked {
    let n = high.len();
    if low.len() != n
        || close.len() != n
        || volume.len() != n
        || high.null_count() + low.null_count() + close.null_count() + volume.null_count() > 0
    {
        return Float64Chunked::full_null("ad".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();
    let volume = volume.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ad".into(), n);

    let mut ad: f64 = 0.;

    for (h, l, c, v) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter(),
        volume.no_null_iter()
    ) {
        ad += calc_mfv(h, l, c, v);
        builder.append_value(ad);
    }

    builder.finish()
}

fn calc_adosc(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    volume: &Float64Chunked,
    fastperiod: usize,
    slowperiod: usize,
) -> Float64Chunked {
    let n = high.len();

    if fastperiod == 0
        || slowperiod == 0
        || fastperiod >= slowperiod
        || n < slowperiod
        || low.len() != n
        || close.len() != n
        || volume.len() != n
        || high.null_count() + low.null_count() + close.null_count() + volume.null_count() > 0
    {
        return Float64Chunked::full_null("adosc".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();
    let volume = volume.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("adosc".into(), n);

    let fast_alpha = 2.0 / (fastperiod as f64 + 1.0);
    let slow_alpha = 2.0 / (slowperiod as f64 + 1.0);
    let mut ad = 0.;
    let mut count = 0;
    let mut ema1 = 0.;
    let mut ema2 = 0.;

    for (h, l, c, v) in izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter(),
        volume.no_null_iter()
    ) {
        count += 1;

        ad += calc_mfv(h, l, c, v);
        
        if count == 1 {
            ema1 = ad;
            ema2 = ad;
            builder.append_null();
        } else if count < slowperiod {
            ema1 += fast_alpha * (ad - ema1);
            ema2 += slow_alpha * (ad - ema2);
            builder.append_null();
        } else {
            ema1 += fast_alpha * (ad - ema1);
            ema2 += slow_alpha * (ad - ema2);
            builder.append_value(ema1 -ema2);
        }
    }

    builder.finish()
}

#[inline(always)]
fn calc_mfv(high: f64, low: f64, close: f64, volume: f64) -> f64 {
    if high != low {
        return close.mul_add(2.0, -high - low) / (high - low) * volume;
    }
    0.
}
