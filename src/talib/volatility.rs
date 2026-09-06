use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

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

    Ok(calc_atr(high, low, close, timeperiod).into_series())
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

    Ok(calc_natr(high, low, close, timeperiod).into_series())
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

pub(crate) fn calc_atr(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if n < timeperiod
        || low.len() != n
        || close.len() != n
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("atr".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("atr".into(), high.len());

    let alpha = 1.0 / timeperiod as f64;
    let mut atr = 0.;
    let mut sum: f64 = 0.;

    let mut value_iter = izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    );

    let (_, _, mut c_pre) = value_iter.next().unwrap();
    builder.append_null();

    for (i, (h, l, c)) in value_iter.enumerate() {
        let tr = if c_pre > h {
            c_pre - l
        } else if c_pre < l {
            h - c_pre
        } else {
            h - l
        };

        if i < timeperiod - 1 {
            sum += tr;
            builder.append_null();
        } else if i == timeperiod - 1 {
            sum += tr;
            atr = sum / timeperiod as f64;
            builder.append_value(atr);
        } else {
            atr += (tr - atr) * alpha;
            builder.append_value(atr);
        }

        c_pre = c;
    }

    builder.finish()
}

pub fn calc_trange(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
) -> Float64Chunked {
    let n = high.len();
    if low.len() != n
        || close.len() != n
        || high.null_count() + low.null_count() + close.null_count() != 0
    {
        return Float64Chunked::full_null("trange".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("trange".into(), high.len());

    let mut value_iter = izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    );

    let (_, _, mut c_pre) = value_iter.next().unwrap();
    builder.append_null();

    for (h, l, c) in value_iter {
        if c_pre > h {
            builder.append_value(c_pre - l);
        } else if c_pre < l {
            builder.append_value(h - c_pre);
        } else {
            builder.append_value(h - l);
        }
        c_pre = c;
    }

    builder.finish()
}

pub fn calc_natr(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = close.len();
    let atr = calc_atr(high, low, close, timeperiod);
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("natr".into(), n);

    for (a, c) in atr.iter().zip(close.iter()) {
        match (a, c) {
            (Some(a), Some(c)) => builder.append_value(a / c * 100.0),
            _ => builder.append_null(),
        }
    }

    builder.finish()
}
