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

    Ok((&calc_atr(high, low, close, timeperiod) / close * 100).into_series())
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

pub fn calc_atr(
    high: &Float64Chunked,
    low: &Float64Chunked,
    close: &Float64Chunked,
    timeperiod: usize,
) -> Float64Chunked {
    let n = high.len();
    if n < timeperiod
        || low.len() != n
        || close.len() != n
        || high.null_count() + low.null_count() + close.null_count() > 0
    {
        return Float64Chunked::full_null("atr".into(), n);
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("atr".into(), high.len());

    let alpha = 1.0 / timeperiod as f64;
    let mut value_iter = izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    );

    let (_, _, mut c_pre_val) = value_iter.next().unwrap();
    let mut sum_ad: f64 = 0.;
    builder.append_null();

    for (h_val, l_val, c_val) in value_iter.by_ref().take(timeperiod - 1) {
        if c_pre_val > h_val {
            sum_ad += c_pre_val - l_val;
        } else if c_pre_val < l_val {
            sum_ad += h_val - c_pre_val;
        } else {
            sum_ad += h_val - l_val;
        }
        c_pre_val = c_val;
        
        builder.append_null();
    }

    let (h_val, l_val, c_val) = value_iter.next().unwrap();
    if c_pre_val > h_val {
        sum_ad += c_pre_val - l_val;
    } else if c_pre_val < l_val {
        sum_ad += h_val - c_pre_val;
    } else {
        sum_ad += h_val - l_val;
    }
    c_pre_val = c_val;
    sum_ad = sum_ad / timeperiod as f64;
    builder.append_value(sum_ad);

    for (h_val, l_val, c_val) in value_iter {
        sum_ad = alpha.mul_add(
            if c_pre_val > h_val {
                c_pre_val - l_val
            } else if c_pre_val < l_val {
                h_val - c_pre_val
            } else {
                h_val - l_val
            } - sum_ad,
            sum_ad,
        );
        c_pre_val = c_val;

        builder.append_value(sum_ad);
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
        || high.null_count() + low.null_count() + close.null_count() > 0
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

    let (_, _, mut c_pre_val) = value_iter.next().unwrap();
    builder.append_null();

    for (h_val, l_val, c_val) in value_iter {
        if c_pre_val > h_val {
            builder.append_value(c_pre_val - l_val);
        } else if c_pre_val < l_val {
            builder.append_value(h_val - c_pre_val);
        } else {
            builder.append_value(h_val - l_val);
        }
        c_pre_val = c_val;
    }

    builder.finish()
}
