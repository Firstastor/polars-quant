use itertools::izip;
use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_arrow::array::Array};
use serde::Deserialize;
use std::collections::VecDeque;

// ====================================================================
// Overlap Studies - 重叠指标 (Alphabetical Order)
// ====================================================================

#[derive(Deserialize)]
pub struct BbandsKwargs {
    pub timeperiod: usize,
    pub nbdevup: Option<f64>,
    pub nbdevdn: Option<f64>,
}

fn bbands_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("bb_upper".into(), DataType::Float64);
    let f2 = Field::new("bb_middle".into(), DataType::Float64);
    let f3 = Field::new("bb_lower".into(), DataType::Float64);
    Ok(Field::new(
        "bbands".into(),
        DataType::Struct(vec![f1, f2, f3]),
    ))
}

fn mama_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("mama".into(), DataType::Float64);
    let f2 = Field::new("fama".into(), DataType::Float64);
    Ok(Field::new("mama".into(), DataType::Struct(vec![f1, f2])))
}

#[polars_expr(output_type_func=bbands_output)]
pub fn bbands(inputs: &[Series], kwargs: BbandsKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod;
    let nbdevup = kwargs.nbdevup.unwrap_or(2.0);
    let nbdevdn = kwargs.nbdevdn.unwrap_or(2.0);

    let n = real.len();

    if timeperiod == 0 || n < timeperiod {
        let null_ca = Float64Chunked::full_null("".into(), n);
        let s1 = null_ca.clone().with_name("bb_upper".into()).into_series();
        let s2 = null_ca.clone().with_name("bb_middle".into()).into_series();
        let s3 = null_ca.with_name("bb_lower".into()).into_series();
        return Ok(
            StructChunked::from_series("bbands".into(), n, [s1, s2, s3].iter())?.into_series(),
        );
    }

    let mut upper_b = PrimitiveChunkedBuilder::<Float64Type>::new("bb_upper".into(), n);
    let mut middle_b = PrimitiveChunkedBuilder::<Float64Type>::new("bb_middle".into(), n);
    let mut lower_b = PrimitiveChunkedBuilder::<Float64Type>::new("bb_lower".into(), n);

    let mut count: usize = 0;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    for array in real.downcast_iter() {
        for i in 0..array.len() {
            if array.is_null(i) {
                upper_b.append_null();
                middle_b.append_null();
                lower_b.append_null();
                continue;
            }
            let value = array.value(i);
            count += 1;
            sum += value;
            sum_sq += value * value;
            window.push_back(value);


            if count < timeperiod {
                upper_b.append_null();
                middle_b.append_null();
                lower_b.append_null();
            } else {
                if count > timeperiod {
                    if let Some(old_v) = window.pop_front() {
                        sum -= old_v;
                        sum_sq -= old_v * old_v;
                        count -= 1;
                    }
                }
                let mean = sum / timeperiod as f64;
                let variance = (sum_sq / timeperiod as f64) - mean * mean;
                let std = variance.max(0.0).sqrt();
                upper_b.append_value(mean + nbdevup * std);
                middle_b.append_value(mean);
                lower_b.append_value(mean - nbdevdn * std);
            }
        }
    }

    let s1 = upper_b.finish().into_series();
    let s2 = middle_b.finish().into_series();
    let s3 = lower_b.finish().into_series();

    Ok(StructChunked::from_series("bbands".into(), n, [s1, s2, s3].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn dema(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    Ok(calc_dema(real, timeperiod))
}

#[polars_expr(output_type=Float64)]
pub fn ema(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    Ok(calc_ema(real, timeperiod))
}

#[polars_expr(output_type=Float64)]
pub fn kama(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    Ok(calc_kama(real, timeperiod))
}

#[polars_expr(output_type=Float64)]
pub fn ma(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;
    let matype = inputs[2].i64()?.get(0).unwrap_or(0);

    let mut buf: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf.push(0.0);
            } else {
                buf.push(arr.value(i));
            }
        }
    }

    let res = calc_ma(&buf, timeperiod, matype);
    Ok(res.into_series())
}

#[polars_expr(output_type_func=mama_output)]
pub fn mama(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let fastlimit = inputs[1].f64()?.get(0).unwrap_or(0.0);
    let slowlimit = inputs[2].f64()?.get(0).unwrap_or(0.0);

    let mut buf: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf.push(0.0);
            } else {
                buf.push(arr.value(i));
            }
        }
    }

    let (mama_ca, fama_ca) = calc_mama(&buf, fastlimit, slowlimit);
    let s1 = mama_ca.into_series();
    let s2 = fama_ca.into_series();
    Ok(StructChunked::from_series("mama".into(), s1.len(), [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn midpoint(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(14) as usize;

    let mut buf: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf.push(0.0);
            } else {
                buf.push(arr.value(i));
            }
        }
    }

    let res = calc_midpoint(&buf, timeperiod);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn midprice(inputs: &[Series]) -> PolarsResult<Series> {
    let high_ca = inputs[0].cast(&DataType::Float64)?;
    let low_ca = inputs[1].cast(&DataType::Float64)?;
    let high = high_ca.f64()?;
    let low = low_ca.f64()?;
    let timeperiod = inputs[2].i64()?.get(0).unwrap_or(14) as usize;

    let mut buf_high: Vec<f64> = Vec::with_capacity(high.len());
    let mut buf_low: Vec<f64> = Vec::with_capacity(low.len());
    for (h_opt, l_opt) in high.into_iter().zip(low.into_iter()) {
        buf_high.push(h_opt.unwrap_or(0.0));
        buf_low.push(l_opt.unwrap_or(0.0));
    }

    let res = calc_midprice(&buf_high, &buf_low, timeperiod);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn mavp(inputs: &[Series]) -> PolarsResult<Series> {
    let real_ca = inputs[0].cast(&DataType::Float64)?;
    let periods_ca = inputs[1].cast(&DataType::Int64)?;
    let real = real_ca.f64()?;
    let periods = periods_ca.i64()?;
    let minperiod = inputs[2].i64()?.get(0).unwrap_or(2) as usize;
    let maxperiod = inputs[3].i64()?.get(0).unwrap_or(30) as usize;
    let matype = inputs[4].i64()?.get(0).unwrap_or(0);

    let mut buf_real: Vec<f64> = Vec::with_capacity(real.len());
    for arr in real.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf_real.push(0.0);
            } else {
                buf_real.push(arr.value(i));
            }
        }
    }

    let mut buf_periods: Vec<i64> = Vec::with_capacity(periods.len());
    for p in periods.into_iter() {
        buf_periods.push(p.unwrap_or(0));
    }

    let res = calc_mavp(&buf_real, &buf_periods, minperiod, maxperiod, matype);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn sar(inputs: &[Series]) -> PolarsResult<Series> {
    let high_ca = inputs[0].cast(&DataType::Float64)?;
    let low_ca = inputs[1].cast(&DataType::Float64)?;
    let high = high_ca.f64()?;
    let low = low_ca.f64()?;
    let acceleration = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let maximum = inputs[3].f64()?.get(0).unwrap_or(0.0);

    let mut buf_high: Vec<f64> = Vec::with_capacity(high.len());
    let mut buf_low: Vec<f64> = Vec::with_capacity(low.len());
    for (h_opt, l_opt) in high.into_iter().zip(low.into_iter()) {
        buf_high.push(h_opt.unwrap_or(0.0));
        buf_low.push(l_opt.unwrap_or(0.0));
    }

    let res = calc_sar(&buf_high, &buf_low, acceleration, maximum);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn sarext(inputs: &[Series]) -> PolarsResult<Series> {
    let high_ca = inputs[0].cast(&DataType::Float64)?;
    let low_ca = inputs[1].cast(&DataType::Float64)?;
    let high = high_ca.f64()?;
    let low = low_ca.f64()?;
    let startvalue = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let offsetonreverse = inputs[3].f64()?.get(0).unwrap_or(0.0);
    let accelerationinitlong = inputs[4].f64()?.get(0).unwrap_or(0.0);
    let accelerationlong = inputs[5].f64()?.get(0).unwrap_or(0.0);
    let accelerationmaxlong = inputs[6].f64()?.get(0).unwrap_or(0.0);
    let accelerationinitshort = inputs[7].f64()?.get(0).unwrap_or(0.0);
    let accelerationshort = inputs[8].f64()?.get(0).unwrap_or(0.0);
    let accelerationmaxshort = inputs[9].f64()?.get(0).unwrap_or(0.0);

    let mut buf_high: Vec<f64> = Vec::with_capacity(high.len());
    let mut buf_low: Vec<f64> = Vec::with_capacity(low.len());
    for (h_opt, l_opt) in high.into_iter().zip(low.into_iter()) {
        buf_high.push(h_opt.unwrap_or(0.0));
        buf_low.push(l_opt.unwrap_or(0.0));
    }

    let res = calc_sarext(
        &buf_high,
        &buf_low,
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort,
    );
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn sma(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    Ok(calc_sma(real, timeperiod))
}

#[polars_expr(output_type=Float64)]
pub fn t3(inputs: &[Series]) -> PlarsResult<Series> {
    let ca = inputs[0].cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(5) as usize;
    let vfactor = inputs[2].f64()?.get(0).unwrap_or(0.7);

    let mut buf: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf.push(0.0);
            } else {
                buf.push(arr.value(i));
            }
        }
    }

    let res = calc_t3(&buf, timeperiod, vfactor);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn tema(inputs: &[Series]) -> PlarsResult<Series> {
    let ca = inputs[0].cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    let mut buf: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf.push(0.0);
            } else {
                buf.push(arr.value(i));
            }
        }
    }

    let res = calc_tema(&buf, timeperiod);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn trima(inputs: &[Series]) -> PlarsResult<Series> {
    let ca = inputs[0].cast(&DataType::Float64)?;
    let ca = ca.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    let mut buf: Vec<f64> = Vec::with_capacity(ca.len());
    for arr in ca.downcast_iter() {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                buf.push(0.0);
            } else {
                buf.push(arr.value(i));
            }
        }
    }

    let res = calc_trima(&buf, timeperiod);
    Ok(res.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn wma(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    Ok(calc_wma(real, timeperiod))
}

// ====================================================================
// Calculation Helpers
// ====================================================================

pub fn calc_dema(value: &ChunkedArray<Float64Type>, timeperiod: usize) -> Series {
    let ema1 = calc_ema(value, timeperiod);
    let ema2 = calc_ema(ema1.f64().unwrap(), timeperiod);
    let ema1 = ema1.f64().unwrap();
    let ema2 = ema2.f64().unwrap();

    let n = ema1.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("dema".into(), n);

    for (ema1_value, ema2_value) in izip!(ema1, ema2) {
        match (ema1_value, ema2_value) {
            (Some(e1), Some(e2)) => builder.append_value(2.0 * e1 - e2),
            _ => builder.append_null(),
        }
    }

    builder.finish().into_series()
}

pub fn calc_ema(value: &ChunkedArray<Float64Type>, timeperiod: usize) -> Series {
    let n = value.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("ema".into(), n).into_series();
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ema".into(), n);

    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    let mut count: usize = 0;
    let mut ema_value = 0.0f64;
    let mut sum = 0.0f64;

    value.downcast_iter().for_each(|array| {
        for i in 0..array.len() {
            if array.is_null(i) {
                builder.append_null();
                continue;
            }
            let value = array.value(i);
            if count < timeperiod {
                sum += value;
                count += 1;
                if count == timeperiod {
                    ema_value = sum / timeperiod as f64;
                    builder.append_value(ema_value);
                } else {
                    builder.append_null();
                }
            } else {
                ema_value = alpha.mul_add(value - ema_value, ema_value);
                builder.append_value(ema_value);
            }
        }
    });

    builder.finish().into_series()
}

pub fn calc_sma(values: &ChunkedArray<Float64Type>, timeperiod: usize) -> Series {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("sma".into(), n).into_series();
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sma".into(), n);

    let denominator = 1.0 / timeperiod as f64;

    let mut count: usize = 0;
    let mut sum = 0.0f64;
    let mut window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    values.downcast_iter().for_each(|array| {
        for i in 0..array.len() {
            if array.is_null(i) {
                builder.append_null();
                continue;
            }

            let value = array.value(i);
            count += 1;
            window.push_back(value);
            sum += value;

            if count < timeperiod {
                builder.append_null();
            } else {
                if count > timeperiod {
                    if let Some(old_value) = window.pop_front() {
                        sum -= old_value;
                        count -= 1;
                    }
                }
                builder.append_value(sum * denominator);
            }
        }
    });

    builder.finish().into_series()
}

pub fn calc_wma(values: &ChunkedArray<Float64Type>, timeperiod: usize) -> Series {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("sma".into(), n).into_series();
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sma".into(), n);

    let mut count = 0;
    let denominator: f64 = (timeperiod * (timeperiod + 1) / 2) as f64;
    let mut numerator: f64 = 0.0;
    let mut sum = 0.0f64;
    let mut window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    values.downcast_iter().for_each(|array| {
        for i in 0..array.len() {
            if array.is_null(i) {
                builder.append_null();
                continue;
            }

            let value = array.value(i);
            count += 1;
            sum += value;
            window.push_back(value);

            if window.len() < timeperiod {
                builder.append_null();
                numerator += value * count as f64;
            } else {
                if window.len() > timeperiod {
                    if let Some(old_value) = window.pop_front() {
                        numerator += value * timeperiod as f64 - sum;
                        sum -= old_value;
                    }
                }
                builder.append_value(numerator / denominator);
            }
        }
    });

    builder.finish().into_series()
}
