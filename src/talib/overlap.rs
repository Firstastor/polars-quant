use arrayvec::ArrayVec;
use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_arrow::array::Array};
use serde::Deserialize;
use std::collections::VecDeque;

// ====================================================================
// Overlap Studies - 重叠指标 (Alphabetical Order)
// ====================================================================

#[derive(Deserialize)]
pub struct BbandsKwargs {
    pub timeperiod: Option<usize>,
    pub nbdevup: Option<f64>,
    pub nbdevdn: Option<f64>,
}

#[derive(Deserialize)]
pub struct MaKwargs {
    pub timeperiod: Option<usize>,
    pub matype: Option<usize>,
}

#[derive(Deserialize)]
pub struct T3Kwargs {
    pub timeperiod: Option<usize>,
    pub vfactor: Option<f64>,
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
    let timeperiod = kwargs.timeperiod.unwrap_or(20);
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
pub fn dema(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_dema(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ema(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_ema(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn kama(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_kama(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ma(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);
    let matype = kwargs.matype.unwrap_or(0);

    Ok(calc_ma(real, timeperiod, matype).into_series())
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
pub fn midpoint(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("midpoint".into(), real.len());

    let mut count = 0;
    let mut max = 0.0f64;
    let mut min = 0.0f64;
    let mut window_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod);
    let mut window_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod);
    real.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    while let Some(back) = window_max.back() {
                        if back.1 <= value {
                            window_max.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_max.front() {
                        if front.0 == count - timeperiod {
                            window_max.pop_front();
                        }
                    }
                    window_max.push_back((count, value));
                    max = window_max.front().unwrap().1;

                    while let Some(back) = window_min.back() {
                        if back.1 >= value {
                            window_min.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_max.front() {
                        if front.0 == count - timeperiod {
                            window_min.pop_front();
                        }
                    }
                    window_min.push_back((count, value));
                    min = window_min.front().unwrap().1;
                    builder.append_value((max + min) / 2.0);
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    while let Some(back) = window_max.back() {
                        if back.1 <= value {
                            window_max.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_max.front() {
                        if front.0 == count - timeperiod {
                            window_max.pop_front();
                        }
                    }
                    window_max.push_back((count, value));
                    max = window_max.front().unwrap().1;

                    while let Some(back) = window_min.back() {
                        if back.1 >= value {
                            window_min.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_max.front() {
                        if front.0 == count - timeperiod {
                            window_min.pop_front();
                        }
                    }
                    window_min.push_back((count, value));
                    min = window_min.front().unwrap().1;
                    builder.append_value((max + min) / 2.0);
                }
            }
        }
    });

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn midprice(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let mut high_builder =
        PrimitiveChunkedBuilder::<Float64Type>::new("high_max".into(), high.len());
    let mut low_builder = PrimitiveChunkedBuilder::<Float64Type>::new("low_min".into(), low.len());

    let mut count = 0;
    let mut window_max: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod);
    let mut window_min: VecDeque<(usize, f64)> = VecDeque::with_capacity(timeperiod);

    high.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        high_builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    while let Some(back) = window_max.back() {
                        if back.1 <= value {
                            window_max.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_max.front() {
                        if front.0 == count - timeperiod {
                            window_max.pop_front();
                        }
                    }
                    window_max.push_back((count, value));
                    high_builder.append_value(window_max.front().unwrap().1);
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    while let Some(back) = window_max.back() {
                        if back.1 <= value {
                            window_max.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_max.front() {
                        if front.0 == count - timeperiod {
                            window_max.pop_front();
                        }
                    }
                    window_max.push_back((count, value));
                    high_builder.append_value(window_max.front().unwrap().1);
                }
            }
        }
    });
    count = 0;
    low.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        high_builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    while let Some(back) = window_min.back() {
                        if back.1 <= value {
                            window_min.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_min.front() {
                        if front.0 == count - timeperiod {
                            window_min.pop_front();
                        }
                    }
                    window_min.push_back((count, value));
                    low_builder.append_value(window_min.front().unwrap().1);
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    while let Some(back) = window_min.back() {
                        if back.1 >= value {
                            window_min.pop_back();
                        } else {
                            break;
                        }
                    }
                    if let Some(front) = window_min.front() {
                        if front.0 == count - timeperiod {
                            window_min.pop_front();
                        }
                    }
                    window_min.push_back((count, value));
                    low_builder.append_value(window_min.front().unwrap().1);
                }
            }
        }
    });
    let mut midprice = (&high_builder.finish() + &low_builder.finish()) / 2.0;
    midprice.rename("midprice".into());
    Ok(midprice.into_series())
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
pub fn sma(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_sma(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn t3(inputs: &[Series], kwargs: T3Kwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(5);
    let vfactor = kwargs.vfactor.unwrap_or(0.0);

    Ok(calc_t3(real, timeperiod, vfactor).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn tema(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_tema(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn trima(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_trima(real, timeperiod).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn wma(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(30);

    Ok(calc_wma(real, timeperiod).into_series())
}

// ====================================================================
// Calculation Helpers
// ====================================================================

pub fn calc_dema(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < 2 * timeperiod - 1 {
        return Float64Chunked::full_null("dema".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("tema".into(), n);

    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    let mut count: usize = 0;
    let mut ema_value = ArrayVec::from([0.0f64; 2]);
    let mut sum = ArrayVec::from([0.0f64; 2]);

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum[0] += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum[0] += value;
                            ema_value[0] = sum[0] / timeperiod as f64;
                            sum[1] = ema_value[0];
                            builder.append_null();
                        }
                        n if n < 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            builder.append_null();
                        }
                        n if n == 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            ema_value[1] = sum[1] / timeperiod as f64;
                            builder.append_null();
                        }
                        _ => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            builder.append_value(2.0 * ema_value[0] - ema_value[1]);
                        }
                    }
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum[0] += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum[0] += value;
                            ema_value[0] = sum[0] / timeperiod as f64;
                            sum[1] = ema_value[0];
                            builder.append_null();
                        }
                        n if n < 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            builder.append_null();
                        }
                        n if n == 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            ema_value[1] = sum[1] / timeperiod as f64;
                            sum[2] = ema_value[1];
                            builder.append_null();
                        }
                        n if n < 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            builder.append_null();
                        }
                        n if n == 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            ema_value[2] = sum[2] / timeperiod as f64;
                            builder.append_value(
                                3.0 * ema_value[0] - 3.0 * ema_value[1] + ema_value[2],
                            );
                        }
                        _ => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            builder.append_value(
                                3.0 * ema_value[0] - 3.0 * ema_value[1] + ema_value[2],
                            );
                        }
                    }
                }
            }
        }
    });

    builder.finish()
}

pub fn calc_ema(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("ema".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ema".into(), n);

    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    let mut count: usize = 0;
    let mut ema_value = 0.0f64;
    let mut sum = 0.0f64;

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum += value;
                            ema_value = sum / timeperiod as f64;
                            builder.append_value(ema_value);
                        }
                        _ => {
                            ema_value = alpha.mul_add(value - ema_value, ema_value);
                            builder.append_value(ema_value);
                        }
                    }
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum += value;
                            ema_value = sum / timeperiod as f64;
                            builder.append_value(ema_value);
                        }
                        _ => {
                            ema_value = alpha.mul_add(value - ema_value, ema_value);
                            builder.append_value(ema_value);
                        }
                    }
                }
            }
        }
    });

    builder.finish()
}

pub fn calc_kama(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("kama".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("er".into(), n);

    let mut count = 0;
    let mut diff_abs = 0.0f64;
    let mut sum = 0.0f64;
    let mut window = VecDeque::with_capacity(timeperiod);
    let mut window_sum = VecDeque::with_capacity(timeperiod);

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }

                    let value = array_slice[i];
                    match count {
                        n if n == 0 => {
                            count += 1;
                            window.push_back(value);
                            builder.append_null();
                        }
                        n if n < timeperiod => {
                            count += 1;
                            diff_abs = (value - window.front().unwrap()).abs();
                            sum += diff_abs;
                            window.push_back(value);
                            window_sum.push_back(diff_abs);
                            builder.append_null();
                        }
                        _ => {
                            diff_abs = (value - window.pop_front().unwrap()).abs();
                            sum += diff_abs - window_sum.pop_front().unwrap();
                            window.push_back(value);
                            window_sum.push_back(diff_abs);
                            builder.append_value(diff_abs / sum);
                        }
                    }
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    match count {
                        n if n == 0 => {
                            count += 1;
                            window.push_back(value);
                            builder.append_null();
                        }
                        n if n < timeperiod => {
                            count += 1;
                            diff_abs = (value - window.front().unwrap()).abs();
                            sum += diff_abs;
                            window.push_back(value);
                            window_sum.push_back(diff_abs);
                            builder.append_null();
                        }
                        _ => {
                            diff_abs = (value - window.pop_front().unwrap()).abs();
                            sum += diff_abs - window_sum.pop_front().unwrap();
                            window.push_back(value);
                            window_sum.push_back(diff_abs);
                            builder.append_value(diff_abs / sum);
                        }
                    }
                }
            }
        }
    });

    let er = builder.finish();

    let fast_sc = 2.0 / 3.0;
    let slow_sc = 2.0 / 31.0;

    let sc_sqrt = er * (fast_sc - slow_sc) + slow_sc;
    let sc = &sc_sqrt * &sc_sqrt;

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("kama".into(), n);

    let mut count = 0;
    let mut kama_value = 0.0f64;
    let mut sum = 0.0f64;
    let values = values.cont_slice().unwrap();
    sc.downcast_iter().for_each(|sc_array| {
        let sc_slice = sc_array.values().as_slice();
        for i in 0..sc_slice.len() {
            if sc_array.is_null(i) {
                builder.append_null();
                continue;
            }
            let sc_value = sc_slice[i];
            match count {
                n if n < timeperiod => {
                    count += 1;
                    sum += values[i];
                    builder.append_null();
                }
                n if n == timeperiod => {
                    count += 1;
                    kama_value = sum / timeperiod as f64;
                    builder.append_value(kama_value);
                }
                _ => {
                    kama_value = sc_value.mul_add(values[i] - kama_value, kama_value);
                    builder.append_value(kama_value);
                }
            }
        }
    });

    builder.finish()
}

pub fn calc_ma(values: &Float64Chunked, timeperiod: usize, matype: usize) -> Float64Chunked {
    match matype {
        1 => calc_ema(values, timeperiod),
        2 => calc_wma(values, timeperiod),
        3 => calc_dema(values, timeperiod),
        4 => calc_tema(values, timeperiod),
        5 => calc_trima(values, timeperiod),
        6 => calc_kama(values, timeperiod),
        7 => calc_sma(values, timeperiod), //MAMA 待实现
        8 => calc_t3(values, timeperiod, 0.0),
        _ => calc_sma(values, timeperiod),
    }
}

pub fn calc_sma(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("sma".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sma".into(), n);

    let denominator = 1.0 / timeperiod as f64;

    let mut count: usize = 0;
    let mut sum = 0.0f64;
    let mut window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;
                    sum += value;
                    window.push_back(value);

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
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;
                    sum += value;
                    window.push_back(value);
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
            }
        }
    });

    builder.finish()
}

pub fn calc_t3(values: &Float64Chunked, timeperiod: usize, vfactor: f64) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < 6 * timeperiod - 5 {
        return Float64Chunked::full_null("ema".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("t3".into(), n);

    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    let c1 = -vfactor.powi(3);
    let c2 = 3.0 * vfactor.powi(2) - 3.0 * c1;
    let c3 = -2.0 * c2 - 3.0 * c1 - 3.0 * vfactor;
    let c4 = 1.0 - c1 - c2 - c3;

    let mut count: usize = 0;
    let mut ema_value = ArrayVec::from([0.0f64; 6]);
    let mut sum = ArrayVec::from([0.0f64; 6]);

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum[0] += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum[0] += value;
                            ema_value[0] = sum[0] / timeperiod as f64;
                            sum[1] = ema_value[0];
                            builder.append_null();
                        }
                        n if n < 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            builder.append_null();
                        }
                        n if n == 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            ema_value[1] = sum[1] / timeperiod as f64;
                            sum[2] = ema_value[1];
                            builder.append_null();
                        }
                        n if n < 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            builder.append_null();
                        }
                        n if n == 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            ema_value[2] = sum[2] / timeperiod as f64;
                            sum[3] = ema_value[2];
                            builder.append_null();
                        }
                        n if n < 4 * timeperiod - 3 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            sum[3] += ema_value[2];
                            builder.append_null();
                        }
                        n if n == 4 * timeperiod - 3 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            sum[3] += ema_value[2];
                            ema_value[3] = sum[3] / timeperiod as f64;
                            sum[4] = ema_value[3];
                            builder.append_null();
                        }
                        n if n < 5 * timeperiod - 4 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            sum[4] += ema_value[3];
                            builder.append_null();
                        }
                        n if n == 5 * timeperiod - 4 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            sum[4] += ema_value[3];
                            ema_value[4] = sum[4] / timeperiod as f64;
                            sum[5] = ema_value[4];
                            builder.append_null();
                        }
                        n if n < 6 * timeperiod - 5 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            ema_value[4] = alpha.mul_add(ema_value[3] - ema_value[4], ema_value[4]);
                            sum[5] += ema_value[4];
                            builder.append_null();
                        }
                        _ => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            ema_value[4] = alpha.mul_add(ema_value[3] - ema_value[4], ema_value[4]);
                            ema_value[5] = alpha.mul_add(ema_value[4] - ema_value[5], ema_value[5]);
                            builder.append_value(
                                6.0 * ema_value[0] - 15.0 * ema_value[1] + 20.0 * ema_value[2]
                                    - 15.0 * ema_value[3]
                                    + 6.0 * ema_value[4]
                                    - ema_value[5],
                            );
                        }
                    }
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum[0] += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum[0] += value;
                            ema_value[0] = sum[0] / timeperiod as f64;
                            sum[1] = ema_value[0];
                            builder.append_null();
                        }
                        n if n < 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            builder.append_null();
                        }
                        n if n == 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            ema_value[1] = sum[1] / timeperiod as f64;
                            sum[2] = ema_value[1];
                            builder.append_null();
                        }
                        n if n < 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            builder.append_null();
                        }
                        n if n == 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            ema_value[2] = sum[2] / timeperiod as f64;
                            sum[3] = ema_value[2];
                            builder.append_null();
                        }
                        n if n < 4 * timeperiod - 3 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            sum[3] += ema_value[2];
                            builder.append_null();
                        }
                        n if n == 4 * timeperiod - 3 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            sum[3] += ema_value[2];
                            ema_value[3] = sum[3] / timeperiod as f64;
                            sum[4] = ema_value[3];
                            builder.append_null();
                        }
                        n if n < 5 * timeperiod - 4 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            sum[4] += ema_value[3];
                            builder.append_null();
                        }
                        n if n == 5 * timeperiod - 4 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            sum[4] += ema_value[3];
                            ema_value[4] = sum[4] / timeperiod as f64;
                            sum[5] = ema_value[4];
                            builder.append_null();
                        }
                        n if n < 6 * timeperiod - 5 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            ema_value[4] = alpha.mul_add(ema_value[3] - ema_value[4], ema_value[4]);
                            sum[5] += ema_value[4];
                            builder.append_null();
                        }
                        _ => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            ema_value[3] = alpha.mul_add(ema_value[2] - ema_value[3], ema_value[3]);
                            ema_value[4] = alpha.mul_add(ema_value[3] - ema_value[4], ema_value[4]);
                            ema_value[5] = alpha.mul_add(ema_value[4] - ema_value[5], ema_value[5]);
                            builder.append_value(c1.mul_add(
                                ema_value[5],
                                c2.mul_add(
                                    ema_value[4],
                                    c3.mul_add(ema_value[3], c4 * ema_value[2]),
                                ),
                            ));
                        }
                    }
                }
            }
        }
    });

    builder.finish()
}

pub fn calc_tema(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < 3 * timeperiod - 2 {
        return Float64Chunked::full_null("tema".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("tema".into(), n);

    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    let mut count: usize = 0;
    let mut ema_value = ArrayVec::from([0.0f64; 3]);
    let mut sum = ArrayVec::from([0.0f64; 3]);

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum[0] += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum[0] += value;
                            ema_value[0] = sum[0] / timeperiod as f64;
                            sum[1] = ema_value[0];
                            builder.append_null();
                        }
                        n if n < 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            builder.append_null();
                        }
                        n if n == 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            ema_value[1] = sum[1] / timeperiod as f64;
                            sum[2] = ema_value[1];
                            builder.append_null();
                        }
                        n if n < 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            builder.append_null();
                        }
                        n if n == 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            ema_value[2] = sum[2] / timeperiod as f64;
                            builder.append_value(
                                3.0 * ema_value[0] - 3.0 * ema_value[1] + ema_value[2],
                            );
                        }
                        _ => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            builder.append_value(
                                3.0 * ema_value[0] - 3.0 * ema_value[1] + ema_value[2],
                            );
                        }
                    }
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;

                    match count {
                        n if n < timeperiod => {
                            sum[0] += value;
                            builder.append_null();
                        }
                        n if n == timeperiod => {
                            sum[0] += value;
                            ema_value[0] = sum[0] / timeperiod as f64;
                            sum[1] = ema_value[0];
                            builder.append_null();
                        }
                        n if n < 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            builder.append_null();
                        }
                        n if n == 2 * timeperiod - 1 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            sum[1] += ema_value[0];
                            ema_value[1] = sum[1] / timeperiod as f64;
                            sum[2] = ema_value[1];
                            builder.append_null();
                        }
                        n if n < 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            builder.append_null();
                        }
                        n if n == 3 * timeperiod - 2 => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            sum[2] += ema_value[1];
                            ema_value[2] = sum[2] / timeperiod as f64;
                            builder.append_value(
                                3.0 * ema_value[0] - 3.0 * ema_value[1] + ema_value[2],
                            );
                        }
                        _ => {
                            ema_value[0] = alpha.mul_add(value - ema_value[0], ema_value[0]);
                            ema_value[1] = alpha.mul_add(ema_value[0] - ema_value[1], ema_value[1]);
                            ema_value[2] = alpha.mul_add(ema_value[1] - ema_value[2], ema_value[2]);
                            builder.append_value(
                                3.0 * ema_value[0] - 3.0 * ema_value[1] + ema_value[2],
                            );
                        }
                    }
                }
            }
        }
    });

    builder.finish()
}

pub fn calc_trima(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let mut trima = match timeperiod % 2 {
        1 => {
            let n = timeperiod / 2 + 1;
            calc_sma(&calc_sma(values, n), n)
        }
        _ => {
            let n = timeperiod / 2;
            calc_sma(&calc_sma(values, n), n + 1)
        }
    };
    trima.rename("trima".into());
    trima
}

pub fn calc_wma(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod {
        return Float64Chunked::full_null("sma".into(), n);
    }

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sma".into(), n);

    let mut count = 0;
    let denominator: f64 = (timeperiod * (timeperiod + 1) / 2) as f64;
    let mut numerator: f64 = 0.0;
    let mut sum = 0.0f64;
    let mut window: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    values.downcast_iter().for_each(|array| {
        let array_slice = array.values().as_slice();
        let array_validity = array.validity();
        match array_validity {
            Some(bitmap) => {
                for i in 0..array_slice.len() {
                    if !bitmap.get_bit(i) {
                        builder.append_null();
                        continue;
                    }
                    let value = array_slice[i];
                    count += 1;
                    sum += value;
                    numerator += (count as f64) * value;
                    window.push_back(value);

                    if count < timeperiod {
                        builder.append_null();
                    } else {
                        if count > timeperiod {
                            if let Some(old_value) = window.pop_front() {
                                sum -= old_value;
                                numerator -= (timeperiod as f64) * old_value;
                                count -= 1;
                            }
                        }
                        builder.append_value(numerator / denominator);
                    }
                }
            }
            None => {
                for i in 0..array_slice.len() {
                    let value = array_slice[i];
                    count += 1;
                    sum += value;
                    numerator += (count as f64) * value;
                    window.push_back(value);

                    if count < timeperiod {
                        builder.append_null();
                    } else {
                        if count > timeperiod {
                            if let Some(old_value) = window.pop_front() {
                                sum -= old_value;
                                numerator -= (timeperiod as f64) * old_value;
                                count -= 1;
                            }
                        }
                        builder.append_value(numerator / denominator);
                    }
                }
            }
        }
    });

    builder.finish()
}
