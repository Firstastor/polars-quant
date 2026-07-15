use arrayvec::ArrayVec;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_arrow::array::Array;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use serde::Deserialize;
use std::collections::VecDeque;

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
pub struct MavpKwargs {
    pub minperiod: Option<usize>,
    pub maxperiod: Option<usize>,
    pub matype: Option<usize>,
}

#[derive(Deserialize)]
pub struct T3Kwargs {
    pub timeperiod: Option<usize>,
    pub vfactor: Option<f64>,
}

fn bbands_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("bbands_upper".into(), DataType::Float64);
    let f2 = Field::new("bbands_middle".into(), DataType::Float64);
    let f3 = Field::new("bbands_lower".into(), DataType::Float64);
    Ok(Field::new(
        "bbands".into(),
        DataType::Struct(vec![f1, f2, f3]),
    ))
}

#[polars_expr(output_type_func=bbands_output)]
pub fn bbands(inputs: &[Series], kwargs: BbandsKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(5);
    let nbdevup = kwargs.nbdevup.unwrap_or(2.0);
    let nbdevdn = kwargs.nbdevdn.unwrap_or(2.0);

    let n = real.len();
    if timeperiod == 0 || n < timeperiod || real.null_count() != 0 {
        let null_ca = Float64Chunked::full_null("".into(), n);
        let s1 = null_ca
            .clone()
            .with_name("bbands_upper".into())
            .into_series();
        let s2 = null_ca
            .clone()
            .with_name("bbands_middle".into())
            .into_series();
        let s3 = null_ca.with_name("bbands_lower".into()).into_series();
        return Ok(
            StructChunked::from_series("bbands".into(), n, [s1, s2, s3].iter())?.into_series(),
        );
    }

    let real = real.rechunk();

    let mut upper = PrimitiveChunkedBuilder::<Float64Type>::new("bbands_upper".into(), n);
    let mut middle = PrimitiveChunkedBuilder::<Float64Type>::new("bbands_middle".into(), n);
    let mut lower = PrimitiveChunkedBuilder::<Float64Type>::new("bbands_lower".into(), n);

    let inverse_period: f64 = 1.0 / timeperiod as f64;
    let mut count = 0;
    let mut mean = 0.;
    let mut old = AllocRingBuffer::new(timeperiod);
    let mut sum: f64 = 0.;
    let mut sum_squares: f64 = 0.;

    for val in real.no_null_iter() {
        count += 1;

        if count < timeperiod {
            old.enqueue(val);
            sum += val;
            sum_squares += val.powi(2);
            upper.append_null();
            middle.append_null();
            lower.append_null();
        } else if count == timeperiod {
            old.enqueue(val);
            sum += val;
            sum_squares += val.powi(2);
            mean = sum * inverse_period;
            let deviation = (sum_squares * inverse_period - mean.powi(2)).sqrt();
            upper.append_value(mean + nbdevup * deviation);
            middle.append_value(mean);
            lower.append_value(mean - nbdevdn * deviation);
        } else {
            let old = old.enqueue(val).unwrap();
            mean += (val - old) * inverse_period;
            sum_squares += val.powi(2) - old.powi(2);
            let deviation = (sum_squares * inverse_period - mean.powi(2)).sqrt();
            upper.append_value(mean + nbdevup * deviation);
            middle.append_value(mean);
            lower.append_value(mean - nbdevdn * deviation);
        }
    }

    let s1 = upper.finish().into_series();
    let s2 = middle.finish().into_series();
    let s3 = lower.finish().into_series();
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

#[polars_expr(output_type=Float64)]
pub fn mavp(inputs: &[Series], kwargs: MavpKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let periods = inputs[1].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let periods = periods.f64()?;
    let minperiod = kwargs.minperiod.unwrap_or(2);
    let maxperiod = kwargs.maxperiod.unwrap_or(30);
    let matype = kwargs.matype.unwrap_or(0);

    Ok(calc_mavp(real, periods, minperiod, maxperiod, matype).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn midpoint(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?;
    let real = real.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let n = real.len();
    if timeperiod == 0 || n < timeperiod || real.null_count() != 0 {
        return Ok(Float64Chunked::full_null("sma".into(), n).into_series());
    }

    let real = real.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("midpoint".into(), n);

    let mut window: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod);
    let mut window_max: VecDeque<f64> = VecDeque::with_capacity(timeperiod);
    let mut window_min: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    let mut real_iter = real.no_null_iter();

    #[inline(always)]
    fn move_window(
        val: f64,
        window: &mut AllocRingBuffer<f64>,
        window_max: &mut VecDeque<f64>,
        window_min: &mut VecDeque<f64>,
    ) {
        window.enqueue(val);

        while let Some(last) = window_max.back() {
            if &val >= last {
                window_max.pop_back();
            } else {
                break;
            }
        }
        window_max.push_back(val);

        while let Some(last) = window_min.back() {
            if &val <= last {
                window_min.pop_back();
            } else {
                break;
            }
        }
        window_min.push_back(val);
    }

    for val in real_iter.by_ref().take(timeperiod - 1) {
        move_window(val, &mut window, &mut window_max, &mut window_min);
        builder.append_null();
    }

    {
        let val = real_iter.by_ref().next().unwrap();
        move_window(val, &mut window, &mut window_max, &mut window_min);
        builder.append_value((window_max.front().unwrap() + window_min.front().unwrap()) / 2.0);
    }

    for val in real_iter {
        if let Some(pop) = window.dequeue() {
            window_max.pop_front_if(|&mut front| pop == front);
            window_min.pop_front_if(|&mut front| pop == front);
        };

        move_window(val, &mut window, &mut window_max, &mut window_min);
        builder.append_value((window_max.front().unwrap() + window_min.front().unwrap()) / 2.0);
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn midprice(inputs: &[Series], kwargs: MaKwargs) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let high = high.f64()?;
    let low = low.f64()?;
    let timeperiod = kwargs.timeperiod.unwrap_or(14);

    let n = high.len();
    if timeperiod == 0 || n < timeperiod || high.null_count() != 0 || low.null_count() != 0 {
        return Ok(Float64Chunked::full_null("midprice".into(), n).into_series());
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("midprice".into(), n);

    let mut window_high: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod);
    let mut window_low: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod);
    let mut window_max: VecDeque<f64> = VecDeque::with_capacity(timeperiod);
    let mut window_min: VecDeque<f64> = VecDeque::with_capacity(timeperiod);

    let mut high_iter = high.no_null_iter();
    let mut low_iter = low.no_null_iter();

    #[inline(always)]
    fn move_window(
        h_val: f64,
        l_val: f64,
        window_high: &mut AllocRingBuffer<f64>,
        window_low: &mut AllocRingBuffer<f64>,
        window_max: &mut VecDeque<f64>,
        window_min: &mut VecDeque<f64>,
    ) {
        window_high.enqueue(h_val);
        window_low.enqueue(l_val);

        while let Some(last) = window_max.back() {
            if &h_val >= last {
                window_max.pop_back();
            } else {
                break;
            }
        }
        window_max.push_back(h_val);

        while let Some(last) = window_min.back() {
            if &l_val <= last {
                window_min.pop_back();
            } else {
                break;
            }
        }
        window_min.push_back(l_val);
    }

    for (h_val, l_val) in high_iter
        .by_ref()
        .take(timeperiod - 1)
        .zip(low_iter.by_ref().take(timeperiod - 1))
    {
        move_window(
            h_val,
            l_val,
            &mut window_high,
            &mut window_low,
            &mut window_max,
            &mut window_min,
        );
        builder.append_null();
    }

    {
        let h_val = high_iter.by_ref().next().unwrap();
        let l_val = low_iter.by_ref().next().unwrap();
        move_window(
            h_val,
            l_val,
            &mut window_high,
            &mut window_low,
            &mut window_max,
            &mut window_min,
        );
        builder.append_value((window_max.front().unwrap() + window_min.front().unwrap()) / 2.0);
    }

    for (h_val, l_val) in high_iter.zip(low_iter) {
        if let Some(high_pop) = window_high.dequeue() {
            window_max.pop_front_if(|&mut front| high_pop == front);
        };

        if let Some(low_pop) = window_low.dequeue() {
            window_min.pop_front_if(|&mut front| low_pop == front);
        };

        move_window(
            h_val,
            l_val,
            &mut window_high,
            &mut window_low,
            &mut window_max,
            &mut window_min,
        );
        builder.append_value((window_max.front().unwrap() + window_min.front().unwrap()) / 2.0);
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn sar(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let high = high.f64()?;
    let low = low.f64()?;
    let acceleration = inputs[2].f64()?.get(0).unwrap_or(0.02);
    let maximum = inputs[3].f64()?.get(0).unwrap_or(0.2);

    Ok(calc_sar(high, low, acceleration, maximum).into_series())
}

#[polars_expr(output_type=Float64)]
pub fn sarext(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let high = high.f64()?;
    let low = low.f64()?;
    let startvalue = inputs[2].f64()?.get(0).unwrap_or(0.0);
    let offsetonreverse = inputs[3].f64()?.get(0).unwrap_or(0.0);
    let accelerationinitlong = inputs[4].f64()?.get(0).unwrap_or(0.02);
    let accelerationlong = inputs[5].f64()?.get(0).unwrap_or(0.02);
    let accelerationmaxlong = inputs[6].f64()?.get(0).unwrap_or(0.2);
    let accelerationinitshort = inputs[7].f64()?.get(0).unwrap_or(0.02);
    let accelerationshort = inputs[8].f64()?.get(0).unwrap_or(0.02);
    let accelerationmaxshort = inputs[9].f64()?.get(0).unwrap_or(0.2);

    Ok(calc_sarext(
        high,
        low,
        startvalue,
        offsetonreverse,
        accelerationinitlong,
        accelerationlong,
        accelerationmaxlong,
        accelerationinitshort,
        accelerationshort,
        accelerationmaxshort,
    )
    .into_series())
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
    let vfactor = kwargs.vfactor.unwrap_or(0.7);
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

pub fn calc_dema(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < 2 * timeperiod - 1 || values.null_count() != 0 {
        return Float64Chunked::full_null("dema".into(), n);
    }

    let values = values.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("dema".into(), n);

    let alpha: f64 = 2.0 / (timeperiod + 1) as f64;
    let inverse_period: f64 = 1.0 / timeperiod as f64;
    let mut count = 0;
    let mut ema1 = 0.;
    let mut sum: f64 = 0.;

    for val in values.no_null_iter() {
        count += 1;
        if count < timeperiod {
            sum += val;
            builder.append_null();
        } else if count == timeperiod {
            ema1 = (sum + val) * inverse_period;
            // Now sum is ema2
            sum = ema1;
            builder.append_null();
        } else if count < 2 * timeperiod - 1 {
            ema1 = alpha.mul_add(val - ema1, ema1);
            sum += ema1;
            builder.append_null();
        } else if count == 2 * timeperiod - 1 {
            ema1 = alpha.mul_add(val - ema1, ema1);
            sum = (sum + ema1) * inverse_period;
            builder.append_value(2.0 * ema1 - sum);
        } else {
            ema1 = alpha.mul_add(val - ema1, ema1);
            sum = alpha.mul_add(ema1 - sum, sum);
            builder.append_value(2.0 * ema1 - sum);
        }
    }

    builder.finish()
}

pub fn calc_ema(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod || values.null_count() != 0 {
        return Float64Chunked::full_null("ema".into(), n);
    }

    let values = values.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ema".into(), n);

    let alpha: f64 = 2.0 / (timeperiod + 1) as f64;
    let mut count = 0;
    let mut sum: f64 = 0.;
    let mut sum_ema = 0.;

    for val in values.no_null_iter() {
        count += 1;

        if count < timeperiod {
            sum += val;
            builder.append_null();
        } else if count == timeperiod {
            sum += val;
            sum_ema = sum / timeperiod as f64;
            builder.append_value(sum_ema);
        } else {
            sum_ema = alpha.mul_add(val - sum_ema, sum_ema);
            builder.append_value(sum_ema);
        }
    }

    builder.finish()
}

pub fn calc_kama(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod + 1 {
        return Float64Chunked::full_null("kama".into(), n);
    }

    let mut er_builder = PrimitiveChunkedBuilder::<Float64Type>::new("er".into(), n);
    let mut count: usize = 0;
    let mut volatility = 0.0f64;
    let mut window: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod + 1);
    let mut adj_diffs: AllocRingBuffer<f64> = AllocRingBuffer::new(timeperiod);
    let mut prev: Option<f64> = None;

    values.downcast_iter().for_each(|arr| {
        if arr.null_count() == 0 {
            for &val in arr.values_iter() {
                count += 1;
                let adj = match prev {
                    Some(p) => (val - p).abs(),
                    None => 0.0,
                };
                prev = Some(val);
                volatility += adj;

                if adj_diffs.is_full() {
                    if let Some(old_diff) = adj_diffs.dequeue() {
                        volatility -= old_diff;
                    }
                }
                adj_diffs.enqueue(adj);

                if window.is_full() {
                    window.dequeue();
                }
                window.enqueue(val);

                if count < timeperiod + 1 {
                    er_builder.append_null();
                } else {
                    let direction = (val - *window.front().unwrap()).abs();
                    if volatility == 0.0 {
                        er_builder.append_value(0.0);
                    } else {
                        er_builder.append_value(direction / volatility);
                    }
                }
            }
        } else {
            for opt in arr.iter() {
                let Some(&val) = opt else {
                    er_builder.append_null();
                    continue;
                };
                count += 1;
                let adj = match prev {
                    Some(p) => (val - p).abs(),
                    None => 0.0,
                };
                prev = Some(val);
                volatility += adj;

                if adj_diffs.is_full() {
                    if let Some(old_diff) = adj_diffs.dequeue() {
                        volatility -= old_diff;
                    }
                }
                adj_diffs.enqueue(adj);

                if window.is_full() {
                    window.dequeue();
                }
                window.enqueue(val);

                if count < timeperiod + 1 {
                    er_builder.append_null();
                } else {
                    let direction = (val - *window.front().unwrap()).abs();
                    if volatility == 0.0 {
                        er_builder.append_value(0.0);
                    } else {
                        er_builder.append_value(direction / volatility);
                    }
                }
            }
        }
    });

    let er = er_builder.finish();

    let fast_sc = 2.0 / 3.0;
    let slow_sc = 2.0 / 31.0;
    let sc_sqrt = &er * (fast_sc - slow_sc) + slow_sc;
    let sc = &sc_sqrt * &sc_sqrt;

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("kama".into(), n);
    let mut kama_val = 0.0f64;
    let mut initialized = false;

    // Build a contiguous slice of values for random access in phase 2
    let vals = values.cont_slice().unwrap_or(&[]);
    let vals_len = vals.len();

    sc.downcast_iter().for_each(|sc_arr| {
        for (i, sc_opt) in sc_arr.iter().enumerate() {
            let Some(&sc_val) = sc_opt else {
                builder.append_null();
                continue;
            };
            if i >= vals_len {
                builder.append_null();
                continue;
            }
            let val = vals[i];
            if !initialized {
                initialized = true;
                kama_val = vals[i - 1]; // yesterdayKAMA is the previous close
            }
            kama_val = sc_val.mul_add(val - kama_val, kama_val);
            builder.append_value(kama_val);
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
        7 => calc_sma(values, timeperiod), // MAMA tempoally drop
        8 => calc_t3(values, timeperiod, 0.0),
        _ => calc_sma(values, timeperiod),
    }
}

pub fn calc_mavp(
    real: &Float64Chunked,
    periods: &Float64Chunked,
    minperiod: usize,
    maxperiod: usize,
    matype: usize,
) -> Float64Chunked {
    let n = real.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("mavp".into(), n);
    if n == 0 {
        return builder.finish();
    }

    let mut buffer = vec![0.0f64; maxperiod.max(1)];
    let buf_len = buffer.len();
    let mut buf_idx: usize = 0;
    let mut count: usize = 0;

    for (real_arr, period_arr) in real.downcast_iter().zip(periods.downcast_iter()) {
        let real_fast = real_arr.null_count() == 0;
        let period_fast = period_arr.null_count() == 0;
        if real_fast && period_fast {
            for (&rv, &pv) in real_arr.values_iter().zip(period_arr.values_iter()) {
                buffer[buf_idx] = rv;
                count += 1;

                let mut period = if pv.is_finite() { pv.trunc() as i64 } else { 0 };
                let minp = minperiod as i64;
                let maxp = maxperiod as i64;
                if period < minp {
                    period = minp;
                }
                if period > maxp {
                    period = maxp;
                }
                let period = period as usize;

                if period == 0 || count < period {
                    builder.append_null();
                    buf_idx = (buf_idx + 1) % buf_len;
                    continue;
                }

                let result = match matype {
                    1 => {
                        let alpha = 2.0 / (period as f64 + 1.0);
                        let mut sum = 0.0f64;
                        let mut ridx = buf_idx;
                        for _ in 0..period {
                            sum += buffer[ridx];
                            ridx = ridx.wrapping_sub(1);
                            if ridx >= buf_len {
                                ridx = buf_len - 1;
                            }
                        }
                        let seed = sum / period as f64;
                        let mut ema_val = seed;
                        let mut oldest = buf_idx;
                        for _ in 1..period {
                            oldest = oldest.wrapping_sub(1);
                            if oldest >= buf_len {
                                oldest = buf_len - 1;
                            }
                        }
                        let mut fidx = oldest;
                        for _ in 0..period {
                            let v = buffer[fidx];
                            ema_val = alpha.mul_add(v - ema_val, ema_val);
                            fidx += 1;
                            if fidx == buf_len {
                                fidx = 0;
                            }
                        }
                        ema_val
                    }
                    2 => {
                        let denom: f64 = (period * (period + 1) / 2) as f64;
                        let mut weight = period as f64;
                        let mut num = 0.0f64;
                        let mut ridx = buf_idx;
                        for _ in 0..period {
                            num += weight * buffer[ridx];
                            weight -= 1.0;
                            ridx = ridx.wrapping_sub(1);
                            if ridx >= buf_len {
                                ridx = buf_len - 1;
                            }
                        }
                        num / denom
                    }
                    _ => {
                        let mut sum = 0.0f64;
                        let mut ridx = buf_idx;
                        for _ in 0..period {
                            sum += buffer[ridx];
                            ridx = ridx.wrapping_sub(1);
                            if ridx >= buf_len {
                                ridx = buf_len - 1;
                            }
                        }
                        sum / period as f64
                    }
                };

                builder.append_value(result);
                buf_idx = (buf_idx + 1) % buf_len;
            }
        } else {
            for (real_opt, period_opt) in real_arr.iter().zip(period_arr.iter()) {
                let rv = real_opt.copied().unwrap_or(0.0);
                let pv = period_opt.copied().unwrap_or(0.0);
                buffer[buf_idx] = rv;
                count += 1;

                let mut period = if pv.is_finite() { pv.trunc() as i64 } else { 0 };
                let minp = minperiod as i64;
                let maxp = maxperiod as i64;
                if period < minp {
                    period = minp;
                }
                if period > maxp {
                    period = maxp;
                }
                let period = period as usize;

                if period == 0 || count < period {
                    builder.append_null();
                    buf_idx = (buf_idx + 1) % buf_len;
                    continue;
                }

                let result = match matype {
                    1 => {
                        let alpha = 2.0 / (period as f64 + 1.0);
                        let mut sum = 0.0f64;
                        let mut ridx = buf_idx;
                        for _ in 0..period {
                            sum += buffer[ridx];
                            ridx = ridx.wrapping_sub(1);
                            if ridx >= buf_len {
                                ridx = buf_len - 1;
                            }
                        }
                        let seed = sum / period as f64;
                        let mut ema_val = seed;
                        let mut oldest = buf_idx;
                        for _ in 1..period {
                            oldest = oldest.wrapping_sub(1);
                            if oldest >= buf_len {
                                oldest = buf_len - 1;
                            }
                        }
                        let mut fidx = oldest;
                        for _ in 0..period {
                            let v = buffer[fidx];
                            ema_val = alpha.mul_add(v - ema_val, ema_val);
                            fidx += 1;
                            if fidx == buf_len {
                                fidx = 0;
                            }
                        }
                        ema_val
                    }
                    2 => {
                        let denom: f64 = (period * (period + 1) / 2) as f64;
                        let mut weight = period as f64;
                        let mut num = 0.0f64;
                        let mut ridx = buf_idx;
                        for _ in 0..period {
                            num += weight * buffer[ridx];
                            weight -= 1.0;
                            ridx = ridx.wrapping_sub(1);
                            if ridx >= buf_len {
                                ridx = buf_len - 1;
                            }
                        }
                        num / denom
                    }
                    _ => {
                        let mut sum = 0.0f64;
                        let mut ridx = buf_idx;
                        for _ in 0..period {
                            sum += buffer[ridx];
                            ridx = ridx.wrapping_sub(1);
                            if ridx >= buf_len {
                                ridx = buf_len - 1;
                            }
                        }
                        sum / period as f64
                    }
                };

                builder.append_value(result);
                buf_idx = (buf_idx + 1) % buf_len;
            }
        }
    }

    builder.finish()
}

pub fn calc_rma(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod || values.null_count() != 0 {
        return Float64Chunked::full_null("rma".into(), n);
    }

    let values = values.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("rma".into(), n);

    let alpha: f64 = 1.0 / timeperiod as f64;
    let mut sum: f64 = 0.;

    let mut values_iter = values.no_null_iter();

    for val in values_iter.by_ref().take(timeperiod - 1) {
        sum += val;
        builder.append_null();
    }

    sum += values_iter.by_ref().next().unwrap();
    let mut sum_rma = sum / timeperiod as f64;
    builder.append_value(sum_rma);

    for val in values_iter {
        sum_rma = alpha.mul_add(val - sum_rma, sum_rma);
        builder.append_value(sum_rma);
    }

    builder.finish()
}

pub fn calc_sar(
    high: &Float64Chunked,
    low: &Float64Chunked,
    acceleration: f64,
    maximum: f64,
) -> Float64Chunked {
    let n = high.len();
    let mut high_vec = Vec::with_capacity(n);
    let mut low_vec = Vec::with_capacity(n);
    for (h_opt, l_opt) in high.iter().zip(low.iter()) {
        high_vec.push(h_opt.unwrap_or(0.0));
        low_vec.push(l_opt.unwrap_or(0.0));
    }
    let high = &high_vec;
    let low = &low_vec;
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sar".into(), n);

    if n < 2 {
        for _ in 0..n {
            builder.append_null();
        }
        return builder.finish();
    }

    let mut af = acceleration;
    if af > maximum {
        af = maximum;
    }

    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let mut is_long = !(diff_m > 0.0 && diff_p < diff_m);

    let mut ep;
    let mut sar;

    if is_long {
        ep = high[1];
        sar = low[0];
    } else {
        ep = low[1];
        sar = high[0];
    }

    builder.append_null();

    let mut new_low = low[1];
    let mut new_high = high[1];

    for i in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[i];
        new_high = high[i];

        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep;
                if sar < prev_high {
                    sar = prev_high;
                }
                if sar < new_high {
                    sar = new_high;
                }

                builder.append_value(sar);
                af = acceleration;
                ep = new_low;
                sar = sar + af * (ep - sar);
                if sar < prev_high {
                    sar = prev_high;
                }
                if sar < new_high {
                    sar = new_high;
                }
            } else {
                builder.append_value(sar);
                if new_high > ep {
                    ep = new_high;
                    af += acceleration;
                    if af > maximum {
                        af = maximum;
                    }
                }
                sar = sar + af * (ep - sar);
                if sar > prev_low {
                    sar = prev_low;
                }
                if sar > new_low {
                    sar = new_low;
                }
            }
        } else {
            if new_high >= sar {
                is_long = true;
                sar = ep;
                if sar > prev_low {
                    sar = prev_low;
                }
                if sar > new_low {
                    sar = new_low;
                }

                builder.append_value(sar);
                af = acceleration;
                ep = new_high;
                sar = sar + af * (ep - sar);
                if sar > prev_low {
                    sar = prev_low;
                }
                if sar > new_low {
                    sar = new_low;
                }
            } else {
                builder.append_value(sar);
                if new_low < ep {
                    ep = new_low;
                    af += acceleration;
                    if af > maximum {
                        af = maximum;
                    }
                }
                sar = sar + af * (ep - sar);
                if sar < prev_high {
                    sar = prev_high;
                }
                if sar < new_high {
                    sar = new_high;
                }
            }
        }
    }

    builder.finish()
}

pub fn calc_sarext(
    high: &Float64Chunked,
    low: &Float64Chunked,
    startvalue: f64,
    offsetonreverse: f64,
    accelerationinitlong: f64,
    accelerationlong: f64,
    accelerationmaxlong: f64,
    accelerationinitshort: f64,
    accelerationshort: f64,
    accelerationmaxshort: f64,
) -> Float64Chunked {
    let n = high.len();
    let mut high_vec = Vec::with_capacity(n);
    let mut low_vec = Vec::with_capacity(n);
    for (h_opt, l_opt) in high.iter().zip(low.iter()) {
        high_vec.push(h_opt.unwrap_or(0.0));
        low_vec.push(l_opt.unwrap_or(0.0));
    }
    let high = &high_vec;
    let low = &low_vec;
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sarext".into(), n);

    if n < 2 {
        for _ in 0..n {
            builder.append_null();
        }
        return builder.finish();
    }

    let diff_p = high[1] - high[0];
    let diff_m = low[0] - low[1];
    let auto_is_long = !(diff_m > 0.0 && diff_p < diff_m);

    let mut is_long = if startvalue > 0.0 {
        true
    } else if startvalue < 0.0 {
        false
    } else {
        auto_is_long
    };

    let mut af = if is_long {
        accelerationinitlong
    } else {
        accelerationinitshort
    };
    if is_long && af > accelerationmaxlong {
        af = accelerationmaxlong;
    } else if !is_long && af > accelerationmaxshort {
        af = accelerationmaxshort;
    }

    let mut ep;
    let mut sar;

    if is_long {
        ep = high[1];
        sar = if startvalue != 0.0 {
            startvalue.abs()
        } else {
            low[0]
        };
    } else {
        ep = low[1];
        sar = if startvalue != 0.0 {
            startvalue.abs()
        } else {
            high[0]
        };
    }

    builder.append_null();

    let mut new_low = low[1];
    let mut new_high = high[1];

    for i in 1..n {
        let prev_low = new_low;
        let prev_high = new_high;
        new_low = low[i];
        new_high = high[i];

        if is_long {
            if new_low <= sar {
                is_long = false;
                sar = ep;
                if sar < prev_high {
                    sar = prev_high;
                }
                if sar < new_high {
                    sar = new_high;
                }

                sar += sar * offsetonreverse;

                builder.append_value(-sar);
                af = accelerationinitshort;
                ep = new_low;
                sar = sar + af * (ep - sar);
                if sar < prev_high {
                    sar = prev_high;
                }
                if sar < new_high {
                    sar = new_high;
                }
            } else {
                builder.append_value(sar);
                if new_high > ep {
                    ep = new_high;
                    af += accelerationlong;
                    if af > accelerationmaxlong {
                        af = accelerationmaxlong;
                    }
                }
                sar = sar + af * (ep - sar);
                if sar > prev_low {
                    sar = prev_low;
                }
                if sar > new_low {
                    sar = new_low;
                }
            }
        } else {
            if new_high >= sar {
                is_long = true;
                sar = ep;
                if sar > prev_low {
                    sar = prev_low;
                }
                if sar > new_low {
                    sar = new_low;
                }

                sar -= sar * offsetonreverse;

                builder.append_value(sar);
                af = accelerationinitlong;
                ep = new_high;
                sar = sar + af * (ep - sar);
                if sar > prev_low {
                    sar = prev_low;
                }
                if sar > new_low {
                    sar = new_low;
                }
            } else {
                builder.append_value(-sar);
                if new_low < ep {
                    ep = new_low;
                    af += accelerationshort;
                    if af > accelerationmaxshort {
                        af = accelerationmaxshort;
                    }
                }
                sar = sar + af * (ep - sar);
                if sar < prev_high {
                    sar = prev_high;
                }
                if sar < new_high {
                    sar = new_high;
                }
            }
        }
    }

    builder.finish()
}

pub fn calc_sma(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();
    if timeperiod == 0 || n < timeperiod || values.null_count() != 0 {
        return Float64Chunked::full_null("sma".into(), n);
    }

    let values = values.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("sma".into(), n);

    let inverse_period = 1.0 / timeperiod as f64;
    let mut count = 0;
    let mut old = AllocRingBuffer::new(timeperiod);
    let mut sum: f64 = 0.;

    for val in values.no_null_iter() {
        count += 1;

        if count < timeperiod {
            sum += val;
            old.enqueue(val);
            builder.append_null();
        } else if count == timeperiod {
            sum += val;
            old.enqueue(val);
            builder.append_value(sum * inverse_period);
        } else {
            sum += val - old.dequeue().unwrap();
            old.enqueue(val);
            builder.append_value(sum * inverse_period);
        }
    }

    builder.finish()
}

pub fn calc_t3(values: &Float64Chunked, timeperiod: usize, vfactor: f64) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < 6 * timeperiod - 5 {
        return Float64Chunked::full_null("t3".into(), n);
    }

    let values = values.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("t3".into(), n);

    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    let c1 = -vfactor.powi(3);
    let c2 = 3.0 * vfactor.powi(2) - 3.0 * c1;
    let c3 = -2.0 * c2 - 3.0 * c1 - 3.0 * vfactor;
    let c4 = 1.0 - c1 - c2 - c3;

    let mut count: usize = 0;
    let mut ema = ArrayVec::from([0.0f64; 6]);
    let mut sum = ArrayVec::from([0.0f64; 6]);

    values.downcast_iter().for_each(|arr| {
        if arr.null_count() == 0 {
            for &val in arr.values_iter() {
                count += 1;
                match count {
                    n if n < timeperiod => {
                        sum[0] += val;
                        builder.append_null();
                    }
                    n if n == timeperiod => {
                        sum[0] += val;
                        ema[0] = sum[0] / timeperiod as f64;
                        sum[1] = ema[0];
                        builder.append_null();
                    }
                    n if n < 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        builder.append_null();
                    }
                    n if n == 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        ema[1] = sum[1] / timeperiod as f64;
                        sum[2] = ema[1];
                        builder.append_null();
                    }
                    n if n < 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        builder.append_null();
                    }
                    n if n == 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        ema[2] = sum[2] / timeperiod as f64;
                        sum[3] = ema[2];
                        builder.append_null();
                    }
                    n if n < 4 * timeperiod - 3 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        sum[3] += ema[2];
                        builder.append_null();
                    }
                    n if n == 4 * timeperiod - 3 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        sum[3] += ema[2];
                        ema[3] = sum[3] / timeperiod as f64;
                        sum[4] = ema[3];
                        builder.append_null();
                    }
                    n if n < 5 * timeperiod - 4 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        sum[4] += ema[3];
                        builder.append_null();
                    }
                    n if n == 5 * timeperiod - 4 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        sum[4] += ema[3];
                        ema[4] = sum[4] / timeperiod as f64;
                        sum[5] = ema[4];
                        builder.append_null();
                    }
                    n if n < 6 * timeperiod - 5 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        ema[4] = alpha.mul_add(ema[3] - ema[4], ema[4]);
                        sum[5] += ema[4];
                        builder.append_null();
                    }
                    n if n == 6 * timeperiod - 5 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        ema[4] = alpha.mul_add(ema[3] - ema[4], ema[4]);
                        sum[5] += ema[4];
                        ema[5] = sum[5] / timeperiod as f64;
                        builder.append_value(
                            c1.mul_add(ema[5], c2.mul_add(ema[4], c3.mul_add(ema[3], c4 * ema[2]))),
                        );
                    }
                    _ => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        ema[4] = alpha.mul_add(ema[3] - ema[4], ema[4]);
                        ema[5] = alpha.mul_add(ema[4] - ema[5], ema[5]);
                        builder.append_value(
                            c1.mul_add(ema[5], c2.mul_add(ema[4], c3.mul_add(ema[3], c4 * ema[2]))),
                        );
                    }
                }
            }
        } else {
            for opt in arr.iter() {
                let Some(&val) = opt else {
                    builder.append_null();
                    continue;
                };
                count += 1;
                match count {
                    n if n < timeperiod => {
                        sum[0] += val;
                        builder.append_null();
                    }
                    n if n == timeperiod => {
                        sum[0] += val;
                        ema[0] = sum[0] / timeperiod as f64;
                        sum[1] = ema[0];
                        builder.append_null();
                    }
                    n if n < 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        builder.append_null();
                    }
                    n if n == 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        ema[1] = sum[1] / timeperiod as f64;
                        sum[2] = ema[1];
                        builder.append_null();
                    }
                    n if n < 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        builder.append_null();
                    }
                    n if n == 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        ema[2] = sum[2] / timeperiod as f64;
                        sum[3] = ema[2];
                        builder.append_null();
                    }
                    n if n < 4 * timeperiod - 3 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        sum[3] += ema[2];
                        builder.append_null();
                    }
                    n if n == 4 * timeperiod - 3 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        sum[3] += ema[2];
                        ema[3] = sum[3] / timeperiod as f64;
                        sum[4] = ema[3];
                        builder.append_null();
                    }
                    n if n < 5 * timeperiod - 4 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        sum[4] += ema[3];
                        builder.append_null();
                    }
                    n if n == 5 * timeperiod - 4 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        sum[4] += ema[3];
                        ema[4] = sum[4] / timeperiod as f64;
                        sum[5] = ema[4];
                        builder.append_null();
                    }
                    n if n < 6 * timeperiod - 5 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        ema[4] = alpha.mul_add(ema[3] - ema[4], ema[4]);
                        sum[5] += ema[4];
                        builder.append_null();
                    }
                    n if n == 6 * timeperiod - 5 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        ema[4] = alpha.mul_add(ema[3] - ema[4], ema[4]);
                        sum[5] += ema[4];
                        ema[5] = sum[5] / timeperiod as f64;
                        builder.append_value(
                            c1.mul_add(ema[5], c2.mul_add(ema[4], c3.mul_add(ema[3], c4 * ema[2]))),
                        );
                    }
                    _ => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        ema[3] = alpha.mul_add(ema[2] - ema[3], ema[3]);
                        ema[4] = alpha.mul_add(ema[3] - ema[4], ema[4]);
                        ema[5] = alpha.mul_add(ema[4] - ema[5], ema[5]);
                        builder.append_value(
                            c1.mul_add(ema[5], c2.mul_add(ema[4], c3.mul_add(ema[3], c4 * ema[2]))),
                        );
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
    let mut ema = ArrayVec::from([0.0f64; 3]);
    let mut sum = ArrayVec::from([0.0f64; 3]);

    values.downcast_iter().for_each(|arr| {
        if arr.null_count() == 0 {
            for &val in arr.values_iter() {
                count += 1;
                match count {
                    n if n < timeperiod => {
                        sum[0] += val;
                        builder.append_null();
                    }
                    n if n == timeperiod => {
                        sum[0] += val;
                        ema[0] = sum[0] / timeperiod as f64;
                        sum[1] = ema[0];
                        builder.append_null();
                    }
                    n if n < 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        builder.append_null();
                    }
                    n if n == 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        ema[1] = sum[1] / timeperiod as f64;
                        sum[2] = ema[1];
                        builder.append_null();
                    }
                    n if n < 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        builder.append_null();
                    }
                    n if n == 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        ema[2] = sum[2] / timeperiod as f64;
                        builder.append_value(3.0 * ema[0] - 3.0 * ema[1] + ema[2]);
                    }
                    _ => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        builder.append_value(3.0 * ema[0] - 3.0 * ema[1] + ema[2]);
                    }
                }
            }
        } else {
            for opt in arr.iter() {
                let Some(&val) = opt else {
                    builder.append_null();
                    continue;
                };
                count += 1;
                match count {
                    n if n < timeperiod => {
                        sum[0] += val;
                        builder.append_null();
                    }
                    n if n == timeperiod => {
                        sum[0] += val;
                        ema[0] = sum[0] / timeperiod as f64;
                        sum[1] = ema[0];
                        builder.append_null();
                    }
                    n if n < 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        builder.append_null();
                    }
                    n if n == 2 * timeperiod - 1 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        sum[1] += ema[0];
                        ema[1] = sum[1] / timeperiod as f64;
                        sum[2] = ema[1];
                        builder.append_null();
                    }
                    n if n < 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        builder.append_null();
                    }
                    n if n == 3 * timeperiod - 2 => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        sum[2] += ema[1];
                        ema[2] = sum[2] / timeperiod as f64;
                        builder.append_value(3.0 * ema[0] - 3.0 * ema[1] + ema[2]);
                    }
                    _ => {
                        ema[0] = alpha.mul_add(val - ema[0], ema[0]);
                        ema[1] = alpha.mul_add(ema[0] - ema[1], ema[1]);
                        ema[2] = alpha.mul_add(ema[1] - ema[2], ema[2]);
                        builder.append_value(3.0 * ema[0] - 3.0 * ema[1] + ema[2]);
                    }
                }
            }
        }
    });

    builder.finish()
}

pub fn calc_trima(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = (timeperiod + 1) / 2;

    if n == 0 || values.len() < timeperiod || values.null_count() != 0 {
        return Float64Chunked::full_null("trima".into(), values.len());
    }
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("trima".into(), values.len());

    let inverse_period = 1.0 / timeperiod as f64;
    let mut sma_1 = AllocRingBuffer::new(n);
    let mut sum_1: f64 = 0.;
    let mut sum_2: f64 = 0.;

    let mut values_iter = values.no_null_iter();

    for val in values_iter.by_ref().take(timeperiod - 1) {
        sum_1 = val.mul_add(inverse_period, sum_1);
        builder.append_null();
    }

    let val = values_iter.by_ref().next().unwrap();
    sum_1 = val.mul_add(inverse_period, sum_1);
    sma_1.enqueue(sum_1);
    builder.append_null();

    let mut values_iter = values_iter.zip(values.no_null_iter());

    for (new, old) in values_iter.by_ref().take(timeperiod - 1) {
        sum_1 = (new - old).mul_add(inverse_period, sum_1);
        sma_1.enqueue(sum_1);
        sum_2 = new.mul_add(inverse_period, sum_2);
        builder.append_null();
    }

    for (new, old) in values_iter {
        sum_1 = (new - old).mul_add(inverse_period, sum_1);
        sum_2 = (sum_1 - sma_1.enqueue(sum_1).unwrap()).mul_add(inverse_period, sum_2);
        builder.append_value(sum_2);
    }
    builder.finish()
}

pub fn calc_wma(values: &Float64Chunked, timeperiod: usize) -> Float64Chunked {
    let n = values.len();

    if timeperiod == 0 || n < timeperiod || values.null_count() != 0 {
        return Float64Chunked::full_null("wma".into(), n);
    }

    let values = values.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("wma".into(), n);

    let inverse_weights = 2.0 / (timeperiod * timeperiod + timeperiod) as f64;
    let mut old = AllocRingBuffer::new(timeperiod);
    let mut sum: f64 = 0.0;
    let mut sum_weight: f64 = 0.0;

    for (i, val) in values.no_null_iter().enumerate() {
        if i < timeperiod - 1 {
            sum += val;
            sum_weight += val * (i + 1) as f64;
            old.enqueue(val);
            builder.append_null();
        } else if i == timeperiod - 1 {
            sum += val;
            sum_weight += val * timeperiod as f64;
            old.enqueue(val);
            builder.append_value(sum_weight * inverse_weights);
        } else {
            sum_weight += val * timeperiod as f64;
            builder.append_value(sum_weight * inverse_weights);
            sum += val;
            sum_weight -= sum;
            sum -= old.enqueue(val).unwrap();
        }
    }

    builder.finish()
}
