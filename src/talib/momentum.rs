#![allow(clippy::needless_range_loop)]
use crate::talib::overlap::{calc_ema, calc_sma};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// ====================================================================
// Momentum Indicators - 动量指标 (Alphabetical Order)
// ====================================================================

#[polars_expr(output_type=Float64)]
pub fn adx(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let (dx_vec, _) = calc_dm(high, low, close, timeperiod);
    let adx = calc_rma(
        &dx_vec
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect::<Vec<f64>>(),
        timeperiod,
    );
    Ok(Series::new("adx".into(), adx))
}

#[polars_expr(output_type=Float64)]
pub fn adxr(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let (dx_vec, _) = calc_dm(high, low, close, timeperiod);
    let adx = calc_rma(
        &dx_vec
            .iter()
            .map(|v| v.unwrap_or(0.0))
            .collect::<Vec<f64>>(),
        timeperiod,
    );

    let n = adx.len();
    let mut adxr = vec![None; n];
    for i in (timeperiod - 1)..n {
        if let (Some(curr), Some(prev)) = (
            adx[i],
            adx.get(i.saturating_sub(timeperiod - 1)).and_then(|&v| v),
        ) {
            adxr[i] = Some((curr + prev) * 0.5);
        }
    }
    Ok(Series::new("adxr".into(), adxr))
}

fn aroon_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("aroon_up".into(), DataType::Float64);
    let f2 = Field::new("aroon_down".into(), DataType::Float64);
    Ok(Field::new("aroon".into(), DataType::Struct(vec![f1, f2])))
}

#[polars_expr(output_type_func=aroon_output)]
pub fn aroon(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let timeperiod = inputs[2].i64()?.get(0).unwrap_or(14) as usize;
    let n = high.len();

    let mut up = vec![None; n];
    let mut down = vec![None; n];

    for i in timeperiod..n {
        let start = i - timeperiod;
        let mut max_idx = 0;
        let mut max_val = f64::MIN;
        let mut min_idx = 0;
        let mut min_val = f64::MAX;

        for j in start..=i {
            if let Some(&v) = high.get(j) {
                if v >= max_val {
                    max_val = v;
                    max_idx = j - start;
                }
            }
            if let Some(&v) = low.get(j) {
                if v <= min_val {
                    min_val = v;
                    min_idx = j - start;
                }
            }
        }

        up[i] = Some((max_idx as f64 / timeperiod as f64) * 100.0);
        down[i] = Some((min_idx as f64 / timeperiod as f64) * 100.0);
    }

    let s1 = Series::new("aroon_up".into(), up);
    let s2 = Series::new("aroon_down".into(), down);
    Ok(StructChunked::from_series("aroon".into(), n, [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn bop(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();

    let open = open.f64()?.cont_slice()?;
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let close = close.f64()?.cont_slice()?;
    let n = open.len();

    let mut res = vec![None; n];
    for i in 0..n {
        if let (Some(o), Some(h), Some(l), Some(c)) =
            (open.get(i), high.get(i), low.get(i), close.get(i))
        {
            let diff = h - l;
            res[i] = Some(if diff == 0.0 { 0.0 } else { (c - o) / diff });
        }
    }
    Ok(Series::new("bop".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn cci(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let n = high.len();
    let mut tp = vec![None; n];
    for i in 0..n {
        if let (Some(&h), Some(&l), Some(&c)) = (high.get(i), low.get(i), close.get(i)) {
            tp[i] = Some((h + l + c) / 3.0);
        }
    }

    let sma_tp = calc_sma(
        &tp.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<f64>>(),
        timeperiod,
    );
    let mut res = vec![None; n];

    for i in (timeperiod - 1)..n {
        if let Some(avg) = sma_tp[i] {
            let mut mean_dev = 0.0;
            let mut count = 0;
            for j in (i + 1 - timeperiod)..=i {
                if let Some(val) = tp[j] {
                    mean_dev += (val - avg).abs();
                    count += 1;
                }
            }
            if count == timeperiod && mean_dev != 0.0 {
                mean_dev /= timeperiod as f64;
                res[i] = Some((tp[i].unwrap() - avg) / (0.015 * mean_dev));
            }
        }
    }
    Ok(Series::new("cci".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn cmo(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(14) as usize;

    let n = real.len();
    let mut ups = vec![0.0; n];
    let mut downs = vec![0.0; n];

    for i in 1..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - 1)) {
            let diff = curr - prev;
            if diff > 0.0 {
                ups[i] = diff;
            } else {
                downs[i] = -diff;
            }
        }
    }

    let mut res = vec![None; n];
    let mut sum_up = 0.0;
    let mut sum_down = 0.0;

    for i in 0..n {
        sum_up += ups[i];
        sum_down += downs[i];
        if i >= timeperiod {
            sum_up -= ups[i - timeperiod];
            sum_down -= downs[i - timeperiod];
        }

        if i >= timeperiod - 1 {
            let total = sum_up + sum_down;
            res[i] = Some(if total == 0.0 {
                0.0
            } else {
                100.0 * (sum_up - sum_down) / total
            });
        }
    }
    Ok(Series::new("cmo".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn dx(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let (dx_vec, _) = calc_dm(high, low, close, timeperiod);
    Ok(Series::new("dx".into(), dx_vec))
}

fn macd_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("macd".into(), DataType::Float64);
    let f2 = Field::new("macd_signal".into(), DataType::Float64);
    let f3 = Field::new("macd_hist".into(), DataType::Float64);
    Ok(Field::new(
        "macd_res".into(),
        DataType::Struct(vec![f1, f2, f3]),
    ))
}

#[polars_expr(output_type_func=macd_output)]
pub fn macd(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let fastperiod = inputs[1].i64()?.get(0).unwrap_or(12) as usize;
    let slowperiod = inputs[2].i64()?.get(0).unwrap_or(26) as usize;
    let signalperiod = inputs[3].i64()?.get(0).unwrap_or(9) as usize;

    let fast_ema = calc_ema(real, fastperiod);
    let slow_ema = calc_ema(real, slowperiod);

    let n = real.len();
    let mut dif = vec![None; n];
    for i in 0..n {
        if let (Some(f), Some(s)) = (fast_ema[i], slow_ema[i]) {
            dif[i] = Some(f - s);
        }
    }

    let dea = calc_ema(
        &dif.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<f64>>(),
        signalperiod,
    );
    let mut hist = vec![None; n];
    for i in 0..n {
        if let (Some(d), Some(s)) = (dif[i], dea[i]) {
            hist[i] = Some(d - s);
        }
    }

    let s1 = Series::new("macd".into(), dif);
    let s2 = Series::new("macd_signal".into(), dea);
    let s3 = Series::new("macd_hist".into(), hist);
    Ok(StructChunked::from_series("macd_res".into(), n, [s1, s2, s3].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn mfi(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let volume = inputs[3].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let close = close.f64()?.cont_slice()?;
    let volume = volume.f64()?.cont_slice()?;
    let timeperiod = inputs[4].i64()?.get(0).unwrap_or(14) as usize;

    let n = high.len();
    let mut tp = vec![0.0; n];
    let mut mf = vec![0.0; n];
    for i in 0..n {
        if let (Some(h), Some(l), Some(c), Some(v)) =
            (high.get(i), low.get(i), close.get(i), volume.get(i))
        {
            tp[i] = (h + l + c) / 3.0;
            mf[i] = tp[i] * v;
        }
    }

    let mut pos_mf_sum = 0.0;
    let mut neg_mf_sum = 0.0;
    let mut res = vec![None; n];

    for i in 1..n {
        if tp[i] > tp[i - 1] {
            pos_mf_sum += mf[i];
        } else if tp[i] < tp[i - 1] {
            neg_mf_sum += mf[i];
        }

        if i >= timeperiod {
            let prev_idx = i - timeperiod;
            if prev_idx > 0 {
                if tp[prev_idx] > tp[prev_idx - 1] {
                    pos_mf_sum -= mf[prev_idx];
                } else if tp[prev_idx] < tp[prev_idx - 1] {
                    neg_mf_sum -= mf[prev_idx];
                }
            }
        }

        if i >= timeperiod {
            if neg_mf_sum == 0.0 {
                res[i] = Some(100.0);
            } else {
                let mr = pos_mf_sum / neg_mf_sum;
                res[i] = Some(100.0 - (100.0 / (1.0 + mr)));
            }
        }
    }

    Ok(Series::new("mfi".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn minus_di(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let (_, minus_di) = calc_dm(high, low, close, timeperiod);
    Ok(Series::new("minus_di".into(), minus_di))
}

#[polars_expr(output_type=Float64)]
pub fn minus_dm(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let timeperiod = inputs[2].i64()?.get(0).unwrap_or(14) as usize;
    let n = high.len();

    let mut m_dm = vec![0.0; n];
    for i in 1..n {
        if let (Some(h), Some(l), Some(ph), Some(pl)) =
            (high.get(i), low.get(i), high.get(i - 1), low.get(i - 1))
        {
            let up_move = h - ph;
            let down_move = pl - l;
            if down_move > up_move && down_move > 0.0 {
                m_dm[i] = down_move;
            }
        }
    }
    let res = calc_rma(&m_dm, timeperiod);
    Ok(Series::new("minus_dm".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn mom(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(10) as usize;

    let n = real.len();
    let mut res = vec![None; n];
    for i in timeperiod..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - timeperiod)) {
            res[i] = Some(curr - prev);
        }
    }
    Ok(Series::new("mom".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn plus_di(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;

    let (plus_di, _) = calc_dm(high, low, close, timeperiod);
    Ok(Series::new("plus_di".into(), plus_di))
}

#[polars_expr(output_type=Float64)]
pub fn plus_dm(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let timeperiod = inputs[2].i64()?.get(0).unwrap_or(14) as usize;
    let n = high.len();

    let mut p_dm = vec![0.0; n];
    for i in 1..n {
        if let (Some(h), Some(l), Some(ph), Some(pl)) =
            (high.get(i), low.get(i), high.get(i - 1), low.get(i - 1))
        {
            let up_move = h - ph;
            let down_move = pl - l;
            if up_move > down_move && up_move > 0.0 {
                p_dm[i] = up_move;
            }
        }
    }
    let res = calc_rma(&p_dm, timeperiod);
    Ok(Series::new("plus_dm".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn roc(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(10) as usize;
    let n = real.len();
    let mut res = vec![None; n];
    for i in timeperiod..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - timeperiod)) {
            if prev != 0.0 {
                res[i] = Some((curr - prev) / prev * 100.0);
            }
        }
    }
    Ok(Series::new("roc".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn rocp(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(10) as usize;
    let n = real.len();
    let mut res = vec![None; n];
    for i in timeperiod..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - timeperiod)) {
            if prev != 0.0 {
                res[i] = Some((curr - prev) / prev);
            }
        }
    }
    Ok(Series::new("rocp".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn rocr(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(10) as usize;
    let n = real.len();
    let mut res = vec![None; n];
    for i in timeperiod..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - timeperiod)) {
            if prev != 0.0 {
                res[i] = Some(curr / prev);
            }
        }
    }
    Ok(Series::new("rocr".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn rocr100(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(10) as usize;
    let n = real.len();
    let mut res = vec![None; n];
    for i in timeperiod..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - timeperiod)) {
            if prev != 0.0 {
                res[i] = Some((curr / prev) * 100.0);
            }
        }
    }
    Ok(Series::new("rocr100".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn rsi(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(14) as usize;
    let n = real.len();

    let mut ups = vec![0.0; n];
    let mut downs = vec![0.0; n];
    for i in 1..n {
        if let (Some(&curr), Some(&prev)) = (real.get(i), real.get(i - 1)) {
            let diff = curr - prev;
            if diff > 0.0 {
                ups[i] = diff;
            } else {
                downs[i] = -diff;
            }
        }
    }

    let avg_up = calc_rma(&ups, timeperiod);
    let avg_down = calc_rma(&downs, timeperiod);

    let mut res = vec![None; n];
    for i in 0..n {
        if let (Some(u), Some(d)) = (avg_up[i], avg_down[i]) {
            if d == 0.0 {
                res[i] = Some(100.0);
            } else {
                let rs = u / d;
                res[i] = Some(100.0 - (100.0 / (1.0 + rs)));
            }
        }
    }
    Ok(Series::new("rsi".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn trix(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let timeperiod = inputs[1].i64()?.get(0).unwrap_or(30) as usize;

    let ema1 = calc_ema(real, timeperiod);
    let ema2 = calc_ema(
        &ema1.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<f64>>(),
        timeperiod,
    );
    let ema3 = calc_ema(
        &ema2.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<f64>>(),
        timeperiod,
    );

    let n = ema3.len();
    let mut res = vec![None; n];
    for i in 1..n {
        if let (Some(curr), Some(prev)) = (ema3[i], ema3[i - 1]) {
            if prev != 0.0 {
                res[i] = Some((curr - prev) / prev * 100.0);
            }
        }
    }
    Ok(Series::new("trix".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn ultosc(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let close = close.f64()?.cont_slice()?;

    let p1 = inputs[3].i64()?.get(0).unwrap_or(7) as usize;
    let p2 = inputs[4].i64()?.get(0).unwrap_or(14) as usize;
    let p3 = inputs[5].i64()?.get(0).unwrap_or(28) as usize;
    let n = high.len();

    let mut bp = vec![0.0; n];
    let mut tr = vec![0.0; n];
    for i in 1..n {
        if let (Some(h), Some(l), Some(c), Some(pc)) =
            (high.get(i), low.get(i), close.get(i), close.get(i - 1))
        {
            let min_l_pc = l.min(*pc);
            let max_h_pc = h.max(*pc);
            bp[i] = c - min_l_pc;
            tr[i] = max_h_pc - min_l_pc;
        }
    }

    fn avg(bp: &[f64], tr: &[f64], p: usize) -> Vec<Option<f64>> {
        let mut res = vec![None; bp.len()];
        let mut s_bp = 0.0;
        let mut s_tr = 0.0;
        for i in 0..bp.len() {
            s_bp += bp[i];
            s_tr += tr[i];
            if i >= p {
                s_bp -= bp[i - p];
                s_tr -= tr[i - p];
            }
            if i >= p - 1 && s_tr != 0.0 {
                res[i] = Some(s_bp / s_tr);
            }
        }
        res
    }

    let a1 = avg(&bp, &tr, p1);
    let a2 = avg(&bp, &tr, p2);
    let a3 = avg(&bp, &tr, p3);

    let mut res = vec![None; n];
    for i in 0..n {
        if let (Some(v1), Some(v2), Some(v3)) = (a1[i], a2[i], a3[i]) {
            res[i] = Some(100.0 * (4.0 * v1 + 2.0 * v2 + v3) / 7.0);
        }
    }
    Ok(Series::new("ultosc".into(), res))
}

#[polars_expr(output_type=Float64)]
pub fn willr(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?.rechunk();
    let low = inputs[1].cast(&DataType::Float64)?.rechunk();
    let close = inputs[2].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = low.f64()?.cont_slice()?;
    let close = close.f64()?.cont_slice()?;
    let timeperiod = inputs[3].i64()?.get(0).unwrap_or(14) as usize;
    let n = high.len();

    let mut res = vec![None; n];
    for i in (timeperiod - 1)..n {
        let mut max_h = f64::MIN;
        let mut min_l = f64::MAX;
        for j in (i + 1 - timeperiod)..=i {
            if let Some(&h) = high.get(j) {
                max_h = max_h.max(h);
            }
            if let Some(&l) = low.get(j) {
                min_l = min_l.min(l);
            }
        }
        if let Some(&c) = close.get(i) {
            let diff = max_h - min_l;
            res[i] = Some(if diff == 0.0 {
                0.0
            } else {
                -100.0 * (max_h - c) / diff
            });
        }
    }
    Ok(Series::new("willr".into(), res))
}

// ====================================================================
// Calculation Helpers
// ====================================================================

pub fn calc_dm(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = high.len();

    let mut p_dm = vec![0.0; n];
    let mut m_dm = vec![0.0; n];
    let mut tr = vec![0.0; n];

    for i in 1..n {
        if let (Some(h), Some(l), Some(ph), Some(pl), Some(pc)) = (
            high.get(i),
            low.get(i),
            high.get(i - 1),
            low.get(i - 1),
            close.get(i - 1),
        ) {
            let up_move = h - ph;
            let down_move = pl - l;

            if up_move > down_move && up_move > 0.0 {
                p_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                m_dm[i] = down_move;
            }
            tr[i] = (h - l).max((h - pc).abs()).max((l - pc).abs());
        }
    }

    let smooth_p_dm = calc_rma(&p_dm, timeperiod);
    let smooth_m_dm = calc_rma(&m_dm, timeperiod);
    let smooth_tr = calc_rma(&tr, timeperiod);

    let mut plus_di = vec![None; n];
    let mut minus_di = vec![None; n];
    for i in 0..n {
        if let (Some(p), Some(m), Some(t)) = (smooth_p_dm[i], smooth_m_dm[i], smooth_tr[i]) {
            if t != 0.0 {
                let p_di = 100.0 * p / t;
                let m_di = 100.0 * m / t;
                plus_di[i] = Some(p_di);
                minus_di[i] = Some(m_di);
            }
        }
    }

    let mut dx = vec![None; n];
    for i in 0..n {
        if let (Some(p), Some(m)) = (plus_di[i], minus_di[i]) {
            let diff = (p - m).abs();
            let sum = p + m;
            dx[i] = Some(if sum == 0.0 { 0.0 } else { 100.0 * diff / sum });
        }
    }
    (dx, minus_di)
}
