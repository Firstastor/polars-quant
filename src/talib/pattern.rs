#![allow(unused_variables)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// ====================================================================
// Pattern Recognition - 模式识别
// ====================================================================

#[polars_expr(output_type=Int32)]
pub fn cdl2crows(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let bear2 = bear(o2, c2);
        let gap_up2 = o2 > c1;
        let bear3 = bear(o, c);
        let open_in2 = (o > o2) && (o < c2);
        let close_in1 = (c > o1) && (c < c1);
        let mask = ((((bull1 && bear2) && gap_up2) && bear3) && open_in2) && close_in1;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl2crows".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdl3blackcrows(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let bear2 = (bear(o2, c2)) && long_body(o2, c2);
        let bear3 = (bear(o, c)) && long_body(o, c);
        let opens_within1 = (o2 < o1) && (o2 > c1);
        let opens_within2 = (o < o2) && (o > c2);
        let lowercloses = (c2 < c1) && (c < c2);
        let mask = ((((bear1 && bear2) && bear3) && opens_within1) && opens_within2) && lowercloses;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl3blackcrows".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdl3inside(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull_pattern = ((((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o2, c2)))
            && (c2 < o1))
            && (o2 > c1))
            && (bull(o, c)))
            && (c > o1);
        let bear_pattern = ((((((bull(o1, c1)) && long_body(o1, c1)) && (bear(o2, c2)))
            && (o2 < c1))
            && (c2 > o1))
            && (bear(o, c)))
            && (c < o1);
        if bull_pattern {
            out[i] = 100;
        } else if bear_pattern {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl3inside".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdl3linestrike(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 3..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 3];
        let o2 = open[i - 2];
        let o3 = open[i - 1];
        let c1 = close[i - 3];
        let c2 = close[i - 2];
        let c3 = close[i - 1];
        let bull_three = ((((((((bear(o1, c1)) && (bear(o2, c2))) && (bear(o3, c3)))
            && (c2 < c1))
            && (c3 < c2))
            && (o2 > c1))
            && (o2 < o1))
            && (o3 > c2))
            && (o3 < o2);
        let bull_strike = ((bull(o, c)) && (o < c3)) && (c > o1);
        let bear_three = ((((((((bull(o1, c1)) && (bull(o2, c2))) && (bull(o3, c3)))
            && (c2 > c1))
            && (c3 > c2))
            && (o2 < c1))
            && (o2 > o1))
            && (o3 < c2))
            && (o3 > o2);
        let bear_strike = ((bear(o, c)) && (o > c3)) && (c < o1);
        if bull_three && bull_strike {
            out[i] = 100;
        } else if bear_three && bear_strike {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl3linestrike".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdl3outside(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull_pattern = (((((bear(o1, c1)) && (bull(o2, c2))) && (o2 <= c1)) && (c2 >= o1))
            && (bull(o, c)))
            && (c > c2);
        let bear_pattern = (((((bull(o1, c1)) && (bear(o2, c2))) && (o2 >= c1)) && (c2 <= o1))
            && (bear(o, c)))
            && (c < c2);
        if bull_pattern {
            out[i] = 100;
        } else if bear_pattern {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl3outside".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdl3starsinsouth(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let h2 = high[i - 1];
        let l1 = low[i - 2];
        let l2 = low[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let has_ls1 = long_dn_shadow(o1, l1, c1);
        let bear2 = bear(o2, c2);
        let lowerlow2 = l2 > l1;
        let higherclose2 = c2 > c1;
        let bear3 = (bear(o, c)) && short_body(o, c);
        let inside3 = (h < h2) && (l > l2);
        let mask =
            (((((bear1 && has_ls1) && bear2) && lowerlow2) && higherclose2) && bear3) && inside3;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl3starsinsouth".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdl3whitesoldiers(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let bull2 = (bull(o2, c2)) && long_body(o2, c2);
        let bull3 = (bull(o, c)) && long_body(o, c);
        let opens_within1 = (o2 > o1) && (o2 <= c1);
        let opens_within2 = (o > o2) && (o <= c2);
        let highercloses = (c2 > c1) && (c > c2);
        let mask =
            ((((bull1 && bull2) && bull3) && opens_within1) && opens_within2) && highercloses;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdl3whitesoldiers".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlabandonedbaby(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let h1 = high[i - 2];
        let h2 = high[i - 1];
        let l1 = low[i - 2];
        let l2 = low[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let doji2 = doji(o2, h2, l2, c2);
        let bull_pattern = (((((bear(o1, c1)) && long_body(o1, c1)) && doji2) && (h2 < l1))
            && (bull(o, c)))
            && (l > h2);
        let bear_pattern = (((((bull(o1, c1)) && long_body(o1, c1)) && doji2) && (l2 > h1))
            && (bear(o, c)))
            && (h < l2);
        if bull_pattern {
            out[i] = 100;
        } else if bear_pattern {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlabandonedbaby".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdladvanceblock(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let bull2 = bull(o2, c2);
        let bull3 = bull(o, c);
        let opens_within1 = (o2 > o1) && (o2 <= c1);
        let opens_within2 = (o > o2) && (o <= c2);
        let highercloses = (c2 > c1) && (c > c2);
        let shrinking = body_abs(o, c) < body_abs(o2, c2);
        let mask = (((((bull1 && bull2) && bull3) && opens_within1) && opens_within2)
            && highercloses)
            && shrinking;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdladvanceblock".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlbelthold(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let is_bull = ((bull(o, c)) && long_body(o, c)) && vshort_dn_shadow(o, h, l, c);
        let is_bear = ((bear(o, c)) && long_body(o, c)) && vshort_up_shadow(o, h, l, c);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlbelthold".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlbreakaway(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 4..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 4];
        let o2 = open[i - 3];
        let c1 = close[i - 4];
        let c2 = close[i - 3];
        let c3 = close[i - 2];
        let bull_pattern = (((((((bear(o1, c1)) && long_body(o1, c1)) && (bear(o2, c2)))
            && (o2 < c1))
            && (c3 < c2))
            && (bull(o, c)))
            && (c > o2))
            && (c < c1);
        let bear_pattern = (((((((bull(o1, c1)) && long_body(o1, c1)) && (bull(o2, c2)))
            && (o2 > c1))
            && (c3 > c2))
            && (bear(o, c)))
            && (c < o2))
            && (c > c1);
        if bull_pattern {
            out[i] = 100;
        } else if bear_pattern {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlbreakaway".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlclosingmarubozu(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let is_bull = ((bull(o, c)) && long_body(o, c)) && vshort_up_shadow(o, h, l, c);
        let is_bear = ((bear(o, c)) && long_body(o, c)) && vshort_dn_shadow(o, h, l, c);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlclosingmarubozu".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlconcealbabyswall(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 3..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 3];
        let o2 = open[i - 2];
        let o3 = open[i - 1];
        let h1 = high[i - 3];
        let h2 = high[i - 2];
        let h3 = high[i - 1];
        let l1 = low[i - 3];
        let l2 = low[i - 2];
        let c1 = close[i - 3];
        let c2 = close[i - 2];
        let c3 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let no_shadow1 = vshort_up_shadow(o1, h1, l1, c1) && vshort_dn_shadow(o1, h1, l1, c1);
        let bear2 = (bear(o2, c2)) && long_body(o2, c2);
        let no_shadow2 = vshort_up_shadow(o2, h2, l2, c2) && vshort_dn_shadow(o2, h2, l2, c2);
        let bear3 = bear(o3, c3);
        let high_gap3 = h3 > c2;
        let bear4 = (bear(o, c)) && long_body(o, c);
        let engulf = (o > h3) && (c < l2);
        let mask = (((((((bear1 && no_shadow1) && bear2) && no_shadow2) && (c2 < c1)) && bear3)
            && high_gap3)
            && bear4)
            && engulf;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlconcealbabyswall".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlcounterattack(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let is_bull = ((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o, c))) && long_body(o, c))
            && near(c, c1, h, l);
        let is_bear = ((((bull(o1, c1)) && long_body(o1, c1)) && (bear(o, c))) && long_body(o, c))
            && near(c, c1, h, l);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlcounterattack".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdldarkcloudcover(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let penetration = inputs
        .get(4)
        .and_then(|s| s.f64().ok()?.get(0))
        .unwrap_or(0.3);
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let bear_cur = bear(o, c);
        let open_above = o > c1;
        let close_into = c < (c1 - (body_abs(o1, c1) * penetration));
        let close_above = c > o1;
        let mask = (((bull1 && bear_cur) && open_above) && close_into) && close_above;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdldarkcloudcover".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdldoji(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = doji(o, h, l, c);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdldoji".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdldojistar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let doji_cur = doji(o, h, l, c);
        let cur_mid = (o + c) / 2.0;
        let is_bull = (((bear(o1, c1)) && long_body(o1, c1)) && doji_cur) && (cur_mid < c1);
        let is_bear = (((bull(o1, c1)) && long_body(o1, c1)) && doji_cur) && (cur_mid > c1);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdldojistar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdldragonflydoji(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = (doji(o, h, l, c) && long_dn_shadow(o, l, c)) && vshort_up_shadow(o, h, l, c);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdldragonflydoji".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlengulfing(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let is_bull = ((((bear(o1, c1)) && (bull(o, c))) && (o <= c1)) && (c >= o1))
            && ((o < c1) || (c > o1));
        let is_bear = ((((bull(o1, c1)) && (bear(o, c))) && (o >= c1)) && (c <= o1))
            && ((o > c1) || (c < o1));
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlengulfing".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdleveningdojistar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let penetration = inputs
        .get(4)
        .and_then(|s| s.f64().ok()?.get(0))
        .unwrap_or(0.3);
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let h2 = high[i - 1];
        let l2 = low[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let doji2 = doji(o2, h2, l2, c2);
        let gap_up = oc_min(o2, c2) > c1;
        let bear3 = bear(o, c);
        let close_into = c < (c1 - (body_abs(o1, c1) * penetration));
        let mask = (((bull1 && doji2) && gap_up) && bear3) && close_into;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdleveningdojistar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdleveningstar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let penetration = inputs
        .get(4)
        .and_then(|s| s.f64().ok()?.get(0))
        .unwrap_or(0.3);
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let short2 = short_body(o2, c2);
        let gap_up = oc_min(o2, c2) > c1;
        let bear3 = bear(o, c);
        let close_into = c < (c1 - (body_abs(o1, c1) * penetration));
        let mask = (((bull1 && short2) && gap_up) && bear3) && close_into;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdleveningstar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlgapsidesidewhite(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull2 = bull(o2, c2);
        let bull3 = bull(o, c);
        let similar_size = near(body_abs(o, c), body_abs(o2, c2), h, l);
        let similaropen = near(o, o2, h, l);
        let up_gap =
            (((((bull(o1, c1)) && (o2 > c1)) && bull2) && bull3) && similar_size) && similaropen;
        let down_gap =
            (((((bear(o1, c1)) && (c2 < c1)) && bull2) && bull3) && similar_size) && similaropen;
        if up_gap {
            out[i] = 100;
        } else if down_gap {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlgapsidesidewhite".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlgravestonedoji(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = (doji(o, h, l, c) && long_up_shadow(o, h, c)) && vshort_dn_shadow(o, h, l, c);
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlgravestonedoji".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlhammer(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let ba = body_abs(o, c);
        let ls = lower_shadow(o, l, c);
        let mask = (short_body(o, c) && (ls > (2.0 * ba))) && vshort_up_shadow(o, h, l, c);
        let downtrend = bear(o1, c1);
        if mask && downtrend {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlhammer".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlhangingman(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let ba = body_abs(o, c);
        let ls = lower_shadow(o, l, c);
        let mask = (short_body(o, c) && (ls > (2.0 * ba))) && vshort_up_shadow(o, h, l, c);
        let uptrend = bull(o1, c1);
        if mask && uptrend {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlhangingman".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlharami(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let is_bull = (((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o, c)))
            && short_body(o, c))
            && (o > c1))
            && (c < o1);
        let is_bear = (((((bull(o1, c1)) && long_body(o1, c1)) && (bear(o, c)))
            && short_body(o, c))
            && (o < c1))
            && (c > o1);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlharami".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlharamicross(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let cur_doji = doji(o, h, l, c);
        let is_bull = ((((bear(o1, c1)) && long_body(o1, c1)) && cur_doji) && (oc_max(o, c) < o1))
            && (oc_min(o, c) > c1);
        let is_bear = ((((bull(o1, c1)) && long_body(o1, c1)) && cur_doji) && (oc_max(o, c) < c1))
            && (oc_min(o, c) > o1);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlharamicross".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlhighwave(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = (short_body(o, c) && long_up_shadow(o, h, c)) && long_dn_shadow(o, l, c);
        if mask && (bull(o, c)) {
            out[i] = 100;
        } else if mask && (bear(o, c)) {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlhighwave".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlhikkake(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let h1 = high[i - 2];
        let h2 = high[i - 1];
        let l1 = low[i - 2];
        let l2 = low[i - 1];
        let inside_bar = (h2 < h1) && (l2 > l1);
        let is_bull = (inside_bar && (c > h1)) && (bull(o, c));
        let is_bear = (inside_bar && (c < l1)) && (bear(o, c));
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlhikkake".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlhikkakemod(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 3..n {
        let o = open[i];
        let c = close[i];
        let h1 = high[i - 3];
        let h2 = high[i - 2];
        let h3 = high[i - 1];
        let l1 = low[i - 3];
        let l2 = low[i - 2];
        let l3 = low[i - 1];
        let inside_bar = (h2 < h1) && (l2 > l1);
        let second_inside = (h3 < h2) && (l3 > l2);
        let is_bull = ((inside_bar && second_inside) && (c > h1)) && (bull(o, c));
        let is_bear = ((inside_bar && second_inside) && (c < l1)) && (bear(o, c));
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlhikkakemod".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlhomingpigeon(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let mask = (((((bear(o1, c1)) && long_body(o1, c1)) && (bear(o, c))) && short_body(o, c))
            && (o < o1))
            && (c > c1);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlhomingpigeon".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlidentical3crows(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let bear2 = (bear(o2, c2)) && long_body(o2, c2);
        let bear3 = (bear(o, c)) && long_body(o, c);
        let eqopen1 = equal(o2, c1, h, l);
        let eqopen2 = equal(o, c2, h, l);
        let lowercloses = (c2 < c1) && (c < c2);
        let mask = ((((bear1 && bear2) && bear3) && eqopen1) && eqopen2) && lowercloses;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlidentical3crows".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlinneck(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let mask = ((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o, c))) && (o < c1))
            && near(c, c1, h, l);
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlinneck".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlinvertedhammer(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let ba = body_abs(o, c);
        let us = upper_shadow(o, h, c);
        let mask = (short_body(o, c) && (us > (2.0 * ba))) && vshort_dn_shadow(o, h, l, c);
        let downtrend = bear(o1, c1);
        if mask && downtrend {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlinvertedhammer".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlkicking(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let h1 = high[i - 1];
        let l1 = low[i - 1];
        let c1 = close[i - 1];
        let marubozu1_bear = (((bear(o1, c1)) && long_body(o1, c1))
            && vshort_up_shadow(o1, h1, l1, c1))
            && vshort_dn_shadow(o1, h1, l1, c1);
        let marubozu1_bull = (((bull(o1, c1)) && long_body(o1, c1))
            && vshort_up_shadow(o1, h1, l1, c1))
            && vshort_dn_shadow(o1, h1, l1, c1);
        let marubozu_cur_bull = (((bull(o, c)) && long_body(o, c)) && vshort_up_shadow(o, h, l, c))
            && vshort_dn_shadow(o, h, l, c);
        let marubozu_cur_bear = (((bear(o, c)) && long_body(o, c)) && vshort_up_shadow(o, h, l, c))
            && vshort_dn_shadow(o, h, l, c);
        let is_bull = (marubozu1_bear && marubozu_cur_bull) && (o > o1);
        let is_bear = (marubozu1_bull && marubozu_cur_bear) && (o < o1);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlkicking".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlkickingbylength(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let h1 = high[i - 1];
        let l1 = low[i - 1];
        let c1 = close[i - 1];
        let marubozu1_bear = (((bear(o1, c1)) && long_body(o1, c1))
            && vshort_up_shadow(o1, h1, l1, c1))
            && vshort_dn_shadow(o1, h1, l1, c1);
        let marubozu1_bull = (((bull(o1, c1)) && long_body(o1, c1))
            && vshort_up_shadow(o1, h1, l1, c1))
            && vshort_dn_shadow(o1, h1, l1, c1);
        let marubozu_cur_bull = (((bull(o, c)) && long_body(o, c)) && vshort_up_shadow(o, h, l, c))
            && vshort_dn_shadow(o, h, l, c);
        let marubozu_cur_bear = (((bear(o, c)) && long_body(o, c)) && vshort_up_shadow(o, h, l, c))
            && vshort_dn_shadow(o, h, l, c);
        let ba1 = body_abs(o1, c1);
        let ba0 = body_abs(o, c);
        let bull_kick = (marubozu1_bear && marubozu_cur_bull) && (o > o1);
        let bear_kick = (marubozu1_bull && marubozu_cur_bear) && (o < o1);
        let bull_longer = bull_kick && (ba0 >= ba1);
        let bear_longer = bear_kick && (ba0 >= ba1);
        if bull_longer || (bull_kick && !bear_longer) {
            out[i] = 100;
        } else if bear_longer || (bear_kick && !bull_longer) {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlkickingbylength".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlladderbottom(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 4..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 4];
        let o2 = open[i - 3];
        let o3 = open[i - 2];
        let o4 = open[i - 1];
        let h4 = high[i - 1];
        let c1 = close[i - 4];
        let c2 = close[i - 3];
        let c3 = close[i - 2];
        let c4 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let bear2 = (bear(o2, c2)) && (c2 < c1);
        let bear3 = (bear(o3, c3)) && (c3 < c2);
        let bear4 = bear(o4, c4);
        let has_upper4 = long_up_shadow(o4, h4, c4);
        let bull5 = (bull(o, c)) && (o > o4);
        let mask = ((((bear1 && bear2) && bear3) && bear4) && has_upper4) && bull5;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlladderbottom".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdllongleggeddoji(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = (doji(o, h, l, c) && long_up_shadow(o, h, c)) && long_dn_shadow(o, l, c);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdllongleggeddoji".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdllongline(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let long_body = long_body(o, c);
        let short_shadows = short_up_shadow(o, h, l, c) && short_dn_shadow(o, h, l, c);
        let mask = long_body && short_shadows;
        if mask && (bull(o, c)) {
            out[i] = 100;
        } else if mask && (bear(o, c)) {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdllongline".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlmarubozu(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask =
            (long_body(o, c) && vshort_up_shadow(o, h, l, c)) && vshort_dn_shadow(o, h, l, c);
        if mask && (bull(o, c)) {
            out[i] = 100;
        } else if mask && (bear(o, c)) {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlmarubozu".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlmatchinglow(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let mask = (((bear(o1, c1)) && long_body(o1, c1)) && (bear(o, c))) && equal(c, c1, h, l);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlmatchinglow".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlmathold(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 4..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 4];
        let o2 = open[i - 3];
        let o3 = open[i - 2];
        let o4 = open[i - 1];
        let l2 = low[i - 3];
        let l3 = low[i - 2];
        let l4 = low[i - 1];
        let c1 = close[i - 4];
        let c2 = close[i - 3];
        let c3 = close[i - 2];
        let c4 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let small2 = short_body(o2, c2) && (o2 > c1);
        let small3 = short_body(o3, c3);
        let small4 = short_body(o4, c4);
        let hold_above = ((l2 > o1) && (l3 > o1)) && (l4 > o1);
        let bull5 = (bull(o, c)) && (c > c1);
        let mask = ((((bull1 && small2) && small3) && small4) && hold_above) && bull5;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlmathold".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlmorningdojistar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let penetration = inputs
        .get(4)
        .and_then(|s| s.f64().ok()?.get(0))
        .unwrap_or(0.3);
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let h2 = high[i - 1];
        let l2 = low[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let doji2 = doji(o2, h2, l2, c2);
        let gap_down = oc_max(o2, c2) < c1;
        let bull3 = bull(o, c);
        let close_into = c > (c1 + (body_abs(o1, c1) * penetration));
        let mask = (((bear1 && doji2) && gap_down) && bull3) && close_into;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlmorningdojistar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlmorningstar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let penetration = inputs
        .get(4)
        .and_then(|s| s.f64().ok()?.get(0))
        .unwrap_or(0.3);
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let short2 = short_body(o2, c2);
        let gap_down = oc_max(o2, c2) < c1;
        let bull3 = bull(o, c);
        let close_into = c > (c1 + (body_abs(o1, c1) * penetration));
        let mask = (((bear1 && short2) && gap_down) && bull3) && close_into;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlmorningstar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlonneck(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let l1 = low[i - 1];
        let c1 = close[i - 1];
        let mask = ((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o, c))) && (o < c1))
            && near(c, l1, h, l);
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlonneck".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlpiercing(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let penetration = inputs
        .get(4)
        .and_then(|s| s.f64().ok()?.get(0))
        .unwrap_or(0.3);
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let bull_cur = bull(o, c);
        let open_below = o < c1;
        let close_into = c > (c1 + (body_abs(o1, c1) * penetration));
        let close_below = c < o1;
        let mask = (((bear1 && bull_cur) && open_below) && close_into) && close_below;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlpiercing".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlrickshawman(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        let mask = ((doji(o, h, l, c) && long_up_shadow(o, h, c)) && long_dn_shadow(o, l, c))
            && near(us, ls, h, l);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlrickshawman".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlrisefall3methods(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 4..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 4];
        let o2 = open[i - 3];
        let o3 = open[i - 2];
        let o4 = open[i - 1];
        let h1 = high[i - 4];
        let h2 = high[i - 3];
        let h3 = high[i - 2];
        let h4 = high[i - 1];
        let l1 = low[i - 4];
        let l2 = low[i - 3];
        let l3 = low[i - 2];
        let l4 = low[i - 1];
        let c1 = close[i - 4];
        let c2 = close[i - 3];
        let c3 = close[i - 2];
        let c4 = close[i - 1];
        let rising = (((((((((((((bull(o1, c1)) && long_body(o1, c1))
            && short_body(o2, c2))
            && short_body(o3, c3))
            && short_body(o4, c4))
            && (h2 < h1))
            && (h3 < h1))
            && (h4 < h1))
            && (l2 > l1))
            && (l3 > l1))
            && (l4 > l1))
            && (bull(o, c)))
            && long_body(o, c))
            && (c > c1);
        let falling = (((((((((((((bear(o1, c1)) && long_body(o1, c1))
            && short_body(o2, c2))
            && short_body(o3, c3))
            && short_body(o4, c4))
            && (l2 > l1))
            && (l3 > l1))
            && (l4 > l1))
            && (h2 < h1))
            && (h3 < h1))
            && (h4 < h1))
            && (bear(o, c)))
            && long_body(o, c))
            && (c < c1);
        if rising {
            out[i] = 100;
        } else if falling {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlrisefall3methods".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlseparatinglines(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let is_bull = ((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o, c))) && long_body(o, c))
            && equal(o, o1, h, l);
        let is_bear = ((((bull(o1, c1)) && long_body(o1, c1)) && (bear(o, c))) && long_body(o, c))
            && equal(o, o1, h, l);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlseparatinglines".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlshootingstar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let ba = body_abs(o, c);
        let us = upper_shadow(o, h, c);
        let mask = (short_body(o, c) && (us > (2.0 * ba))) && vshort_dn_shadow(o, h, l, c);
        let uptrend = bull(o1, c1);
        if mask && uptrend {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlshootingstar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlshortline(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let short_body = short_body(o, c);
        let short_shadows = short_up_shadow(o, h, l, c) && short_dn_shadow(o, h, l, c);
        let mask = short_body && short_shadows;
        if mask && (bull(o, c)) {
            out[i] = 100;
        } else if mask && (bear(o, c)) {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlshortline".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlspinningtop(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = (short_body(o, c) && (upper_shadow(o, h, c) > body_abs(o, c)))
            && (lower_shadow(o, l, c) > body_abs(o, c));
        if mask && (bull(o, c)) {
            out[i] = 100;
        } else if mask && (bear(o, c)) {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlspinningtop".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlstalledpattern(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let bull2 = ((bull(o2, c2)) && long_body(o2, c2)) && (c2 > c1);
        let bull3 = ((bull(o, c)) && short_body(o, c)) && (c > c2);
        let opens_near = (o > o2) && (o <= c2);
        let mask = ((bull1 && bull2) && bull3) && opens_near;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlstalledpattern".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlsticksandwich(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let mask = (((((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o2, c2)))
            && long_body(o2, c2))
            && (o2 > c1))
            && (bear(o, c)))
            && long_body(o, c))
            && equal(c, c1, h, l);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlsticksandwich".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdltakuri(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 0..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let mask = (doji(o, h, l, c) && vlong_dn_shadow(o, l, c)) && vshort_up_shadow(o, h, l, c);
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdltakuri".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdltasukigap(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let is_bull = (((((((bull(o1, c1)) && (bull(o2, c2))) && (o2 > c1)) && (bear(o, c)))
            && (o > o2))
            && (o < c2))
            && (c > o1))
            && (c < c1);
        let is_bear = (((((((bear(o1, c1)) && (bear(o2, c2))) && (o2 < c1)) && (bull(o, c)))
            && (o < o2))
            && (o > c2))
            && (c < o1))
            && (c > c1);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdltasukigap".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlthrusting(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 1..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 1];
        let c1 = close[i - 1];
        let midpoint = c1 + (body_abs(o1, c1) * 0.5);
        let mask = (((((bear(o1, c1)) && long_body(o1, c1)) && (bull(o, c))) && (o < c1))
            && (c > c1))
            && (c < midpoint);
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlthrusting".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdltristar(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let h = high[i];
        let l = low[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let h1 = high[i - 2];
        let h2 = high[i - 1];
        let l1 = low[i - 2];
        let l2 = low[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let doji1 = doji(o1, h1, l1, c1);
        let doji2 = doji(o2, h2, l2, c2);
        let doji3 = doji(o, h, l, c);
        let mid1 = (o1 + c1) / 2.0;
        let mid2 = (o2 + c2) / 2.0;
        let mid3 = (o + c) / 2.0;
        let is_bull = (((doji1 && doji2) && doji3) && (mid2 < mid1)) && (mid3 > mid2);
        let is_bear = (((doji1 && doji2) && doji3) && (mid2 > mid1)) && (mid3 < mid2);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdltristar".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlunique3river(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let l1 = low[i - 2];
        let l2 = low[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bear1 = (bear(o1, c1)) && long_body(o1, c1);
        let bear2 = ((bear(o2, c2)) && (l2 < l1)) && (c2 > l2);
        let harami = (o2 < o1) && (o2 > c1);
        let bull3 = ((bull(o, c)) && short_body(o, c)) && (c < c2);
        let mask = ((bear1 && bear2) && harami) && bull3;
        if mask {
            out[i] = 100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlunique3river".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlupsidegap2crows(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let bull1 = (bull(o1, c1)) && long_body(o1, c1);
        let bear2 = ((bear(o2, c2)) && (o2 > c1)) && (c2 > c1);
        let bear3 = (((bear(o, c)) && (o > o2)) && (c > c1)) && (c < c2);
        let mask = (bull1 && bear2) && bear3;
        if mask {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlupsidegap2crows".into(), &out).into_series())
}

#[polars_expr(output_type=Int32)]
pub fn cdlxsidegap3methods(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?.rechunk();
    let open = open.f64()?.cont_slice()?;
    let high = inputs[1].cast(&DataType::Float64)?.rechunk();
    let high = high.f64()?.cont_slice()?;
    let low = inputs[2].cast(&DataType::Float64)?.rechunk();
    let low = low.f64()?.cont_slice()?;
    let close = inputs[3].cast(&DataType::Float64)?.rechunk();
    let close = close.f64()?.cont_slice()?;
    let n = open.len();
    let mut out = vec![0i32; n];
    for i in 2..n {
        let o = open[i];
        let c = close[i];
        let o1 = open[i - 2];
        let o2 = open[i - 1];
        let c1 = close[i - 2];
        let c2 = close[i - 1];
        let is_bull = (((((((bull(o1, c1)) && (bull(o2, c2))) && (o2 > c1)) && (bear(o, c)))
            && (o < c2))
            && (o > o2))
            && (c > o1))
            && (c < c1);
        let is_bear = (((((((bear(o1, c1)) && (bear(o2, c2))) && (o2 < c1)) && (bull(o, c)))
            && (o > c2))
            && (o < o2))
            && (c < o1))
            && (c > c1);
        if is_bull {
            out[i] = 100;
        } else if is_bear {
            out[i] = -100;
        }
    }
    Ok(Int32Chunked::from_slice("cdlxsidegap3methods".into(), &out).into_series())
}

// --- Auxiliary Functions ---

// --- Auxiliary Functions ---
#[inline(always)]
fn bull(o: f64, c: f64) -> bool {
    c > o
}
#[inline(always)]
fn bear(o: f64, c: f64) -> bool {
    c < o
}

#[inline(always)]
fn body_abs(o: f64, c: f64) -> f64 {
    (o - c).abs()
}
#[inline(always)]
fn oc_min(o: f64, c: f64) -> f64 {
    o.min(c)
}
#[inline(always)]
fn oc_max(o: f64, c: f64) -> f64 {
    o.max(c)
}
#[inline(always)]
fn upper_shadow(o: f64, h: f64, c: f64) -> f64 {
    h - oc_max(o, c)
}
#[inline(always)]
fn lower_shadow(o: f64, l: f64, c: f64) -> f64 {
    oc_min(o, c) - l
}
#[inline(always)]
fn long_body(o: f64, c: f64) -> bool {
    body_abs(o, c) > 0.05 * (o + c) * 0.5
}
#[inline(always)]
fn short_body(o: f64, c: f64) -> bool {
    body_abs(o, c) < 0.1 * (o + c) * 0.5
}
#[inline(always)]
fn doji(o: f64, _h: f64, _l: f64, c: f64) -> bool {
    body_abs(o, c) <= 0.005 * (o + c) * 0.5
}
#[inline(always)]
fn long_up_shadow(o: f64, h: f64, c: f64) -> bool {
    upper_shadow(o, h, c) > 2.0 * body_abs(o, c)
}
#[inline(always)]
fn long_dn_shadow(o: f64, l: f64, c: f64) -> bool {
    lower_shadow(o, l, c) > 2.0 * body_abs(o, c)
}
#[inline(always)]
fn short_up_shadow(o: f64, h: f64, _l: f64, c: f64) -> bool {
    upper_shadow(o, h, c) < 0.5 * body_abs(o, c)
}
#[inline(always)]
fn short_dn_shadow(o: f64, _h: f64, l: f64, c: f64) -> bool {
    lower_shadow(o, l, c) < 0.5 * body_abs(o, c)
}
#[inline(always)]
fn vshort_up_shadow(o: f64, h: f64, _l: f64, c: f64) -> bool {
    upper_shadow(o, h, c) < 0.1 * body_abs(o, c)
}
#[inline(always)]
fn vshort_dn_shadow(o: f64, _h: f64, l: f64, c: f64) -> bool {
    lower_shadow(o, l, c) < 0.1 * body_abs(o, c)
}
#[inline(always)]
fn vlong_dn_shadow(o: f64, l: f64, c: f64) -> bool {
    lower_shadow(o, l, c) > 3.0 * body_abs(o, c)
}
#[inline(always)]
fn near(v1: f64, v2: f64, h: f64, l: f64) -> bool {
    (v1 - v2).abs() < 0.01 * (h + l) * 0.5
}
#[inline(always)]
fn equal(v1: f64, v2: f64, h: f64, l: f64) -> bool {
    (v1 - v2).abs() < 0.001 * (h + l) * 0.5
}
