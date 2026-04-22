use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// ====================================================================
// Volume Indicators - 成交量指标 (Alphabetical Order)
// ====================================================================

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

    let n = high.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("ad".into(), n);
    let mut sum = 0.0f64;

    for (h_opt, l_opt, c_opt, v_opt) in izip!(high, low, close, volume) {
        match (h_opt, l_opt, c_opt, v_opt) {
            (Some(h), Some(l), Some(c), Some(v)) => {
                let diff = h - l;
                if diff != 0.0 {
                    sum += ((c + c - l - h) / diff) * v;
                }
                builder.append_value(sum);
            }
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn adosc(inputs: &[Series]) -> PolarsResult<Series> {
    // ADOSC - Chaikin A/D Oscillator (fast EMA of AD - slow EMA of AD).
    // Behavior:
    // - Build AD series similar to `ad`, but for EMA calculation we fill missing AD points with 0.0
    //   when calling the underlying calc_ema helper to keep alignment with prior behavior.
    // - The final output at each position is None if either EMA is None.
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;
    let volume = inputs[3].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;
    let volume = volume.f64()?;

    let fastperiod = inputs[4].i64()?.get(0).unwrap_or(3) as usize;
    let slowperiod = inputs[5].i64()?.get(0).unwrap_or(10) as usize;

    // Build AD as Vec<Option<f64>> preserving nulls
    let mut sum = 0.0f64;
    let mut ad_opt: Vec<Option<f64>> = Vec::with_capacity(high.len());
    for (h_opt, l_opt, c_opt, v_opt) in izip!(
        high.into_iter(),
        low.into_iter(),
        close.into_iter(),
        volume.into_iter()
    ) {
        match (h_opt, l_opt, c_opt, v_opt) {
            (Some(h), Some(l), Some(c), Some(v)) => {
                let diff = h - l;
                if diff != 0.0 {
                    sum += ((c + c - l - h) / diff) * v;
                }
                ad_opt.push(Some(sum));
            }
            _ => {
                // preserve running sum, emit Null
                ad_opt.push(None);
            }
        }
    }

    // Prepare filled AD values (None -> 0.0) for underlying EMA computation
    let ad_filled: Vec<f64> = ad_opt.iter().map(|v| v.unwrap_or(0.0)).collect();

    // Use existing overlap helpers which operate on Vec<f64> and return Vec<Option<f64>>
    let fast_ema = crate::talib::overlap::calc_ema(&ad_filled, fastperiod);
    let slow_ema = crate::talib::overlap::calc_ema(&ad_filled, slowperiod);

    // Subtract element-wise, preserving None if either is None
    let mut out: Vec<Option<f64>> = Vec::with_capacity(fast_ema.len());
    for (f_opt, s_opt) in izip!(fast_ema.into_iter(), slow_ema.into_iter()) {
        match (f_opt, s_opt) {
            (Some(fv), Some(sv)) => out.push(Some(fv - sv)),
            _ => out.push(None),
        }
    }

    Ok(Series::new("adosc".into(), out))
}

#[polars_expr(output_type=Float64)]
pub fn obv(inputs: &[Series]) -> PolarsResult<Series> {
    // OBV - On-Balance Volume
    // Behavior:
    // - If any of the involved values are null at a position, output is null at that position.
    // - The running OBV sum state is preserved across nulls (forward-hold).
    let real = inputs[0].cast(&DataType::Float64)?;
    let volume = inputs[1].cast(&DataType::Float64)?;

    let real = real.f64()?;
    let volume = volume.f64()?;

    let mut out: Vec<Option<f64>> = Vec::with_capacity(real.len());
    let mut sum = 0.0f64;

    // Handle first element specially and prepare iterators for the rest.
    // Create explicit first element values so we can continue using the same iterators
    // (avoids recreating temporaries that would lead to borrow issues).
    let mut iter_real = real.into_iter();
    let mut iter_volume = volume.into_iter();

    let first_price_opt = iter_real.next();
    let first_vol_opt = iter_volume.next();

    // prev_price holds the previous price for comparisons; initialize from first_price_opt
    let mut prev_price: Option<f64> = first_price_opt.unwrap_or(None);

    // Process first element: if both first price and volume present, set initial sum and emit, else emit None.
    match (first_price_opt, first_vol_opt) {
        (Some(Some(_p)), Some(Some(v))) => {
            sum = v;
            out.push(Some(sum));
        }
        _ => {
            out.push(None);
        }
    }

    // Process remaining elements using the iterators (they are already positioned at the second element).
    for (curr_opt, vol_opt) in izip!(iter_real, iter_volume) {
        match (prev_price, curr_opt, vol_opt) {
            (Some(prev), Some(curr), Some(v)) => {
                if curr > prev {
                    sum += v;
                } else if curr < prev {
                    sum -= v;
                }
                out.push(Some(sum));
            }
            _ => {
                // if any required value missing then output Null and do not change sum
                out.push(None);
            }
        }
        prev_price = curr_opt;
    }

    // If the series had length 0, ensure out length matches real.len()
    // The above logic produces len-1 entries for subsequent elements plus first; ensure alignment
    // However the construction keeps correct length for non-empty series.
    Ok(Series::new("obv".into(), out))
}
