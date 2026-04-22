use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::f64::consts::{PI, TAU};

// ====================================================================
// Cycle Indicators - 周期指标 (Alphabetical Order)
// ====================================================================

#[polars_expr(output_type=Float64)]
pub fn ht_dcperiod(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let n = real.len();

    let mut out = vec![None; n];
    if n < 32 {
        return Ok(Series::new("ht_dcperiod".into(), out));
    }

    let smooth = calc_smooth(real);
    let mut detrend = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut i1 = [0.0; 7];
    let (mut i2, mut q2, mut re, mut im, mut period) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut smooth_period = 0.0;

    for i in 6..n {
        let prev_period = if i > 6 { period } else { 6.0 };
        let adj = 0.075 * prev_period + 0.54;

        let detrend_curr = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * adj;
        shift_push7(&mut detrend, detrend_curr);

        let q1_curr =
            (0.0962 * detrend[0] + 0.5769 * detrend[2] - 0.5769 * detrend[4] - 0.0962 * detrend[6])
                * adj;
        shift_push7(&mut q1, q1_curr);
        shift_push7(&mut i1, detrend[3]);

        let ji = (0.0962 * i1[0] + 0.5769 * i1[2] - 0.5769 * i1[4] - 0.0962 * i1[6]) * adj;
        let jq = (0.0962 * q1[0] + 0.5769 * q1[2] - 0.5769 * q1[4] - 0.0962 * q1[6]) * adj;

        let i2_curr = 0.2 * (i1[0] - jq) + 0.8 * i2;
        let q2_curr = 0.2 * (q1[0] + ji) + 0.8 * q2;

        let re_curr = 0.2 * (i2_curr * i2 + q2_curr * q2) + 0.8 * re;
        let im_curr = 0.2 * (i2_curr * q2 - q2_curr * i2) + 0.8 * im;

        i2 = i2_curr;
        q2 = q2_curr;
        re = re_curr;
        im = im_curr;

        if im != 0.0 && re != 0.0 {
            period = TAU / (im / re).atan();
        }
        period = period
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period = 0.2 * period + 0.8 * prev_period;
        smooth_period = 0.33 * period + 0.67 * smooth_period;

        if i >= 31 {
            out[i] = Some(smooth_period);
        }
    }

    Ok(Series::new("ht_dcperiod".into(), out))
}

#[polars_expr(output_type=Float64)]
pub fn ht_dcphase(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let n = real.len();

    let mut out = vec![None; n];
    if n < 32 {
        return Ok(Series::new("ht_dcphase".into(), out));
    }

    let smooth = calc_smooth(real);
    let mut detrend = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut i1 = [0.0; 7];
    let (mut i2, mut q2, mut re, mut im, mut period) = (0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 6..n {
        let prev_period = if i > 6 { period } else { 6.0 };
        let adj = 0.075 * prev_period + 0.54;

        let detrend_curr = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * adj;
        shift_push7(&mut detrend, detrend_curr);

        let q1_curr =
            (0.0962 * detrend[0] + 0.5769 * detrend[2] - 0.5769 * detrend[4] - 0.0962 * detrend[6])
                * adj;
        shift_push7(&mut q1, q1_curr);
        shift_push7(&mut i1, detrend[3]);

        let ji = (0.0962 * i1[0] + 0.5769 * i1[2] - 0.5769 * i1[4] - 0.0962 * i1[6]) * adj;
        let jq = (0.0962 * q1[0] + 0.5769 * q1[2] - 0.5769 * q1[4] - 0.0962 * q1[6]) * adj;

        let i2_curr = 0.2 * (i1[0] - jq) + 0.8 * i2;
        let q2_curr = 0.2 * (q1[0] + ji) + 0.8 * q2;

        let re_curr = 0.2 * (i2_curr * i2 + q2_curr * q2) + 0.8 * re;
        let im_curr = 0.2 * (i2_curr * q2 - q2_curr * i2) + 0.8 * im;

        i2 = i2_curr;
        q2 = q2_curr;
        re = re_curr;
        im = im_curr;

        if im != 0.0 && re != 0.0 {
            period = TAU / (im / re).atan();
        }
        period = period
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period = 0.2 * period + 0.8 * prev_period;

        if i >= 31 {
            let mut dc_phase = if i1[0] != 0.0 {
                (q1[0] / i1[0]).atan() * 180.0 / PI
            } else {
                0.0
            };
            dc_phase += 90.0;
            if i1[0] < 0.0 {
                dc_phase += 180.0;
            }
            if dc_phase > 315.0 {
                dc_phase -= 360.0;
            }
            out[i] = Some(dc_phase);
        }
    }

    Ok(Series::new("ht_dcphase".into(), out))
}

fn ht_phasor_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("inphase".into(), DataType::Float64);
    let f2 = Field::new("quadrature".into(), DataType::Float64);
    Ok(Field::new(
        "ht_phasor".into(),
        DataType::Struct(vec![f1, f2]),
    ))
}

#[polars_expr(output_type_func=ht_phasor_output)]
pub fn ht_phasor(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let n = real.len();

    let mut inphase = vec![None; n];
    let mut quadrature = vec![None; n];
    if n < 32 {
        let s1 = Series::new("inphase".into(), inphase);
        let s2 = Series::new("quadrature".into(), quadrature);
        return Ok(
            StructChunked::from_series("ht_phasor".into(), n, [s1, s2].iter())?.into_series(),
        );
    }

    let smooth = calc_smooth(real);
    let mut detrend = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut i1 = [0.0; 7];
    let (mut i2, mut q2, mut re, mut im, mut period) = (0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 6..n {
        let prev_period = if i > 6 { period } else { 6.0 };
        let adj = 0.075 * prev_period + 0.54;

        let detrend_curr = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * adj;
        shift_push7(&mut detrend, detrend_curr);

        let q1_curr =
            (0.0962 * detrend[0] + 0.5769 * detrend[2] - 0.5769 * detrend[4] - 0.0962 * detrend[6])
                * adj;
        shift_push7(&mut q1, q1_curr);
        shift_push7(&mut i1, detrend[3]);

        let ji = (0.0962 * i1[0] + 0.5769 * i1[2] - 0.5769 * i1[4] - 0.0962 * i1[6]) * adj;
        let jq = (0.0962 * q1[0] + 0.5769 * q1[2] - 0.5769 * q1[4] - 0.0962 * q1[6]) * adj;

        let i2_curr = 0.2 * (i1[0] - jq) + 0.8 * i2;
        let q2_curr = 0.2 * (q1[0] + ji) + 0.8 * q2;

        let re_curr = 0.2 * (i2_curr * i2 + q2_curr * q2) + 0.8 * re;
        let im_curr = 0.2 * (i2_curr * q2 - q2_curr * i2) + 0.8 * im;

        i2 = i2_curr;
        q2 = q2_curr;
        re = re_curr;
        im = im_curr;

        if im != 0.0 && re != 0.0 {
            period = TAU / (im / re).atan();
        }
        period = period
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period = 0.2 * period + 0.8 * prev_period;

        if i >= 31 {
            inphase[i] = Some(i1[0]);
            quadrature[i] = Some(q1[0]);
        }
    }

    let s1 = Series::new("inphase".into(), inphase);
    let s2 = Series::new("quadrature".into(), quadrature);
    Ok(StructChunked::from_series("ht_phasor".into(), n, [s1, s2].iter())?.into_series())
}

fn ht_sine_output(_: &[Field]) -> PolarsResult<Field> {
    let f1 = Field::new("sine".into(), DataType::Float64);
    let f2 = Field::new("leadsine".into(), DataType::Float64);
    Ok(Field::new("ht_sine".into(), DataType::Struct(vec![f1, f2])))
}

#[polars_expr(output_type_func=ht_sine_output)]
pub fn ht_sine(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let n = real.len();

    let mut sine = vec![None; n];
    let mut leadsine = vec![None; n];
    if n < 32 {
        let s1 = Series::new("sine".into(), sine);
        let s2 = Series::new("leadsine".into(), leadsine);
        return Ok(StructChunked::from_series("ht_sine".into(), n, [s1, s2].iter())?.into_series());
    }

    let smooth = calc_smooth(real);
    let mut detrend = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut i1 = [0.0; 7];
    let (mut i2, mut q2, mut re, mut im, mut period) = (0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 6..n {
        let prev_period = if i > 6 { period } else { 6.0 };
        let adj = 0.075 * prev_period + 0.54;

        let detrend_curr = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * adj;
        shift_push7(&mut detrend, detrend_curr);

        let q1_curr =
            (0.0962 * detrend[0] + 0.5769 * detrend[2] - 0.5769 * detrend[4] - 0.0962 * detrend[6])
                * adj;
        shift_push7(&mut q1, q1_curr);
        shift_push7(&mut i1, detrend[3]);

        let ji = (0.0962 * i1[0] + 0.5769 * i1[2] - 0.5769 * i1[4] - 0.0962 * i1[6]) * adj;
        let jq = (0.0962 * q1[0] + 0.5769 * q1[2] - 0.5769 * q1[4] - 0.0962 * q1[6]) * adj;

        let i2_curr = 0.2 * (i1[0] - jq) + 0.8 * i2;
        let q2_curr = 0.2 * (q1[0] + ji) + 0.8 * q2;

        let re_curr = 0.2 * (i2_curr * i2 + q2_curr * q2) + 0.8 * re;
        let im_curr = 0.2 * (i2_curr * q2 - q2_curr * i2) + 0.8 * im;

        i2 = i2_curr;
        q2 = q2_curr;
        re = re_curr;
        im = im_curr;

        if im != 0.0 && re != 0.0 {
            period = TAU / (im / re).atan();
        }
        period = period
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period = 0.2 * period + 0.8 * prev_period;

        if i >= 31 {
            let dc_phase = if i1[0] != 0.0 {
                (q1[0] / i1[0]).atan() * 180.0 / PI
            } else {
                0.0
            };
            sine[i] = Some((dc_phase * PI / 180.0).sin());
            leadsine[i] = Some(((dc_phase + 45.0) * PI / 180.0).sin());
        }
    }

    let s1 = Series::new("sine".into(), sine);
    let s2 = Series::new("leadsine".into(), leadsine);
    Ok(StructChunked::from_series("ht_sine".into(), n, [s1, s2].iter())?.into_series())
}

#[polars_expr(output_type=Float64)]
pub fn ht_trendline(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let n = real.len();

    let mut out = vec![None; n];
    if n < 32 {
        return Ok(Series::new("ht_trendline".into(), out));
    }

    let smooth = calc_smooth(real);
    let mut detrend = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut i1 = [0.0; 7];
    let (mut i2, mut q2, mut re, mut im, mut period) = (0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 6..n {
        let prev_period = if i > 6 { period } else { 6.0 };
        let adj = 0.075 * prev_period + 0.54;

        let detrend_curr = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * adj;
        shift_push7(&mut detrend, detrend_curr);

        let q1_curr =
            (0.0962 * detrend[0] + 0.5769 * detrend[2] - 0.5769 * detrend[4] - 0.0962 * detrend[6])
                * adj;
        shift_push7(&mut q1, q1_curr);
        shift_push7(&mut i1, detrend[3]);

        let ji = (0.0962 * i1[0] + 0.5769 * i1[2] - 0.5769 * i1[4] - 0.0962 * i1[6]) * adj;
        let jq = (0.0962 * q1[0] + 0.5769 * q1[2] - 0.5769 * q1[4] - 0.0962 * q1[6]) * adj;

        let i2_curr = 0.2 * (i1[0] - jq) + 0.8 * i2;
        let q2_curr = 0.2 * (q1[0] + ji) + 0.8 * q2;

        let re_curr = 0.2 * (i2_curr * i2 + q2_curr * q2) + 0.8 * re;
        let im_curr = 0.2 * (i2_curr * q2 - q2_curr * i2) + 0.8 * im;

        i2 = i2_curr;
        q2 = q2_curr;
        re = re_curr;
        im = im_curr;

        if im != 0.0 && re != 0.0 {
            period = TAU / (im / re).atan();
        }
        period = period
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period = 0.2 * period + 0.8 * prev_period;

        if i >= 31 {
            let mut trendline = 0.0;
            for j in 0..4 {
                trendline += real[i - j];
            }
            out[i] = Some(trendline * 0.25);
        }
    }

    Ok(Series::new("ht_trendline".into(), out))
}

#[polars_expr(output_type=Int32)]
pub fn ht_trendmode(inputs: &[Series]) -> PolarsResult<Series> {
    let real = inputs[0].cast(&DataType::Float64)?.rechunk();
    let real = real.f64()?.cont_slice()?;
    let n = real.len();

    let mut out = vec![None; n];
    if n < 32 {
        return Ok(Series::new("ht_trendmode".into(), out));
    }

    let smooth = calc_smooth(real);
    let mut detrend = [0.0; 7];
    let mut q1 = [0.0; 7];
    let mut i1 = [0.0; 7];
    let (mut i2, mut q2, mut re, mut im, mut period) = (0.0, 0.0, 0.0, 0.0, 0.0);

    for i in 6..n {
        let prev_period = if i > 6 { period } else { 6.0 };
        let adj = 0.075 * prev_period + 0.54;

        let detrend_curr = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6])
            * adj;
        shift_push7(&mut detrend, detrend_curr);

        let q1_curr =
            (0.0962 * detrend[0] + 0.5769 * detrend[2] - 0.5769 * detrend[4] - 0.0962 * detrend[6])
                * adj;
        shift_push7(&mut q1, q1_curr);
        shift_push7(&mut i1, detrend[3]);

        let ji = (0.0962 * i1[0] + 0.5769 * i1[2] - 0.5769 * i1[4] - 0.0962 * i1[6]) * adj;
        let jq = (0.0962 * q1[0] + 0.5769 * q1[2] - 0.5769 * q1[4] - 0.0962 * q1[6]) * adj;

        let i2_curr = 0.2 * (i1[0] - jq) + 0.8 * i2;
        let q2_curr = 0.2 * (q1[0] + ji) + 0.8 * q2;

        let re_curr = 0.2 * (i2_curr * i2 + q2_curr * q2) + 0.8 * re;
        let im_curr = 0.2 * (i2_curr * q2 - q2_curr * i2) + 0.8 * im;

        i2 = i2_curr;
        q2 = q2_curr;
        re = re_curr;
        im = im_curr;

        if im != 0.0 && re != 0.0 {
            period = TAU / (im / re).atan();
        }
        period = period
            .clamp(0.67 * prev_period, 1.5 * prev_period)
            .clamp(6.0, 50.0);
        period = 0.2 * period + 0.8 * prev_period;

        if i >= 31 {
            let mut trendline = 0.0;
            for j in 0..4 {
                trendline += real[i - j];
            }
            trendline *= 0.25;

            let trend = if (real[i] - trendline).abs() > 0.01 * trendline {
                1
            } else {
                0
            };
            out[i] = Some(trend);
        }
    }

    Ok(Series::new("ht_trendmode".into(), out))
}

// ====================================================================
// Calculation Helpers
// ====================================================================

#[inline(always)]
fn shift_push7(dq: &mut [f64; 7], val: f64) {
    for i in (1..7).rev() {
        dq[i] = dq[i - 1];
    }
    dq[0] = val;
}

fn calc_smooth(real: &[f64]) -> Vec<f64> {
    let n = real.len();
    let mut smooth = vec![0.0; n];

    for i in 3..n {
        smooth[i] = (4.0 * real[i] + 3.0 * real[i - 1] + 2.0 * real[i - 2] + real[i - 3]) * 0.1;
    }
    smooth
}
