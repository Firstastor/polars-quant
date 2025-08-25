use polars::prelude::*;
use polars::datatypes::DataType::Float64;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

#[pyfunction]
#[pyo3(signature = (data, timeperiod=5, nbdevup=2.0, nbdevdn=2.0))]
pub fn bband(
    data: PyDataFrame,
    timeperiod: usize,
    nbdevup: f64,
    nbdevdn: f64
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result = df.clone();
    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let series = col.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: 1,
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_middle", col.name())).into());
            let std_series = col.rolling_std(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: 1,
                center: false,
                ..Default::default()
            }).unwrap();
            let upper_band = (&series + &(&std_series * nbdevup)).unwrap().with_name((&format!("{}__upper", col.name())).into());
            let lower_band = (&series - &(&std_series * nbdevdn)).unwrap().with_name((&format!("{}_lower", col.name())).into());
            result = result.hstack(&[series.into(), upper_band.into(), lower_band.into()]).unwrap();
        }
    }
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=30))]
pub fn dema(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());
    let alpha = 2.0 / (timeperiod as f64 + 1.0);
    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut ema_short = Vec::with_capacity(col.len());
            let mut ema_long = Vec::with_capacity(ema_short.len());
            let mut dema = Vec::with_capacity(ema_short.len());
            for i in 0..col.len() {
                if i == 0 {
                    ema_short.push(col.f64().unwrap().get(i).unwrap());
                } else {
                    ema_short.push(alpha * col.f64().unwrap().get(i).unwrap() + (1.0 - alpha) * ema_short[i - 1]);
                }
            }
            for i in 0..ema_short.len() {
                if i == 0 {
                    ema_long.push(ema_short[i]);
                } else {
                    ema_long.push(alpha * ema_short[i] + (1.0 - alpha) * ema_long[i - 1]);
                }
            }
            for i in 0..ema_short.len() {
                dema.push(2.0 * ema_short[i] - ema_long[i]);
            }
            let dema_column = Column::new((&format!("{}_dema{}", col.name(), timeperiod)).into(), dema);
            result.push(dema_column);
        }
    }
    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=30))]
pub fn ema(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());
    let alpha =  2.0 / (timeperiod as f64 + 1.0);
    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut ema = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                if i == 0 {
                    ema.push(col.f64().unwrap().get(i).unwrap());
                } else {
                    ema.push(alpha * col.f64().unwrap().get(i).unwrap() + (1.0 - alpha) * ema[i - 1]);
                }
            }
            let ema = Column::new((&format!("{}_ema{}", col.name(), timeperiod)).into(), ema);
            result.push(ema)
        }
    }
    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=30, fast_limit=2.0, slow_limit=30.0))]
pub fn kama(
    data: PyDataFrame,
    timeperiod: usize,
    fast_limit: f64,
    slow_limit: f64
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());
    let fast_limit = 2.0 / (fast_limit + 1.0);
    let slow_limit = 2.0 / (slow_limit + 1.0);

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut kama = Vec::with_capacity(col.len());
            let sma = col.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: 1,
                center: false,
                ..Default::default()
            }).unwrap();

            for i in 0..timeperiod {
                kama.push(sma.f64().unwrap().get(i).unwrap());
            }

            let sum_absolute_change = abs(&diff(col, timeperiod as i64, polars::series::ops::NullBehavior::Ignore)
                                    .unwrap()).unwrap();
            let sum_total_change = abs(&diff(col, 1, polars::series::ops::NullBehavior::Ignore)
                                .unwrap()).unwrap().rolling_sum(RollingOptionsFixedWindow {
                                    window_size: timeperiod,
                                    min_periods: 1,
                                    center: false,
                                    ..Default::default()
                                }).unwrap();
            let er = (sum_absolute_change / sum_total_change).unwrap();
            let sc = (er * (fast_limit - slow_limit) + slow_limit).f64().unwrap()
                .apply(|opt_v| opt_v.map(|v| v * v)).into_series();

            for i in timeperiod..col.len() {
                let sc_value = sc.f64().unwrap().get(i).unwrap();
                let kama_value = kama[i - 1] * (1.0 - sc_value) + sc_value * col.f64().unwrap().get(i).unwrap();
                kama.push(kama_value);
            }

            let kama_column = Column::new((&format!("{}_kama{}", col.name(), timeperiod)).into(), kama);
            result.push(kama_column);
        }
    }
    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=30))]
pub fn ma(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let sma = col.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: timeperiod,
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_ma{}", col.name(), timeperiod)).into());
            
            let sma = Column::new((&format!("{}_ma{}", col.name(), timeperiod)).into(), sma);
            result.push(sma);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, c=10.0))]
pub fn mama(
    data: PyDataFrame,
    c: f64,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut mesa_values = Vec::with_capacity(col.len());

            let mut prev_price = col.f64().unwrap().get(0).unwrap();
            let mut prev_filtered = prev_price;
            mesa_values.push(prev_price.clone());
            for i in 1..col.len() {
                let price = col.f64().unwrap().get(i).unwrap();
                let acceleration = price - prev_price;
                let alpha = 2.0 / (1.0 + (-(acceleration / c)).exp());
                let filtered_value = alpha * price + (1.0 - alpha) * prev_filtered;
                mesa_values.push(filtered_value);
                prev_price = price;
                prev_filtered = filtered_value;
            }

            let mesa_col = Column::new((&format!("{}_mesa", col.name())).into(), mesa_values);
            result.push(mesa_col);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=30))]
pub fn mavp(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let std = col.rolling_std(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: timeperiod,
                center: false,
                ..Default::default()
            }).unwrap();
            let mavp_smoothed = std.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: 1,
                center: false,
                ..Default::default()
            }).unwrap();
            let mavp_column = Column::new((&format!("{}_mavp{}", col.name(), timeperiod)).into(), mavp_smoothed);
            result.push(mavp_column);
        }
    }
    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=20))]
pub fn sma(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let sma = col.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: 1,
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_sma{}", col.name(), timeperiod)).into());
            
            let sma = Column::new((&format!("{}_sma{}", col.name(), timeperiod)).into(), sma);
            result.push(sma);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=20, b=0.7))]
pub fn t3(
    data: PyDataFrame,
    timeperiod: usize,
    b: f64
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    let alpha = 2.0 / (timeperiod as f64 + 1.0);

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut ema1 = Vec::with_capacity(col.len());
            let mut ema2 = Vec::with_capacity(col.len());

            for i in 0..col.len() {
                if i == 0 {
                    ema1.push(col.f64().unwrap().get(i).unwrap());
                } else {
                    ema1.push(alpha * col.f64().unwrap().get(i).unwrap() + (1.0 - alpha) * ema1[i - 1]);
                }
            }
            for i in 0..col.len() {
                if i == 0 {
                    ema2.push(ema1[i]);
                } else {
                    ema2.push(alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1]);
                }
            }

            let mut t3 = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                let t = (1.0 + b) * ema1[i] - b * ema2[i];
                t3.push(t);
            }

            let t3_col = Column::new((&format!("{}_t3{}_b{}", col.name(), timeperiod, b)).into(), t3);
            result.push(t3_col);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=20))]
pub fn tema(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    let alpha = 2.0 / (timeperiod as f64 + 1.0);

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut ema1 = Vec::with_capacity(col.len());
            let mut ema2 = Vec::with_capacity(col.len());
            let mut ema3 = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                if i == 0 {
                    ema1.push(col.f64().unwrap().get(i).unwrap());
                } else {
                    ema1.push(alpha * col.f64().unwrap().get(i).unwrap() + (1.0 - alpha) * ema1[i - 1]);
                }
            }
            for i in 0..col.len() {
                if i == 0 {
                    ema2.push(ema1[i]);
                } else {
                    ema2.push(alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1]);
                }
            }

            for i in 0..col.len() {
                if i == 0 {
                    ema3.push(ema2[i]);
                } else {
                    ema3.push(alpha * ema2[i] + (1.0 - alpha) * ema3[i - 1]);
                }
            }
            let mut tema = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                let t = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i];
                tema.push(t);
            }

            let tema = Column::new((&format!("{}_tema{}", col.name(), timeperiod)).into(), tema);
            result.push(tema);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=20))]
pub fn trima(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let sma1 = col.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: timeperiod,
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_sma1{}", col.name(), timeperiod)).into());
            let sma2 = sma1.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: timeperiod,
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_sma2{}", col.name(), timeperiod)).into());
            let sma3 = sma2.rolling_mean(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: timeperiod,
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_sma3{}", col.name(), timeperiod)).into());

            let trima = ((sma1 * 3 - sma2 * 3).unwrap() + sma3).unwrap().into_column();
            result.push(trima);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=20))]
pub fn wma(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());
    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        let weights: Vec<f64> = (1..=timeperiod).rev()
            .map(|i| i as f64 / (timeperiod * (timeperiod + 1) / 2) as f64)
            .collect(); 
        if col.dtype() == &DataType::Float64 {
            let wma = col.rolling_sum(RollingOptionsFixedWindow {
                window_size: timeperiod,
                min_periods: timeperiod,
                weights: Some(weights),
                center: false,
                ..Default::default()
            }).unwrap().with_name((&format!("{}_wma{}", col.name(), timeperiod)).into());

            let wma_col = Column::new((&format!("{}_wma{}", col.name(), timeperiod)).into(), wma);
            result.push(wma_col);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn adx(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();
    let close = (&data.0).column("close").unwrap().f64().unwrap();
    let df = df!(
        "tr1" => high - low,
        "tr2" => (high - &close.shift(1)).wrapping_abs(),
        "tr3" => (low - &close.shift(1)).wrapping_abs(),
    ).unwrap();
    let tr = df.max_horizontal().unwrap().unwrap();
    let tr = tr.as_series().unwrap();

    let high = (&data.0).column("high").unwrap().as_series().unwrap();
    let low = (&data.0).column("low").unwrap().as_series().unwrap();
    let dm_plus = clip_min(&(high - &high.shift(1)).unwrap(), &Series::new("_".into(), &[0.0])).unwrap();
    let dm_minus = clip_min(&(&low.shift(1) - low).unwrap(), &Series::new("_".into(), &[0.0])).unwrap();

    let smoothed_tr = tr.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod,..Default::default()}).unwrap();
    let smoothed_dm_plus = dm_plus.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod,..Default::default()}).unwrap();
    let smoothed_dm_minus = dm_minus.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod,..Default::default()}).unwrap();

    let di_plus = (&smoothed_dm_plus / &smoothed_tr).unwrap();
    let di_minus = (&smoothed_dm_minus / &smoothed_tr).unwrap();

    let adx = abs(&((&di_plus - &di_minus).unwrap() / (&di_plus + &di_minus).unwrap()).unwrap()).unwrap().f64().unwrap() * 100.0;
    let adx = Series::new("adx".into(), adx);
    let adx = adx.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();
    let adx = Column::new("adxr".into(), adx);

    let result = data.0.hstack(&vec![adx]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn adxr(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();
    let close = (&data.0).column("close").unwrap().f64().unwrap();
    
    let df = df!(
        "tr1" => high - low,
        "tr2" => (high - &close.shift(1)).wrapping_abs(),
        "tr3" => (low - &close.shift(1)).wrapping_abs(),
    ).unwrap();
    
    let tr = df.max_horizontal().unwrap().unwrap();
    let tr = tr.as_series().unwrap();

    let high = (&data.0).column("high").unwrap().as_series().unwrap();
    let low = (&data.0).column("low").unwrap().as_series().unwrap();
    let dm_plus = clip_min(&(high - &high.shift(1)).unwrap(), &Series::new("_".into(), &[0.0])).unwrap();
    let dm_minus = clip_min(&(&low.shift(1) - low).unwrap(), &Series::new("_".into(), &[0.0])).unwrap();

    let smoothed_tr = tr.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();
    let smoothed_dm_plus = dm_plus.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();
    let smoothed_dm_minus = dm_minus.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();

    let di_plus = (&smoothed_dm_plus / &smoothed_tr).unwrap();
    let di_minus = (&smoothed_dm_minus / &smoothed_tr).unwrap();

    let adx = abs(&((&di_plus - &di_minus).unwrap() / (&di_plus + &di_minus).unwrap()).unwrap()).unwrap().f64().unwrap() * 100.0;
    let adx = Series::new("adx".into(), adx);
    let adx = adx.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();
    let adxr = (&adx + &adx.shift(timeperiod as i64)).unwrap() / 2.0;
    let adxr = Column::new("adxr".into(), adxr);
    let result = data.0.hstack(&vec![adxr]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, fastperiod=12, slowperiod=26))]
pub fn apo(
    data: PyDataFrame,
    fastperiod: usize,
    slowperiod: usize,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    let alpha_fast = 2.0 / (fastperiod as f64 + 1.0);
    let alpha_slow = 2.0 / (slowperiod as f64 + 1.0);

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let series = col.f64().unwrap();

            let mut ema_fast = Vec::with_capacity(col.len());
            let mut ema_slow = Vec::with_capacity(col.len());
            let mut apo = Vec::with_capacity(col.len());

            for i in 0..col.len() {
                let value = series.get(i).unwrap();

                if i == 0 {
                    ema_fast.push(value);
                    ema_slow.push(value);
                } else {
                    ema_fast.push(alpha_fast * value + (1.0 - alpha_fast) * ema_fast[i - 1]);
                    ema_slow.push(alpha_slow * value + (1.0 - alpha_slow) * ema_slow[i - 1]);
                }
                apo.push(ema_fast[i] - ema_slow[i]);
            }

            let apo = Column::new(
                (&format!("{}_apo{}_{}", col.name(), fastperiod, slowperiod)).into(),
                apo,
            );
            result.push(apo);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn aroon(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();

    let mut aroon_up = Vec::with_capacity(high.len());
    let mut aroon_down = Vec::with_capacity(low.len());

    for i in 0..high.len() {
        if i + 1 < timeperiod {
            aroon_up.push(None);
            aroon_down.push(None);
        } else {
            let start = i + 1 - timeperiod;

            let window_high = high.slice(start as i64, timeperiod);
            let window_low = low.slice(start as i64, timeperiod);
            let window_high = Series::new("window_high".into(), window_high);
            let window_low = Series::new("window_low".into(), window_low);

            let max_idx = window_high.arg_max().unwrap() as usize;
            let min_idx = window_low.arg_min().unwrap() as usize;

            aroon_up.push(Some(100.0 * (1 + max_idx) as f64 / timeperiod as f64));
            aroon_down.push(Some(100.0 * (1 + min_idx) as f64 / timeperiod as f64));
        }
    }

    let aroon_up_col = Column::new(format!("aroon_up{}", timeperiod).into(), aroon_up);
    let aroon_down_col = Column::new(format!("aroon_down{}", timeperiod).into(), aroon_down);

    let result = data.0.hstack(&vec![aroon_up_col, aroon_down_col]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn aroonosc(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();

    let mut aroonosc = Vec::with_capacity(high.len());

    for i in 0..high.len() {
        if i + 1 < timeperiod {
            aroonosc.push(None);
        } else {
            let start = i + 1 - timeperiod;

            let window_high = high.slice(start as i64, timeperiod);
            let window_low = low.slice(start as i64, timeperiod);
            let window_high = Series::new("window_high".into(), window_high);
            let window_low = Series::new("window_low".into(), window_low);

            let max_idx = window_high.arg_max().unwrap() as usize;
            let min_idx = window_low.arg_min().unwrap() as usize;

            let aroon_up = 100.0 * (1 + max_idx) as f64 / timeperiod as f64;
            let aroon_down = 100.0 * (1 + min_idx) as f64 / timeperiod as f64;

            aroonosc.push(Some(aroon_up - aroon_down));
        }
    }

    let aroonosc_col = Column::new(format!("aroonosc{}", timeperiod).into(), aroonosc);
    let result = data.0.hstack(&vec![aroonosc_col]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn bop(
    data: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let open = (&data.0).column("open").unwrap().f64().unwrap();
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();
    let close = (&data.0).column("close").unwrap().f64().unwrap();

    let mut bop_vals = Vec::with_capacity(close.len());

    for i in 0..close.len() {
        let o = open.get(i).unwrap();
        let h = high.get(i).unwrap();
        let l = low.get(i).unwrap();
        let c = close.get(i).unwrap();

        if h != l {
            bop_vals.push(Some((c - o) / (h - l)));
        } else {
            bop_vals.push(None);
        }
    }

    let bop_col = Column::new("bop".into(), bop_vals);
    let result = data.0.hstack(&vec![bop_col]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn cci(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();
    let close = (&data.0).column("close").unwrap().f64().unwrap();

    let tp = (&(high + low) + close) / 3.0;
    let tp = Series::new("tp".into(), tp);
    
    let sma_tp = tp.rolling_mean(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();

    let diff = abs(&(&tp - &sma_tp).unwrap()).unwrap();
    let mean_dev = diff.rolling_mean(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();

    let cci = ((&tp - &sma_tp).unwrap() / (&mean_dev * 0.015)).unwrap();
    let cci = Column::new(format!("cci{}", timeperiod).into(), cci);
    let result = data.0.hstack(&vec![cci]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn cmo(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let close = (&data.0).column("close").unwrap().as_series().unwrap();
    let diff = (close - &close.shift(1)).unwrap();

    let zero_series = Series::new("_".into(), &[0.0]);
    let up = clip_min(&diff, &zero_series).unwrap();
    let down = (clip_max(&diff, &zero_series).unwrap()) * -1.0;

    let sum_up = up.rolling_sum(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();

    let sum_down = down.rolling_sum(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();

    let numerator = (&sum_up - &sum_down).unwrap();
    let denominator = (&sum_up + &sum_down).unwrap();
    let cmo = ((&numerator) / &denominator).unwrap() * 100.0;
    let cmo = Column::new(format!("cmo{}", timeperiod).into(), cmo);
    let result = data.0.hstack(&vec![cmo]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn dx(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().f64().unwrap();
    let low = (&data.0).column("low").unwrap().f64().unwrap();
    let close = (&data.0).column("close").unwrap().f64().unwrap();
    let df = df![
        "tr1" => high - low,
        "tr2" => (high - &close.shift(1)).wrapping_abs(),
        "tr3" => (low - &close.shift(1)).wrapping_abs(),
    ].unwrap();
    let tr = df.max_horizontal().unwrap().unwrap();
    let tr = tr.as_series().unwrap();

    // +DM, -DM
    let high = (&data.0).column("high").unwrap().as_series().unwrap();
    let low = (&data.0).column("low").unwrap().as_series().unwrap();
    let dm_plus = clip_min(&(high - &high.shift(1)).unwrap(), &Series::new("_".into(), &[0.0])).unwrap();
    let dm_minus = clip_min(&(&low.shift(1) - low).unwrap(), &Series::new("_".into(), &[0.0])).unwrap();

    // 平滑 (rolling mean)
    let smoothed_tr = tr.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();
    let smoothed_dm_plus = dm_plus.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();
    let smoothed_dm_minus = dm_minus.rolling_mean(RollingOptionsFixedWindow{window_size: timeperiod, ..Default::default()}).unwrap();

    let di_plus = (&smoothed_dm_plus / &smoothed_tr).unwrap() * 100.0;
    let di_minus = (&smoothed_dm_minus / &smoothed_tr).unwrap() * 100.0;
    let dx = abs(&((&di_plus - &di_minus).unwrap() / (&di_plus + &di_minus).unwrap()).unwrap()).unwrap().f64().unwrap() * 100.0;
    let dx = Column::new("dx".into(), dx);
    let result = data.0.hstack(&vec![dx]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, fast=12, slow=26, signal=9))]
pub fn macd(
    data: PyDataFrame,
    fast: usize,
    slow: usize,
    signal: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::new();

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let close = col.f64().unwrap();

            let alpha_fast = 2.0 / (fast as f64 + 1.0);
            let mut ema_fast = Vec::with_capacity(close.len());
            for i in 0..close.len() {
                if i == 0 {
                    ema_fast.push(close.get(i).unwrap());
                } else {
                    ema_fast.push(alpha_fast * close.get(i).unwrap() + (1.0 - alpha_fast) * ema_fast[i - 1]);
                }
            }

            let alpha_slow = 2.0 / (slow as f64 + 1.0);
            let mut ema_slow = Vec::with_capacity(close.len());
            for i in 0..close.len() {
                if i == 0 {
                    ema_slow.push(close.get(i).unwrap());
                } else {
                    ema_slow.push(alpha_slow * close.get(i).unwrap() + (1.0 - alpha_slow) * ema_slow[i - 1]);
                }
            }

            let dif: Vec<f64> = ema_fast.iter().zip(ema_slow.iter()).map(|(f, s)| f - s).collect();

            let alpha_signal = 2.0 / (signal as f64 + 1.0);
            let mut dea = Vec::with_capacity(dif.len());
            for i in 0..dif.len() {
                if i == 0 {
                    dea.push(dif[i]);
                } else {
                    dea.push(alpha_signal * dif[i] + (1.0 - alpha_signal) * dea[i - 1]);
                }
            }

            let macd: Vec<f64> = dif.iter().zip(dea.iter()).map(|(d, e)| 2.0 * (d - e)).collect();
            result.push(Column::new(format!("{}_dif", col.name()).into(), dif));
            result.push(Column::new(format!("{}_dea", col.name()).into(), dea));
            result.push(Column::new(format!("{}_macd", col.name()).into(), macd));
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, period=14))]
pub fn mfi(
    data: PyDataFrame,
    period: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();

    let high = df.column("high").unwrap().f64().unwrap();
    let low = df.column("low").unwrap().f64().unwrap();
    let close = df.column("close").unwrap().f64().unwrap();
    let volume = df.column("volume").unwrap().cast(&Float64).unwrap();
    let volume = volume.f64().unwrap();
    let mut tp: Vec<f64> = Vec::with_capacity(high.len());
    let mut rmf: Vec<f64> = Vec::with_capacity(high.len());

    for i in 0..high.len() {
        let h = high.get(i).unwrap();
        let l = low.get(i).unwrap();
        let c = close.get(i).unwrap();
        let v = volume.get(i).unwrap();

        let t = (h + l + c) / 3.0;
        tp.push(t);
        rmf.push(t * v);
    }

    let mut pos_mf: Vec<f64> = vec![0.0; tp.len()];
    let mut neg_mf: Vec<f64> = vec![0.0; tp.len()];

    for i in 1..tp.len() {
        if tp[i] > tp[i - 1] {
            pos_mf[i] = rmf[i];
        } else if tp[i] < tp[i - 1] {
            neg_mf[i] = rmf[i];
        }
    }

    let mut mfi: Vec<f64> = Vec::with_capacity(tp.len());
    for i in 0..tp.len() {
        if i < period {
            mfi.push(f64::NAN);
        } else {
            let pos_sum: f64 = pos_mf[i + 1 - period..=i].iter().sum();
            let neg_sum: f64 = neg_mf[i + 1 - period..=i].iter().sum();

            if neg_sum == 0.0 {
                mfi.push(100.0);
            } else {
                let mfr = pos_sum / neg_sum;
                mfi.push(100.0 - (100.0 / (1.0 + mfr)));
            }
        }
    }

    let mut result = df.clone();
    result = result.hstack(&[Column::new("mfi".into(), mfi)]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, period=14))]
pub fn mom(
    data: PyDataFrame,
    period: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    
    let close = df.column("close").unwrap().f64().unwrap();

    let mut mom: Vec<f64> = Vec::with_capacity(close.len());

    for i in 0..close.len() {
        if i < period {
            mom.push(f64::NAN);
        } else {
            let mom_value = close.get(i).unwrap() - close.get(i - period).unwrap();
            mom.push(mom_value);
        }
    }

    let mut result = df.clone();
    result = result.hstack(&[Column::new("mom".into(), mom)]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, fastperiod=12, slowperiod=26))]
pub fn ppo(
    data: PyDataFrame,
    fastperiod: usize,
    slowperiod: usize,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    let alpha_fast = 2.0 / (fastperiod as f64 + 1.0);
    let alpha_slow = 2.0 / (slowperiod as f64 + 1.0);

    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let series = col.f64().unwrap();

            let mut ema_fast = Vec::with_capacity(col.len());
            let mut ema_slow = Vec::with_capacity(col.len());
            let mut ppo = Vec::with_capacity(col.len());

            for i in 0..col.len() {
                let value = series.get(i).unwrap();

                if i == 0 {
                    ema_fast.push(value);
                    ema_slow.push(value);
                } else {
                    ema_fast.push(alpha_fast * value + (1.0 - alpha_fast) * ema_fast[i - 1]);
                    ema_slow.push(alpha_slow * value + (1.0 - alpha_slow) * ema_slow[i - 1]);
                }

                if ema_slow[i] != 0.0 {
                    ppo.push(Some((ema_fast[i] - ema_slow[i]) / ema_slow[i] * 100.0));
                } else {
                    ppo.push(None);
                }
            }

            let ppo = Column::new(
                (&format!("{}_ppo{}_{}", col.name(), fastperiod, slowperiod)).into(),
                ppo,
            );
            result.push(ppo);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=10))]
pub fn roc(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    
    let close = df.column("close").unwrap().f64().unwrap();

    let mut roc: Vec<f64> = Vec::with_capacity(close.len());

    for i in 0..close.len() {
        if i < timeperiod {
            roc.push(f64::NAN);
        } else {
            let roc_value = (close.get(i).unwrap() - close.get(i - timeperiod).unwrap()) 
                            / close.get(i - timeperiod).unwrap() * 100.0;
            roc.push(roc_value);
        }
    }

    let mut result = df.clone();
    result = result.hstack(&[Column::new("roc".into(), roc)]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn rsi(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    
    let close = (&df).column("close").unwrap().as_series().unwrap();
    
    let diff = (close - &close.shift(1)).unwrap();

    let zero_series = Series::new("_".into(), &[0.0]);
    let up = clip_min(&diff, &zero_series).unwrap();
    let down = clip_max(&diff, &zero_series).unwrap() * -1.0;
    let avg_up = up.rolling_mean(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();
    let avg_down = down.rolling_mean(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();
    let rs = (avg_up / avg_down).unwrap();
    let rsi = ((&rs * 100.0) / (&rs + 1)).unwrap();
    let result = df.hstack(&[Column::new("rsi".into(), rsi)]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, fastk_period=5, slowk_period=3, slowd_period=3))]
pub fn stoch(
    data: PyDataFrame,
    fastk_period: usize,
    slowk_period: usize,
    slowd_period: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    
    let high = df.column("high").unwrap().as_series().unwrap();
    let low = df.column("low").unwrap().as_series().unwrap();
    let close = df.column("close").unwrap().as_series().unwrap();

    let highest_high = high.rolling_max(RollingOptionsFixedWindow{window_size:fastk_period,..Default::default()}).unwrap();
    let lowest_low = low.rolling_min(RollingOptionsFixedWindow{window_size:fastk_period,..Default::default()}).unwrap();

    let stoch_k = ((close - &lowest_low).unwrap() / (&highest_high - &lowest_low).unwrap()).unwrap() * 100.0;

    let smoothed_k = stoch_k.rolling_mean(RollingOptionsFixedWindow{window_size:slowk_period,..Default::default()}).unwrap();

    let stoch_d = smoothed_k.rolling_mean(RollingOptionsFixedWindow{window_size:slowd_period,..Default::default()}).unwrap();

    let result = df.hstack(&[
        Column::new("stoch_k".into(), smoothed_k),
        Column::new("stoch_d".into(), stoch_d)
    ]).unwrap();
    
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, fastk_period=5, fastd_period=3))]
pub fn stochf(
    data: PyDataFrame,
    fastk_period: usize,
    fastd_period: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    
    let high = df.column("high").unwrap().as_series().unwrap();
    let low = df.column("low").unwrap().as_series().unwrap();
    let close = df.column("close").unwrap().as_series().unwrap();

    let highest_high = high.rolling_max(RollingOptionsFixedWindow{window_size:fastk_period,..Default::default()}).unwrap();
    let lowest_low = low.rolling_min(RollingOptionsFixedWindow{window_size:fastk_period,..Default::default()}).unwrap();

    let stoch_k = ((close - &lowest_low).unwrap() / (&highest_high - &lowest_low).unwrap()).unwrap() * 100.0;

    let stoch_d = stoch_k.rolling_mean(RollingOptionsFixedWindow{window_size:fastd_period,..Default::default()}).unwrap();

    let result = df.hstack(&[
        Column::new("stoch_k".into(), stoch_k),
        Column::new("stoch_d".into(), stoch_d)
    ]).unwrap();
    
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14, fastk_period=5, fastd_period=3))]
pub fn stochrsi(
    data: PyDataFrame,
    timeperiod: usize,
    fastk_period: usize,
    fastd_period: usize,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();

    let close = (&df).column("close").unwrap().as_series().unwrap();
    
    let diff = (close - &close.shift(1)).unwrap();

    let zero_series = Series::new("_".into(), &[0.0]);
    let up = clip_min(&diff, &zero_series).unwrap();
    let down = (clip_max(&diff, &zero_series).unwrap()) * -1.0;
    let avg_up = up.rolling_sum(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();
    let avg_down = down.rolling_sum(RollingOptionsFixedWindow {
        window_size: timeperiod,
        min_periods: timeperiod,
        ..Default::default()
    }).unwrap();
    let rs = (avg_up / avg_down).unwrap();
    let rsi = ((&rs * 100.0) / (&rs + 1)).unwrap();

    let highest_rsi = rsi.rolling_max(RollingOptionsFixedWindow { window_size: fastk_period, ..Default::default() }).unwrap();
    let lowest_rsi = rsi.rolling_min(RollingOptionsFixedWindow { window_size: fastk_period, ..Default::default() }).unwrap();

    let stoch_rsi_k = ((&rsi - &lowest_rsi).unwrap() / (&highest_rsi - &lowest_rsi).unwrap()).unwrap() * 100.0;

    let stoch_rsi_d = stoch_rsi_k.rolling_mean(RollingOptionsFixedWindow { window_size: fastd_period, ..Default::default() }).unwrap();

    let result = df.hstack(&[
        Column::new("stochrsi_k".into(), stoch_rsi_k),
        Column::new("stochrsi_d".into(), stoch_rsi_d)
    ]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=30))]
pub fn trix(
    data: PyDataFrame,
    timeperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());
    let alpha =  2.0 / (timeperiod as f64 + 1.0);
    
    for col in df.get_columns() {
        let col = col.as_series().unwrap();
        if col.dtype() == &DataType::Float64 {
            let mut ema1 = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                if i == 0 {
                    ema1.push(col.f64().unwrap().get(i).unwrap());
                } else {
                    ema1.push(alpha * col.f64().unwrap().get(i).unwrap() + (1.0 - alpha) * ema1[i - 1]);
                }
            }
            let mut ema2 = Vec::with_capacity(ema1.len());
            for i in 0..ema1.len() {
                if i == 0 {
                    ema2.push(ema1[i]);
                } else {
                    ema2.push(alpha * ema1[i] + (1.0 - alpha) * ema2[i - 1]);
                }
            }
            let mut ema3 = Vec::with_capacity(ema2.len());
            for i in 0..ema2.len() {
                if i == 0 {
                    ema3.push(ema2[i]);
                } else {
                    ema3.push(alpha * ema2[i] + (1.0 - alpha) * ema3[i - 1]);
                }
            }
            let mut trix = Vec::with_capacity(ema3.len());
            trix.push(0.0);
            for i in 1..ema3.len() {
                let trix_value = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100.0;
                trix.push(trix_value);
            }
            
            let trix = Column::new((&format!("{}_trix{}", col.name(), timeperiod)).into(), trix);
            result.push(trix);
        }
    }

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, short_period=7, medium_period=14, long_period=28))]
pub fn ultosc(
    data: PyDataFrame,
    short_period: usize,
    medium_period: usize,
    long_period: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let mut result: Vec<Column> = Vec::with_capacity(df.width());

    let high = df.column("High").unwrap().f64().unwrap();
    let low = df.column("Low").unwrap().f64().unwrap();
    let close = df.column("Close").unwrap().f64().unwrap();

    let mut tr = Vec::with_capacity(df.width());
    let mut bp = Vec::with_capacity(df.width());
    
    for i in 0..df.width() {
        let tr_value = high.get(i).unwrap() - low.get(i).unwrap()
            .max((high.get(i).unwrap() - close.get(i - 1).unwrap()).abs())
            .max((low.get(i).unwrap() - close.get(i - 1).unwrap()).abs());
        let bp_value = close.get(i).unwrap() - low.get(i).unwrap();
        
        tr.push(tr_value);
        bp.push(bp_value);
    }
    
    let tr = Series::new("tr".into(), tr);
    let bp = Series::new("tr".into(), bp);
    let sma_bp_short = &bp.rolling_mean(RollingOptionsFixedWindow{window_size:short_period,..Default::default()}).unwrap();
    let sma_tr_short = &tr.rolling_mean(RollingOptionsFixedWindow{window_size:short_period,..Default::default()}).unwrap();
    let sma_bp_medium = &bp.rolling_mean(RollingOptionsFixedWindow{window_size:medium_period,..Default::default()}).unwrap();
    let sma_tr_medium = &tr.rolling_mean(RollingOptionsFixedWindow{window_size:medium_period,..Default::default()}).unwrap();
    let sma_bp_long = &bp.rolling_mean(RollingOptionsFixedWindow{window_size:long_period,..Default::default()}).unwrap();
    let sma_tr_long = &tr.rolling_mean(RollingOptionsFixedWindow{window_size:long_period,..Default::default()}).unwrap();
    
    let ultosc = (((sma_bp_short / sma_tr_short).unwrap() * 4.0 +
                        (sma_bp_medium / sma_tr_medium).unwrap() * 2.0).unwrap() +
                        (sma_bp_long / sma_tr_long).unwrap()).unwrap() / 7.0;

    let ultosc = Column::new("ultosc".into(), ultosc);
    result.push(ultosc);

    let result = df.hstack(&result).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, timeperiod=14))]
pub fn willr(
    data: PyDataFrame,
    timeperiod: usize,
) -> PyResult<PyDataFrame> {
    let high = (&data.0).column("high").unwrap().as_series().unwrap();
    let low = (&data.0).column("low").unwrap().as_series().unwrap();
    let close = (&data.0).column("close").unwrap().as_series().unwrap();

    let highest_high = high.rolling_max(RollingOptionsFixedWindow {window_size: timeperiod,..Default::default()}).unwrap();
    let lowest_low = low.rolling_min(RollingOptionsFixedWindow {window_size: timeperiod,..Default::default()}).unwrap();

    let willr = ((&highest_high - close).unwrap() / (&highest_high - &lowest_low).unwrap()).unwrap() * (-100.0);
    let willr = Column::new(format!("willr{}", timeperiod).into(), willr);
    let result = data.0.hstack(&vec![willr]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn ad(
    data: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();

    let high = df.column("high").unwrap().f64().unwrap();
    let low = df.column("low").unwrap().f64().unwrap();
    let close = df.column("close").unwrap().f64().unwrap();
    let volume = df.column("volume").unwrap().cast(&Float64).unwrap();
    let volume = volume.f64().unwrap();

    let mut clv: Vec<f64> = Vec::with_capacity(high.len());
    let mut ad: Vec<f64> = Vec::with_capacity(high.len());

    let mut running_ad = 0.0;

    for i in 0..high.len() {
        let h = high.get(i).unwrap();
        let l = low.get(i).unwrap();
        let c = close.get(i).unwrap();
        let v = volume.get(i).unwrap();

        let denom = h - l;
        let clv_i = if denom.abs() < f64::EPSILON {
            0.0
        } else {
            (2.0 * c - h - l) / denom
        };

        let money_flow_vol = clv_i * v;
        running_ad += money_flow_vol;

        clv.push(clv_i);
        ad.push(running_ad);
    }

    let mut result = df.clone();
    result = result.hstack(&[
        Column::new("clv".into(), clv),
        Column::new("ad".into(), ad),
    ]).unwrap();

    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data, fastperiod=3, slowperiod=10))]
pub fn adosc(
    data: PyDataFrame,
    fastperiod: usize,
    slowperiod: usize
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();

    let high = df.column("high").unwrap().f64().unwrap();
    let low = df.column("low").unwrap().f64().unwrap();
    let close = df.column("close").unwrap().f64().unwrap();
    let volume = df.column("volume").unwrap().cast(&Float64).unwrap();
    let volume = volume.f64().unwrap();
    let mut ad: Vec<f64> = Vec::with_capacity(high.len());
    let mut running_ad = 0.0;

    for i in 0..high.len() {
        let h = high.get(i).unwrap();
        let l = low.get(i).unwrap();
        let c = close.get(i).unwrap();
        let v = volume.get(i).unwrap();

        let denom = h - l;
        let clv_i = if denom.abs() < f64::EPSILON {
            0.0
        } else {
            (2.0 * c - h - l) / denom
        };

        running_ad += clv_i * v;
        ad.push(running_ad);
    }
    let ad = Series::new("ad".into(), ad);
    let mut ad_fast = Vec::with_capacity(ad.len());
    let alpha_fast = 2.0 / (1 + fastperiod) as f64;
    for i in 0..ad.len() {
        let value = ad.f64().unwrap().get(i).unwrap();
        if i == 0 {
            ad_fast.push(value);
        } else {
            ad_fast.push(alpha_fast * value + (1.0 - alpha_fast) * ad_fast[i - 1]);
        }
    }

    let mut ad_slow = Vec::with_capacity(ad.len());
    let alpha_slow = 2.0 / (1 + slowperiod) as f64;
    for i in 0..ad.len() {
        let value = ad.f64().unwrap().get(i).unwrap();
        if i == 0 {
            ad_slow.push(value);
        } else {
            ad_slow.push(alpha_slow * value + (1.0 - alpha_slow) * ad_slow[i - 1]);
        }
    }
    let adosc = (Column::new("ad_fast".into(),ad_fast) - Column::new("ad_slow".into(),ad_slow)).unwrap();
    let result = df.clone().hstack(&[adosc]).unwrap();
    Ok(PyDataFrame(result))
}

#[pyfunction]
#[pyo3(signature = (data))]
pub fn obv(
    data: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();

    let close = df.column("close").unwrap().f64().unwrap();
    let volume = df.column("volume").unwrap().cast(&Float64).unwrap();
    let volume = volume.f64().unwrap();

    let mut obv: Vec<f64> = Vec::with_capacity(close.len());
    let mut running_obv = volume.get(0).unwrap();

    for i in 0..close.len() {
        if i == 0 {
            obv.push(running_obv);
            continue;
        }

        let c = close.get(i).unwrap();
        let prev_c = close.get(i - 1).unwrap();
        let v = volume.get(i).unwrap();

        if c > prev_c {
            running_obv += v;
        } else if c < prev_c {
            running_obv -= v;
        } 

        obv.push(running_obv);
    }

    let mut result = df.clone();
    result = result.hstack(&[Column::new("obv".into(), obv)]).unwrap();

    Ok(PyDataFrame(result))
}