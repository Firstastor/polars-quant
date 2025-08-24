use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

#[pyfunction]
#[pyo3(signature = (data, timeperiod=20, nbdevup=2.0, nbdevdn=2.0))]
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
#[pyo3(signature = (data, timeperiod=20))]
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
#[pyo3(signature = (data, timeperiod=20))]
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
#[pyo3(signature = (data, timeperiod=14, fast_limit=2.0, slow_limit=30.0))]
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
#[pyo3(signature = (data, timeperiod=20))]
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
#[pyo3(signature = (data, timeperiod=20))]
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
