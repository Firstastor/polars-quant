use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::talib::*;

// ============================================================================
// 策略模块 - 交易策略集合
// 
// 说明:
// - 策略不限于使用talib函数，可以包含任何技术分析、因子计算等
// - 每个策略返回包含buy_signal和sell_signal列的DataFrame
// - 信号值: true表示触发，false表示不触发
// - 用户可以自行组合多个策略的信号
// ============================================================================

#[pyclass]
pub struct Strategy;

#[pymethods]
impl Strategy {
    #[new]
    pub fn new() -> Self {
        Strategy
    }

    /// MA均线策略
    /// 
    /// 支持多种均线类型、趋势过滤、斜率过滤、距离过滤
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `fast_period`: 快线周期
    /// - `slow_period`: 慢线周期
    /// - `ma_type`: 均线类型 ("sma", "ema", "wma", "dema", "tema")
    /// - `trend_period`: 趋势过滤均线周期（0表示不使用）
    /// - `trend_filter`: 是否启用趋势过滤（价格在趋势线上方才买入）
    /// - `slope_filter`: 是否启用斜率过滤（均线斜率向上才买入）
    /// - `distance_pct`: 价格与均线最小距离百分比过滤（0.0表示不使用）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", fast_period=10, slow_period=20, ma_type="sma", trend_period=0, trend_filter=false, slope_filter=false, distance_pct=0.0))]
    pub fn ma(
        &self,
        df: PyDataFrame,
        price_col: &str,
        fast_period: usize,
        slow_period: usize,
        ma_type: &str,
        trend_period: usize,
        trend_filter: bool,
        slope_filter: bool,
        distance_pct: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算快慢均线
        let fast_ma = match ma_type.to_lowercase().as_str() {
            "ema" => calculate_ema(&price_series, fast_period),
            "wma" => calculate_wma(&price_series, fast_period),
            "dema" => calculate_dema(&price_series, fast_period),
            "tema" => calculate_tema(&price_series, fast_period),
            _ => calculate_sma(&price_series, fast_period),
        }.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算快线失败: {}", e)))?;
        
        let slow_ma = match ma_type.to_lowercase().as_str() {
            "ema" => calculate_ema(&price_series, slow_period),
            "wma" => calculate_wma(&price_series, slow_period),
            "dema" => calculate_dema(&price_series, slow_period),
            "tema" => calculate_tema(&price_series, slow_period),
            _ => calculate_sma(&price_series, slow_period),
        }.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算慢线失败: {}", e)))?;
        
        // 可选：计算趋势过滤线
        let trend_ma = if trend_filter && trend_period > 0 {
            Some(match ma_type.to_lowercase().as_str() {
                "ema" => calculate_ema(&price_series, trend_period),
                "wma" => calculate_wma(&price_series, trend_period),
                "dema" => calculate_dema(&price_series, trend_period),
                "tema" => calculate_tema(&price_series, trend_period),
                _ => calculate_sma(&price_series, trend_period),
            }.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算趋势线失败: {}", e)))?)
        } else {
            None
        };
        
        // 计算信号
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let fast_values = fast_ma.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let slow_values = slow_ma.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let trend_values = trend_ma.as_ref().map(|s| s.f64().ok()).flatten();
        
        let len = fast_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 2..len {
            if let (Some(fast_prev), Some(slow_prev), Some(fast_curr), Some(slow_curr)) = 
                (fast_values.get(i-1), slow_values.get(i-1), fast_values.get(i), slow_values.get(i)) {
                
                // 趋势过滤：价格在趋势线上方
                let trend_ok = if let Some(ref tv) = trend_values {
                    if let (Some(price), Some(trend)) = (price_values.get(i), tv.get(i)) {
                        price > trend
                    } else {
                        !trend_filter
                    }
                } else {
                    true
                };
                
                // 斜率过滤：均线向上
                let slope_ok = if slope_filter {
                    if let (Some(fast_prev2), Some(_slow_prev2)) = (fast_values.get(i-2), slow_values.get(i-2)) {
                        fast_curr > fast_prev && slow_curr > slow_prev && fast_prev > fast_prev2
                    } else {
                        false
                    }
                } else {
                    true
                };
                
                // 距离过滤：金叉时价格与快线距离
                let distance_ok = if distance_pct > 0.0 {
                    if let Some(price) = price_values.get(i) {
                        ((price - fast_curr).abs() / fast_curr) >= distance_pct / 100.0
                    } else {
                        false
                    }
                } else {
                    true
                };
                
                // 金叉：快线上穿慢线
                if fast_prev <= slow_prev && fast_curr > slow_curr && trend_ok && slope_ok && distance_ok {
                    buy_signals[i] = true;
                }
                // 死叉：快线下穿慢线
                if fast_prev >= slow_prev && fast_curr < slow_curr {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// MACD策略
    /// 
    /// MACD金叉买入，死叉卖出，支持零轴过滤、柱状图过滤、背离检测
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `fast_period`: 快线周期（默认12）
    /// - `slow_period`: 慢线周期（默认26）
    /// - `signal_period`: 信号线周期（默认9）
    /// - `histogram_threshold`: 柱状图绝对值阈值过滤（默认0.0，不过滤）
    /// - `zero_cross_filter`: 是否只在零轴上方买入
    /// - `divergence_lookback`: 背离检测回溯周期（0表示不检测）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", fast_period=12, slow_period=26, signal_period=9, histogram_threshold=0.0, zero_cross_filter=false, divergence_lookback=0))]
    pub fn macd(
        &self,
        df: PyDataFrame,
        price_col: &str,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
        histogram_threshold: f64,
        zero_cross_filter: bool,
        divergence_lookback: usize,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算MACD
        let (macd, signal, histogram) = calculate_macd(&price_series, fast_period, slow_period, signal_period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("MACD计算失败: {}", e)))?;
        
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let macd_values = macd.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let signal_values = signal.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let hist_values = histogram.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = macd_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 1..len {
            if let (Some(macd_prev), Some(signal_prev), Some(macd_curr), Some(signal_curr), Some(hist)) = 
                (macd_values.get(i-1), signal_values.get(i-1), macd_values.get(i), signal_values.get(i), hist_values.get(i)) {
                
                // 零轴过滤
                let zero_ok = !zero_cross_filter || macd_curr > 0.0;
                
                // 柱状图过滤
                let hist_ok = hist.abs() >= histogram_threshold;
                
                // 背离检测（简化版：检测底背离）
                let divergence_ok = if divergence_lookback > 0 && i >= divergence_lookback {
                    let lookback_idx = i.saturating_sub(divergence_lookback);
                    if let (Some(price_old), Some(price_new), Some(macd_old), Some(macd_new)) = 
                        (price_values.get(lookback_idx), price_values.get(i), 
                         macd_values.get(lookback_idx), macd_values.get(i)) {
                        // 底背离：价格创新低但MACD没有创新低
                        price_new < price_old && macd_new > macd_old
                    } else {
                        false
                    }
                } else {
                    true
                };
                
                // 金叉：MACD上穿信号线
                if macd_prev <= signal_prev && macd_curr > signal_curr && zero_ok && hist_ok && divergence_ok {
                    buy_signals[i] = true;
                }
                // 死叉：MACD下穿信号线
                if macd_prev >= signal_prev && macd_curr < signal_curr {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// RSI策略
    /// 
    /// RSI超卖买入，超买卖出，支持多级阈值、背离检测
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `period`: RSI周期（默认14）
    /// - `oversold`: 超卖阈值（默认30）
    /// - `overbought`: 超买阈值（默认70）
    /// - `extreme_oversold`: 极度超卖阈值（默认20，更强信号）
    /// - `extreme_overbought`: 极度超买阈值（默认80，更强信号）
    /// - `divergence_lookback`: 背离检测回溯周期（0表示不检测）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", period=14, oversold=30.0, overbought=70.0, extreme_oversold=20.0, extreme_overbought=80.0, divergence_lookback=0))]
    pub fn rsi(
        &self,
        df: PyDataFrame,
        price_col: &str,
        period: usize,
        oversold: f64,
        overbought: f64,
        extreme_oversold: f64,
        extreme_overbought: f64,
        divergence_lookback: usize,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算RSI
        let rsi = calculate_rsi(&price_series, period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("RSI计算失败: {}", e)))?;
        
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let rsi_values = rsi.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = rsi_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 1..len {
            if let (Some(rsi_prev), Some(rsi_curr)) = (rsi_values.get(i-1), rsi_values.get(i)) {
                
                // 背离检测（简化版）
                let bull_divergence = if divergence_lookback > 0 && i >= divergence_lookback {
                    let lookback_idx = i.saturating_sub(divergence_lookback);
                    if let (Some(price_old), Some(price_new), Some(rsi_old), Some(rsi_new)) = 
                        (price_values.get(lookback_idx), price_values.get(i), 
                         rsi_values.get(lookback_idx), rsi_values.get(i)) {
                        // 底背离：价格创新低但RSI没有创新低
                        price_new < price_old && rsi_new > rsi_old
                    } else {
                        false
                    }
                } else {
                    false
                };
                
                let bear_divergence = if divergence_lookback > 0 && i >= divergence_lookback {
                    let lookback_idx = i.saturating_sub(divergence_lookback);
                    if let (Some(price_old), Some(price_new), Some(rsi_old), Some(rsi_new)) = 
                        (price_values.get(lookback_idx), price_values.get(i), 
                         rsi_values.get(lookback_idx), rsi_values.get(i)) {
                        // 顶背离：价格创新高但RSI没有创新高
                        price_new > price_old && rsi_new < rsi_old
                    } else {
                        false
                    }
                } else {
                    false
                };
                
                // 超卖区域向上突破（普通信号）
                if rsi_prev <= oversold && rsi_curr > oversold {
                    buy_signals[i] = true;
                }
                // 极度超卖反弹（强信号）
                if rsi_prev <= extreme_oversold && rsi_curr > extreme_oversold {
                    buy_signals[i] = true;
                }
                // 底背离买入
                if bull_divergence && rsi_curr < oversold {
                    buy_signals[i] = true;
                }
                
                // 超买区域向下突破（普通信号）
                if rsi_prev >= overbought && rsi_curr < overbought {
                    sell_signals[i] = true;
                }
                // 极度超买回落（强信号）
                if rsi_prev >= extreme_overbought && rsi_curr < extreme_overbought {
                    sell_signals[i] = true;
                }
                // 顶背离卖出
                if bear_divergence && rsi_curr > overbought {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 布林带策略
    /// 
    /// 支持突破、反转、挤压后扩张等多种形态
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `period`: 布林带周期（默认20）
    /// - `std_dev`: 标准差倍数（默认2.0）
    /// - `strategy_type`: 策略类型 ("bounce"=反弹, "breakout"=突破, "squeeze"=挤压)
    /// - `bandwidth_threshold`: 带宽阈值（用于挤压策略，默认0.1）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", period=20, std_dev=2.0, strategy_type="bounce", bandwidth_threshold=0.1))]
    pub fn bband(
        &self,
        df: PyDataFrame,
        price_col: &str,
        period: usize,
        std_dev: f64,
        strategy_type: &str,
        bandwidth_threshold: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算布林带
        let (upper, middle, lower) = calculate_bband(&price_series, period, std_dev)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("布林带计算失败: {}", e)))?;
        
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let upper_values = upper.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let middle_values = middle.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let lower_values = lower.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 1..len {
            if let (Some(price_prev), Some(price_curr), Some(lower_prev), Some(lower_curr), 
                    Some(upper_prev), Some(upper_curr), Some(middle_curr)) = 
                (price_values.get(i-1), price_values.get(i), lower_values.get(i-1), lower_values.get(i),
                 upper_values.get(i-1), upper_values.get(i), middle_values.get(i)) {
                
                match strategy_type.to_lowercase().as_str() {
                    "breakout" => {
                        // 突破策略：价格突破上轨买入，跌破下轨卖出
                        if price_prev <= upper_prev && price_curr > upper_curr {
                            buy_signals[i] = true;
                        }
                        if price_prev >= lower_prev && price_curr < lower_curr {
                            sell_signals[i] = true;
                        }
                    },
                    "squeeze" => {
                        // 挤压策略：带宽收窄后的突破
                        let bandwidth = (upper_curr - lower_curr) / middle_curr;
                        let bandwidth_prev = if let (Some(u), Some(l), Some(m)) = 
                            (upper_values.get(i-1), lower_values.get(i-1), middle_values.get(i-1)) {
                            (u - l) / m
                        } else {
                            bandwidth
                        };
                        
                        // 带宽从收窄状态扩张
                        if bandwidth_prev < bandwidth_threshold && bandwidth > bandwidth_threshold {
                            if price_curr > middle_curr {
                                buy_signals[i] = true;
                            } else {
                                sell_signals[i] = true;
                            }
                        }
                    },
                    _ => {
                        // 默认反弹策略：价格从下轨反弹买入，从上轨回落卖出
                        if price_prev <= lower_prev && price_curr > lower_curr {
                            buy_signals[i] = true;
                        }
                        if price_prev >= upper_prev && price_curr < upper_curr {
                            sell_signals[i] = true;
                        }
                    }
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 随机指标策略
    /// 
    /// K线D线金叉买入，死叉卖出，支持超买超卖过滤、背离检测
    /// 
    /// # 参数
    /// - `df`: 包含OHLC数据的DataFrame
    /// - `k_period`: K线周期（默认14）
    /// - `d_period`: D线周期（默认3）
    /// - `oversold`: 超卖阈值（默认20）
    /// - `overbought`: 超买阈值（默认80）
    /// - `cross_in_zone`: 是否要求在区域内交叉（True=区域内，False=区域外也可以）
    /// - `divergence_lookback`: 背离检测回溯周期（0表示不检测）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, k_period=14, d_period=3, oversold=20.0, overbought=80.0, cross_in_zone=true, divergence_lookback=0))]
    pub fn stoch(
        &self,
        df: PyDataFrame,
        k_period: usize,
        d_period: usize,
        oversold: f64,
        overbought: f64,
        cross_in_zone: bool,
        divergence_lookback: usize,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let high = data.column("high")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取high列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let low = data.column("low")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取low列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算STOCH
        let (k_series, d_series) = calculate_stoch(&high, &low, &close, k_period, d_period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("STOCH计算失败: {}", e)))?;
        
        let close_values = close.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let k_values = k_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let d_values = d_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = k_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 1..len {
            if let (Some(k_prev), Some(d_prev), Some(k_curr), Some(d_curr)) = 
                (k_values.get(i-1), d_values.get(i-1), k_values.get(i), d_values.get(i)) {
                
                // 背离检测
                let bull_divergence = if divergence_lookback > 0 && i >= divergence_lookback {
                    let lookback_idx = i.saturating_sub(divergence_lookback);
                    if let (Some(price_old), Some(price_new), Some(k_old), Some(k_new)) = 
                        (close_values.get(lookback_idx), close_values.get(i), 
                         k_values.get(lookback_idx), k_values.get(i)) {
                        price_new < price_old && k_new > k_old
                    } else {
                        false
                    }
                } else {
                    false
                };
                
                // 金叉买入
                if k_prev <= d_prev && k_curr > d_curr {
                    if cross_in_zone {
                        // 要求在超卖区域内金叉
                        if k_curr < oversold || bull_divergence {
                            buy_signals[i] = true;
                        }
                    } else {
                        // 任意位置金叉都可以
                        buy_signals[i] = true;
                    }
                }
                
                // 死叉卖出
                if k_prev >= d_prev && k_curr < d_curr {
                    if cross_in_zone {
                        // 要求在超买区域内死叉
                        if k_curr > overbought {
                            sell_signals[i] = true;
                        }
                    } else {
                        // 任意位置死叉都可以
                        sell_signals[i] = true;
                    }
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// CCI策略
    /// 
    /// CCI超卖买入，超买卖出，支持趋势过滤
    /// 
    /// # 参数
    /// - `df`: 包含OHLC数据的DataFrame
    /// - `period`: CCI周期（默认20）
    /// - `oversold`: 超卖阈值（默认-100）
    /// - `overbought`: 超买阈值（默认100）
    /// - `extreme_oversold`: 极度超卖（默认-200）
    /// - `extreme_overbought`: 极度超买（默认200）
    /// - `trend_filter`: 是否启用趋势过滤（CCI>0才买，CCI<0才卖）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, period=20, oversold=-100.0, overbought=100.0, extreme_oversold=-200.0, extreme_overbought=200.0, trend_filter=false))]
    pub fn cci(
        &self,
        df: PyDataFrame,
        period: usize,
        oversold: f64,
        overbought: f64,
        extreme_oversold: f64,
        extreme_overbought: f64,
        trend_filter: bool,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let high = data.column("high")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取high列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let low = data.column("low")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取low列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算CCI
        let cci = calculate_cci(&high, &low, &close, period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CCI计算失败: {}", e)))?;
        
        let cci_values = cci.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = cci_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 1..len {
            if let (Some(cci_prev), Some(cci_curr)) = (cci_values.get(i-1), cci_values.get(i)) {
                
                // 趋势过滤
                let trend_ok_buy = !trend_filter || cci_curr > 0.0;
                let trend_ok_sell = !trend_filter || cci_curr < 0.0;
                
                // 从超卖区域向上突破（普通信号）
                if cci_prev <= oversold && cci_curr > oversold && trend_ok_buy {
                    buy_signals[i] = true;
                }
                // 从极度超卖反弹（强信号）
                if cci_prev <= extreme_oversold && cci_curr > extreme_oversold {
                    buy_signals[i] = true;
                }
                
                // 从超买区域向下突破（普通信号）
                if cci_prev >= overbought && cci_curr < overbought && trend_ok_sell {
                    sell_signals[i] = true;
                }
                // 从极度超买回落（强信号）
                if cci_prev >= extreme_overbought && cci_curr < extreme_overbought {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// ADX趋势强度策略
    /// 
    /// ADX>阈值表示趋势强劲，结合DI方向判断买卖
    /// 
    /// # 参数
    /// - `df`: 包含OHLC数据的DataFrame
    /// - `period`: ADX周期（默认14）
    /// - `adx_threshold`: ADX阈值（默认25）
    /// - `adx_rising`: 是否要求ADX上升（True=趋势加强中）
    /// - `di_spread`: DI+和DI-的最小差值要求（默认0.0）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, period=14, adx_threshold=25.0, adx_rising=false, di_spread=0.0))]
    pub fn adx(
        &self,
        df: PyDataFrame,
        period: usize,
        adx_threshold: f64,
        adx_rising: bool,
        di_spread: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let high = data.column("high")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取high列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let low = data.column("low")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取low列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        // 计算ADX和DI
        let adx = calculate_adx(&high, &low, &close, period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("ADX计算失败: {}", e)))?;
        let plus_di = calculate_plus_di(&high, &low, &close, period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("DI+计算失败: {}", e)))?;
        let minus_di = calculate_minus_di(&high, &low, &close, period)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("DI-计算失败: {}", e)))?;
        
        let adx_values = adx.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let plus_values = plus_di.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let minus_values = minus_di.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = adx_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in 1..len {
            if let (Some(adx_curr), Some(plus_prev), Some(minus_prev), Some(plus_curr), Some(minus_curr)) = 
                (adx_values.get(i), plus_values.get(i-1), minus_values.get(i-1), 
                 plus_values.get(i), minus_values.get(i)) {
                
                // ADX上升过滤
                let adx_rising_ok = if adx_rising {
                    if let Some(adx_prev) = adx_values.get(i-1) {
                        adx_curr > adx_prev
                    } else {
                        false
                    }
                } else {
                    true
                };
                
                // DI差值过滤
                let spread_ok_buy = (plus_curr - minus_curr) >= di_spread;
                let spread_ok_sell = (minus_curr - plus_curr) >= di_spread;
                
                // 趋势强劲且DI+上穿DI-
                if adx_curr > adx_threshold && plus_prev <= minus_prev && plus_curr > minus_curr 
                    && adx_rising_ok && spread_ok_buy {
                    buy_signals[i] = true;
                }
                // 趋势强劲且DI-上穿DI+
                if adx_curr > adx_threshold && minus_prev <= plus_prev && minus_curr > plus_curr 
                    && adx_rising_ok && spread_ok_sell {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 动量突破策略
    /// 
    /// 价格突破N日高点买入，跌破N日低点卖出
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `lookback`: 回溯周期（默认20）
    /// - `volume_confirm`: 是否需要成交量确认（默认False）
    /// - `volume_multiplier`: 成交量倍数阈值（默认1.5）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, lookback=20, volume_confirm=false, volume_multiplier=1.5))]
    pub fn breakout(
        &self,
        df: PyDataFrame,
        lookback: usize,
        volume_confirm: bool,
        volume_multiplier: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let high = data.column("high")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取high列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let low = data.column("low")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取low列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let high_values = high.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let low_values = low.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let close_values = close.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        // 可选：成交量确认
        let volume_values = if volume_confirm {
            Some(data.column("volume")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取volume列失败: {}", e)))?
                .as_materialized_series()
                .f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?)
        } else {
            None
        };
        
        let len = close_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        for i in lookback..len {
            // 计算N日最高和最低
            let mut max_high = f64::NEG_INFINITY;
            let mut min_low = f64::INFINITY;
            let mut avg_volume = 0.0;
            let mut volume_count = 0;
            
            for j in (i-lookback)..i {
                if let Some(h) = high_values.get(j) {
                    max_high = max_high.max(h);
                }
                if let Some(l) = low_values.get(j) {
                    min_low = min_low.min(l);
                }
                if let Some(ref vv) = volume_values {
                    if let Some(v) = vv.get(j) {
                        avg_volume += v;
                        volume_count += 1;
                    }
                }
            }
            
            if volume_count > 0 {
                avg_volume /= volume_count as f64;
            }
            
            if let Some(close_curr) = close_values.get(i) {
                // 成交量确认
                let volume_ok = if let Some(ref vv) = volume_values {
                    if let Some(v) = vv.get(i) {
                        v >= avg_volume * volume_multiplier
                    } else {
                        !volume_confirm
                    }
                } else {
                    true
                };
                
                // 突破N日高点
                if close_curr > max_high && volume_ok {
                    buy_signals[i] = true;
                }
                // 跌破N日低点
                if close_curr < min_low && volume_ok {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 均值回归策略
    /// 
    /// 价格偏离均线过多时买入/卖出，等待回归
    /// 均值回归策略
    /// 
    /// 基于多种统计方法检测价格偏离并回归交易
    /// 
    /// # 参数
    /// - `df`: 包含OHLCV数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `method`: 回归方法
    ///   * "ma": 移动平均偏离法 - 价格偏离均线一定百分比时交易
    ///   * "bb": 布林带法 - 价格触及布林带上下轨时交易
    ///   * "zscore": Z分数法 - 价格标准化偏离达到阈值时交易
    ///   * "ttest": t检验法 - 价格偏离具有统计显著性时交易
    /// - `period`: 计算周期（历史数据窗口大小）
    /// - `threshold`: 开仓阈值
    ///   * MA模式：偏离百分比（如2.0表示偏离2%）
    ///   * BB模式：带宽倍数（如2.0表示2倍标准差）
    ///   * ZScore模式：Z分数（如2.0表示偏离2个标准差）
    ///   * TTest模式：t统计量（如2.0表示95%置信度显著性）
    /// - `exit_threshold`: 平仓阈值（回归到此阈值时平仓获利）
    /// 
    /// # TTest方法详解
    /// t检验方法通过统计显著性判断价格是否异常偏离：
    /// - t统计量 = (当前价格 - 历史均值) / 标准误差
    /// - 标准误差 = std / sqrt(n)，考虑样本量影响
    /// - threshold建议值：2.0（95%置信度）或 2.5（99%置信度）
    /// - 买入信号：t < -threshold（价格显著低于均值，超卖）
    /// - 卖出信号：t > +threshold（价格显著高于均值，超买）
    /// - 平仓信号：从极值区回归到 |t| < exit_threshold（均值回归完成）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", method="ma", period=20, threshold=2.0, exit_threshold=0.5))]
    pub fn reversion(
        &self,
        df: PyDataFrame,
        price_col: &str,
        method: &str,
        period: usize,
        threshold: f64,
        exit_threshold: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        match method.to_lowercase().as_str() {
            "ma" => {
                // 移动平均回归法
                let ma = calculate_sma(&price_series, period)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算均线失败: {}", e)))?;
                let ma_values = ma.f64()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
                
                for i in 1..len {
                    if let (Some(price_curr), Some(ma_curr), Some(price_prev), Some(ma_prev)) = 
                        (price_values.get(i), ma_values.get(i), price_values.get(i-1), ma_values.get(i-1)) {
                        
                        let deviation_curr = ((price_curr - ma_curr) / ma_curr) * 100.0;
                        let deviation_prev = ((price_prev - ma_prev) / ma_prev) * 100.0;
                        
                        // 价格严重低于均线后开始回升
                        if deviation_prev <= -threshold && deviation_curr > deviation_prev {
                            buy_signals[i] = true;
                        }
                        // 价格严重高于均线后开始回落
                        if deviation_prev >= threshold && deviation_curr < deviation_prev {
                            sell_signals[i] = true;
                        }
                        // 回归平仓
                        if deviation_curr.abs() < exit_threshold {
                            if deviation_prev < -exit_threshold {
                                sell_signals[i] = true;
                            } else if deviation_prev > exit_threshold {
                                buy_signals[i] = true;
                            }
                        }
                    }
                }
            },
            "bb" => {
                // 布林带回归法
                let ma = calculate_sma(&price_series, period)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算均线失败: {}", e)))?;
                let ma_values = ma.f64()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
                
                // 计算标准差
                let mut std_dev = vec![f64::NAN; len];
                for i in period..len {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for j in (i - period)..i {
                        if let (Some(p), Some(m)) = (price_values.get(j), ma_values.get(j)) {
                            sum += (p - m).powi(2);
                            count += 1;
                        }
                    }
                    if count > 0 {
                        std_dev[i] = (sum / count as f64).sqrt();
                    }
                }
                
                for i in period..len {
                    if let (Some(price), Some(ma), Some(std)) = 
                        (price_values.get(i), ma_values.get(i), std_dev.get(i).copied()) {
                        if std.is_nan() || std == 0.0 {
                            continue;
                        }
                        
                        let lower_band = ma - threshold * std;
                        let upper_band = ma + threshold * std;
                        
                        // 价格触及下轨
                        if price <= lower_band {
                            buy_signals[i] = true;
                        }
                        // 价格触及上轨
                        if price >= upper_band {
                            sell_signals[i] = true;
                        }
                        // 回归中轨附近
                        if (price - ma).abs() < exit_threshold * std {
                            if i > 0 {
                                if let Some(price_prev) = price_values.get(i-1) {
                                    if price_prev < lower_band {
                                        sell_signals[i] = true;
                                    } else if price_prev > upper_band {
                                        buy_signals[i] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "zscore" => {
                // Z分数回归法
                for i in period..len {
                    // 计算滚动均值和标准差
                    let mut sum = 0.0;
                    let mut count = 0;
                    for j in (i - period)..i {
                        if let Some(p) = price_values.get(j) {
                            sum += p;
                            count += 1;
                        }
                    }
                    if count == 0 {
                        continue;
                    }
                    
                    let mean = sum / count as f64;
                    
                    let mut variance = 0.0;
                    for j in (i - period)..i {
                        if let Some(p) = price_values.get(j) {
                            variance += (p - mean).powi(2);
                        }
                    }
                    let std = (variance / count as f64).sqrt();
                    
                    if let Some(price) = price_values.get(i) {
                        if std == 0.0 {
                            continue;
                        }
                        
                        let z_score = (price - mean) / std;
                        
                        // Z分数低于负阈值（超卖）
                        if z_score <= -threshold {
                            buy_signals[i] = true;
                        }
                        // Z分数高于正阈值（超买）
                        if z_score >= threshold {
                            sell_signals[i] = true;
                        }
                        // 回归正常区间
                        if z_score.abs() < exit_threshold && i > 0 {
                            if let Some(price_prev) = price_values.get(i-1) {
                                let z_prev = (price_prev - mean) / std;
                                if z_prev < -exit_threshold {
                                    sell_signals[i] = true;
                                } else if z_prev > exit_threshold {
                                    buy_signals[i] = true;
                                }
                            }
                        }
                    }
                }
            },
            "ttest" => {
                // t检验回归法 - 检测价格偏离的统计显著性
                // 原理：当价格相对历史均值有统计显著偏离时，预期会回归均值
                for i in period..len {
                    // 计算历史窗口的均值和标准差
                    let mut sum = 0.0;
                    let mut count = 0;
                    for j in (i - period)..i {
                        if let Some(p) = price_values.get(j) {
                            sum += p;
                            count += 1;
                        }
                    }
                    if count < 2 {
                        continue;
                    }
                    
                    let mean = sum / count as f64;
                    
                    let mut variance = 0.0;
                    for j in (i - period)..i {
                        if let Some(p) = price_values.get(j) {
                            variance += (p - mean).powi(2);
                        }
                    }
                    let std = (variance / (count - 1) as f64).sqrt(); // 样本标准差
                    
                    if let Some(price) = price_values.get(i) {
                        if std == 0.0 {
                            continue;
                        }
                        
                        // 计算t统计量: t = (X - μ) / (s / sqrt(n))
                        // X: 当前价格, μ: 历史均值, s: 样本标准差, n: 样本量
                        let standard_error = std / (count as f64).sqrt();
                        let t_stat = (price - mean) / standard_error;
                        
                        // 策略1: 极值反转信号（强信号）
                        // 负t值表示价格显著低于历史均值（超卖），预期反弹
                        if t_stat <= -threshold {
                            buy_signals[i] = true;
                        }
                        // 正t值表示价格显著高于历史均值（超买），预期回落
                        if t_stat >= threshold {
                            sell_signals[i] = true;
                        }
                        
                        // 策略2: 均值回归平仓（获利了结）
                        // 当价格回归到统计正常范围时平仓
                        if i > 0 {
                            if let Some(price_prev) = price_values.get(i-1) {
                                let t_prev = (price_prev - mean) / standard_error;
                                
                                // 从超卖区回归到正常区间 - 多头平仓
                                if t_prev <= -threshold && t_stat > -exit_threshold {
                                    sell_signals[i] = true;
                                }
                                // 从超买区回归到正常区间 - 空头平仓
                                if t_prev >= threshold && t_stat < exit_threshold {
                                    buy_signals[i] = true;
                                }
                            }
                        }
                    }
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "method必须是 'ma', 'bb', 'zscore' 或 'ttest'"
                ));
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 成交量策略
    /// 
    /// 基于成交量的多种分析方法生成交易信号
    /// 
    /// # 参数
    /// - `df`: 包含OHLCV数据的DataFrame
    /// - `method`: 分析方法 ("surge" 量能突破, "price" 量价配合, "obv" 能量潮, "accumulation" 吸筹分布)
    /// - `period`: 计算周期
    /// - `threshold`: 触发阈值
    /// - `price_confirm`: 是否需要价格确认
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, method="surge", period=20, threshold=2.0, price_confirm=true))]
    pub fn volume(
        &self,
        df: PyDataFrame,
        method: &str,
        period: usize,
        threshold: f64,
        price_confirm: bool,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let volume = data.column("volume")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取volume列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let close_values = close.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let volume_values = volume.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = close_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        match method.to_lowercase().as_str() {
            "surge" => {
                // 量能突破法：成交量突然放大
                let volume_ma = calculate_sma(&volume, period)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算成交量均线失败: {}", e)))?;
                let volume_ma_values = volume_ma.f64()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
                
                for i in 1..len {
                    if let (Some(vol), Some(vol_ma), Some(close_curr), Some(close_prev)) = 
                        (volume_values.get(i), volume_ma_values.get(i), close_values.get(i), close_values.get(i-1)) {
                        
                        let volume_surge = vol >= vol_ma * threshold;
                        
                        if volume_surge {
                            if !price_confirm {
                                // 不需要价格确认，量增即信号
                                if close_curr > close_prev {
                                    buy_signals[i] = true;
                                } else {
                                    sell_signals[i] = true;
                                }
                            } else {
                                // 需要价格确认
                                let price_change_pct = (close_curr - close_prev) / close_prev * 100.0;
                                if price_change_pct > 1.0 {
                                    buy_signals[i] = true;
                                } else if price_change_pct < -1.0 {
                                    sell_signals[i] = true;
                                }
                            }
                        }
                    }
                }
            },
            "price" => {
                // 量价配合法：量价同步上涨/下跌
                for i in 1..len {
                    if let (Some(close_curr), Some(close_prev), Some(vol_curr), Some(vol_prev)) = 
                        (close_values.get(i), close_values.get(i-1), volume_values.get(i), volume_values.get(i-1)) {
                        
                        let price_up = close_curr > close_prev * (1.0 + threshold / 100.0);
                        let price_down = close_curr < close_prev * (1.0 - threshold / 100.0);
                        let volume_up = vol_curr > vol_prev;
                        
                        // 量价齐升
                        if price_up && volume_up {
                            buy_signals[i] = true;
                        }
                        // 量价齐跌
                        if price_down && volume_up {
                            sell_signals[i] = true;
                        }
                    }
                }
            },
            "obv" => {
                // OBV能量潮法
                let mut obv = vec![0.0; len];
                for i in 1..len {
                    if let (Some(close_curr), Some(close_prev), Some(vol)) = 
                        (close_values.get(i), close_values.get(i-1), volume_values.get(i)) {
                        
                        if close_curr > close_prev {
                            obv[i] = obv[i-1] + vol;
                        } else if close_curr < close_prev {
                            obv[i] = obv[i-1] - vol;
                        } else {
                            obv[i] = obv[i-1];
                        }
                    }
                }
                
                // OBV均线
                for i in period..len {
                    let mut sum = 0.0;
                    for j in (i - period)..i {
                        sum += obv[j];
                    }
                    let obv_ma = sum / period as f64;
                    
                    // OBV上穿均线
                    if obv[i] > obv_ma && obv.get(i-1).map_or(false, |&prev| prev <= obv_ma) {
                        buy_signals[i] = true;
                    }
                    // OBV下穿均线
                    if obv[i] < obv_ma && obv.get(i-1).map_or(false, |&prev| prev >= obv_ma) {
                        sell_signals[i] = true;
                    }
                }
            },
            "accumulation" => {
                // 吸筹分布法：低位放量视为吸筹，高位放量视为派发
                let price_ma = calculate_sma(&close, period)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算价格均线失败: {}", e)))?;
                let price_ma_values = price_ma.f64()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
                
                let volume_ma = calculate_sma(&volume, period)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算成交量均线失败: {}", e)))?;
                let volume_ma_values = volume_ma.f64()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
                
                for i in period..len {
                    if let (Some(close), Some(price_ma), Some(vol), Some(vol_ma)) = 
                        (close_values.get(i), price_ma_values.get(i), volume_values.get(i), volume_ma_values.get(i)) {
                        
                        let volume_surge = vol > vol_ma * threshold;
                        let price_low = close < price_ma;
                        let price_high = close > price_ma;
                        
                        // 低位放量吸筹
                        if volume_surge && price_low {
                            buy_signals[i] = true;
                        }
                        // 高位放量派发
                        if volume_surge && price_high {
                            sell_signals[i] = true;
                        }
                    }
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "method必须是 'surge', 'price', 'obv' 或 'accumulation'"
                ));
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 网格策略
    /// 
    /// 在价格区间内设置网格，下跌买入，上涨卖出，适合震荡行情
    /// 
    /// # 参数
    /// - `df`: 包含OHLCV数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `grid_num`: 网格数量
    /// - `price_range_pct`: 价格区间百分比（基于初始价格）
    /// - `base_price`: 基准价格（None则使用第一个有效价格）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", grid_num=10, price_range_pct=20.0, base_price=None))]
    pub fn grid(
        &self,
        df: PyDataFrame,
        price_col: &str,
        grid_num: usize,
        price_range_pct: f64,
        base_price: Option<f64>,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        // 确定基准价格
        let base = if let Some(bp) = base_price {
            bp
        } else {
            price_values.iter().find_map(|x| x).unwrap_or(100.0)
        };
        
        // 计算网格价格
        let range = base * price_range_pct / 100.0;
        let upper_bound = base + range / 2.0;
        let lower_bound = base - range / 2.0;
        let grid_size = range / grid_num as f64;
        
        let len = price_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        let mut last_grid_level: Option<i32> = None;
        
        for i in 0..len {
            if let Some(price) = price_values.get(i) {
                if price < lower_bound || price > upper_bound {
                    continue;
                }
                
                // 计算当前价格所在网格层级
                let current_level = ((price - lower_bound) / grid_size).floor() as i32;
                
                if let Some(last_level) = last_grid_level {
                    // 下跌穿越网格线 - 买入
                    if current_level < last_level {
                        buy_signals[i] = true;
                    }
                    // 上涨穿越网格线 - 卖出
                    else if current_level > last_level {
                        sell_signals[i] = true;
                    }
                }
                
                last_grid_level = Some(current_level);
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 跳空缺口策略
    /// 
    /// 捕捉开盘跳空缺口，回补缺口时交易
    /// 
    /// # 参数
    /// - `df`: 包含OHLCV数据的DataFrame
    /// - `gap_threshold`: 跳空阈值百分比
    /// - `wait_days`: 等待回补的最大天数（0表示不限制）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, gap_threshold=2.0, wait_days=0))]
    pub fn gap(
        &self,
        df: PyDataFrame,
        gap_threshold: f64,
        wait_days: usize,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let open = data.column("open")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取open列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let high = data.column("high")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取high列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let low = data.column("low")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取low列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let open_values = open.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let high_values = high.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let low_values = low.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let close_values = close.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = open_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        // 记录缺口信息：(缺口日索引, 缺口价格, 是否向上跳空)
        let mut active_gaps: Vec<(usize, f64, bool)> = Vec::new();
        
        for i in 1..len {
            if let (Some(prev_close), Some(curr_open), Some(curr_high), Some(curr_low)) = 
                (close_values.get(i-1), open_values.get(i), high_values.get(i), low_values.get(i)) {
                
                // 向上跳空（开盘价高于昨收）
                let gap_up_pct = (curr_open - prev_close) / prev_close * 100.0;
                if gap_up_pct >= gap_threshold {
                    active_gaps.push((i, prev_close, true));
                }
                
                // 向下跳空（开盘价低于昨收）
                let gap_down_pct = (prev_close - curr_open) / prev_close * 100.0;
                if gap_down_pct >= gap_threshold {
                    active_gaps.push((i, prev_close, false));
                }
                
                // 检查现有缺口是否被回补
                active_gaps.retain(|(gap_day, gap_price, is_up_gap)| {
                    // 检查是否超过等待天数
                    let days_passed = i - gap_day;
                    if wait_days > 0 && days_passed > wait_days {
                        return false; // 超时，移除缺口
                    }
                    
                    if *is_up_gap {
                        // 向上跳空，价格回落到缺口 - 卖出信号
                        if curr_low <= *gap_price {
                            sell_signals[i] = true;
                            return false; // 缺口已回补，移除
                        }
                    } else {
                        // 向下跳空，价格回升到缺口 - 买入信号
                        if curr_high >= *gap_price {
                            buy_signals[i] = true;
                            return false; // 缺口已回补，移除
                        }
                    }
                    
                    true // 缺口仍然有效
                });
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 形态识别策略
    /// 
    /// 识别经典的K线形态（锤子线、射击之星、吞没形态等）
    /// 
    /// # 参数
    /// - `df`: 包含OHLCV数据的DataFrame
    /// - `pattern_type`: 形态类型 ("hammer", "shooting_star", "engulfing")
    /// - `body_ratio`: 实体与影线的最小比例
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, pattern_type="hammer", body_ratio=2.0))]
    pub fn pattern(
        &self,
        df: PyDataFrame,
        pattern_type: &str,
        body_ratio: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let open = data.column("open")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取open列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let high = data.column("high")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取high列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let low = data.column("low")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取low列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        let close = data.column("close")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取close列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let open_values = open.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let high_values = high.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let low_values = low.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let close_values = close.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = open_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        match pattern_type.to_lowercase().as_str() {
            "hammer" => {
                // 锤子线：下影线长，上影线短，实体小，出现在下跌趋势底部
                for i in 1..len {
                    if let (Some(o), Some(h), Some(l), Some(c)) = 
                        (open_values.get(i), high_values.get(i), low_values.get(i), close_values.get(i)) {
                        
                        let body = (c - o).abs();
                        let upper_shadow = h - o.max(c);
                        let lower_shadow = o.min(c) - l;
                        
                        // 锤子线特征：下影线 >= body_ratio * 实体
                        if lower_shadow >= body * body_ratio && upper_shadow < body {
                            buy_signals[i] = true;
                        }
                    }
                }
            },
            "shooting_star" => {
                // 射击之星：上影线长，下影线短，实体小，出现在上涨趋势顶部
                for i in 1..len {
                    if let (Some(o), Some(h), Some(l), Some(c)) = 
                        (open_values.get(i), high_values.get(i), low_values.get(i), close_values.get(i)) {
                        
                        let body = (c - o).abs();
                        let upper_shadow = h - o.max(c);
                        let lower_shadow = o.min(c) - l;
                        
                        // 射击之星特征：上影线 >= body_ratio * 实体
                        if upper_shadow >= body * body_ratio && lower_shadow < body {
                            sell_signals[i] = true;
                        }
                    }
                }
            },
            "engulfing" => {
                // 吞没形态：当前K线实体完全吞没前一根K线实体
                for i in 1..len {
                    if let (Some(o_prev), Some(c_prev), Some(o_curr), Some(c_curr)) = 
                        (open_values.get(i-1), close_values.get(i-1), open_values.get(i), close_values.get(i)) {
                        
                        // 看涨吞没：前一根阴线，当前阳线吞没
                        if c_prev < o_prev && c_curr > o_curr {
                            if o_curr <= c_prev && c_curr >= o_prev {
                                buy_signals[i] = true;
                            }
                        }
                        
                        // 看跌吞没：前一根阳线，当前阴线吞没
                        if c_prev > o_prev && c_curr < o_curr {
                            if o_curr >= c_prev && c_curr <= o_prev {
                                sell_signals[i] = true;
                            }
                        }
                    }
                }
            },
            _ => {}
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 分解策略
    /// 
    /// 基于价格的趋势和残差分解进行交易（均值回归）
    /// 
    /// # 参数
    /// - `df`: 包含OHLCV数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `trend_window`: 趋势窗口大小
    /// - `residual_threshold`: 残差阈值（标准差倍数）
    /// 
    /// # 返回
    /// 包含buy_signal和sell_signal列的DataFrame
    #[pyo3(signature = (df, price_col="close", trend_window=20, residual_threshold=2.0))]
    pub fn trend(
        &self,
        df: PyDataFrame,
        price_col: &str,
        trend_window: usize,
        residual_threshold: f64,
    ) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price_series = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut buy_signals = vec![false; len];
        let mut sell_signals = vec![false; len];
        
        // 计算移动平均作为趋势
        let mut trend = vec![f64::NAN; len];
        for i in trend_window..len {
            let mut sum = 0.0;
            let mut count = 0;
            for j in (i - trend_window)..i {
                if let Some(p) = price_values.get(j) {
                    sum += p;
                    count += 1;
                }
            }
            if count > 0 {
                trend[i] = sum / count as f64;
            }
        }
        
        // 计算残差
        let mut residuals = vec![f64::NAN; len];
        for i in 0..len {
            if let Some(price) = price_values.get(i) {
                if !trend[i].is_nan() {
                    residuals[i] = price - trend[i];
                }
            }
        }
        
        // 计算残差的标准差
        let valid_residuals: Vec<f64> = residuals.iter()
            .filter_map(|&r| if r.is_nan() { None } else { Some(r) })
            .collect();
        
        if valid_residuals.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("无法计算残差"));
        }
        
        let mean = valid_residuals.iter().sum::<f64>() / valid_residuals.len() as f64;
        let variance = valid_residuals.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / valid_residuals.len() as f64;
        let std_dev = variance.sqrt();
        
        // 残差超过阈值时生成反向信号（均值回归）
        for i in trend_window..len {
            if !residuals[i].is_nan() {
                let z_score = (residuals[i] - mean) / std_dev;
                
                // 价格显著低于趋势 - 买入
                if z_score <= -residual_threshold {
                    buy_signals[i] = true;
                }
                // 价格显著高于趋势 - 卖出
                else if z_score >= residual_threshold {
                    sell_signals[i] = true;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("buy_signal"), buy_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加买入信号失败: {}", e)))?;
        data.with_column(Series::new(PlSmallStr::from_str("sell_signal"), sell_signals))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加卖出信号失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }
}
