use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PySeries;

// ============================================================================
// 辅助函数和枚举定义 (内部 Rust 函数)
// ============================================================================

/// 移动平均类型
#[derive(Clone, Copy)]
pub enum MAType {
    SMA, // 简单移动平均
    EMA, // 指数移动平均
    WMA, // 加权移动平均
}

/// 通用移动平均计算函数
/// 
/// 支持多种移动平均类型（SMA, EMA, DEMA, TEMA 等）
pub fn calculate_ma(values: &[f64], period: usize, ma_type: MAType) -> Vec<Option<f64>> {
    let len = values.len();
    let mut result = vec![None; len];
    
    if period == 0 || period > len {
        return result;
    }
    
    match ma_type {
        MAType::SMA => {
            // 滑动窗口计算
            let mut period_total = 0.0;
            let lookback = period - 1;
            
            // 初始化累加和（前 period-1 个值）
            for i in 0..lookback {
                period_total += values[i];
            }
            
            // 滑动窗口增量更新
            let mut trailing_idx = 0;
            for i in lookback..len {
                period_total += values[i];              // 加入新值
                result[i] = Some(period_total / period as f64); // 计算平均
                period_total -= values[trailing_idx];   // 移除旧值
                trailing_idx += 1;
            }
        },
        MAType::EMA => {
            // 指数平滑计算
            let alpha = 2.0 / (period as f64 + 1.0);
            let one_minus_alpha = 1.0 - alpha;
            
            // 使用 SMA 作为初始值
            let mut sum = 0.0;
            for i in 0..period {
                sum += values[i];
            }
            let mut ema = sum / period as f64;
            result[period - 1] = Some(ema);
            
            // 指数平滑更新
            for i in period..len {
                ema = alpha * values[i] + one_minus_alpha * ema;
                result[i] = Some(ema);
            }
        },
        MAType::WMA => {
            // 加权移动平均计算
            // 权重为 1, 2, 3, ..., period，最近的值权重最大
            let weight_sum = (period * (period + 1)) as f64 / 2.0;
            let lookback = period - 1;
            
            for i in lookback..len {
                let mut weighted_sum = 0.0;
                for j in 0..period {
                    let weight = (j + 1) as f64;
                    weighted_sum += values[i - lookback + j] * weight;
                }
                result[i] = Some(weighted_sum / weight_sum);
            }
        }
    }
    
    result
}

// RSI计算函数
fn calculate_rsi(values: &[f64], period: usize) -> Vec<f64> {
    let len = values.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        return result;
    }
    
    let mut gains = Vec::with_capacity(len - 1);
    let mut losses = Vec::with_capacity(len - 1);
    
    // 计算价格变化
    for i in 1..len {
        let change = values[i] - values[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    // 计算初始平均值
    let mut avg_gain = gains.iter().take(period).sum::<f64>() / period as f64;
    let mut avg_loss = losses.iter().take(period).sum::<f64>() / period as f64;
    
    let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
    result[period] = 100.0 - 100.0 / (1.0 + rs);
    
    // 使用指数平滑计算后续值
    let alpha = 1.0 / period as f64;
    let one_minus_alpha = 1.0 - alpha;
    
    for i in (period + 1)..(len) {
        avg_gain = alpha * gains[i - 1] + one_minus_alpha * avg_gain;
        avg_loss = alpha * losses[i - 1] + one_minus_alpha * avg_loss;
        
        let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
        result[i] = 100.0 - 100.0 / (1.0 + rs);
    }
    
    result
}

// ATR计算函数
fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        return result;
    }
    
    // 直接在一次遍历中计算TR和累积求和,避免额外Vec分配
    // 第一个TR值
    let first_tr = high[0] - low[0];
    let mut tr_sum = first_tr;
    
    // 累积前period个TR值的和
    for i in 1..period {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        let tr = tr1.max(tr2).max(tr3);
        tr_sum += tr;
    }
    
    // 第一个ATR值 = SMA of TR
    result[period - 1] = tr_sum / period as f64;
    
    // 使用Wilder的平滑方法: ATR = ((period-1)*prev_ATR + current_TR) / period
    let smoothing_factor = 1.0 / period as f64;
    let retention_factor = (period - 1) as f64 / period as f64;
    
    for i in period..len {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        let tr = tr1.max(tr2).max(tr3);
        
        // 增量更新
        result[i] = result[i - 1] * retention_factor + tr * smoothing_factor;
    }
    
    result
}

// 蜡烛图辅助函数
fn candle_metrics(open: f64, high: f64, low: f64, close: f64) -> (f64, f64, f64, f64) {
    let body = (close - open).abs();
    let upper_shadow = high - open.max(close);
    let lower_shadow = open.min(close) - low;
    let range = high - low;
    (body, upper_shadow, lower_shadow, range)
}

fn is_long_body(body: f64, range: f64) -> bool {
    body > range * 0.6
}

fn is_short_body(body: f64, range: f64) -> bool {
    body < range * 0.3
}

fn is_doji_body(body_size: f64, range: f64) -> bool {
    body_size < range * 0.1
}

/// 布林带 (BBAND)
/// 计算上轨、中轨、下轨
#[pyfunction]
#[pyo3(signature = (series, period=20, std_dev=2.0))]
pub fn bband(series: PySeries, period: usize, std_dev: f64) -> PyResult<(PySeries, PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut upper_out = vec![f64::NAN; len];
    let mut middle_out = vec![f64::NAN; len];
    let mut lower_out = vec![f64::NAN; len];
    
    if period == 0 || period > len {
        let base_name = s.name();
        return Ok((
            PySeries(Series::new(PlSmallStr::from_str(&format!("{}_bb_upper", base_name)), upper_out)),
            PySeries(Series::new(PlSmallStr::from_str(&format!("{}_bb_middle", base_name)), middle_out)),
            PySeries(Series::new(PlSmallStr::from_str(&format!("{}_bb_lower", base_name)), lower_out))
        ));
    }
    
    // 快速路径：连续内存访问
    if let Ok(input) = values.cont_slice() {
        let lookback = period - 1;
        let period_f64 = period as f64;
        let inv_period = 1.0 / period_f64;
        
        // 初始化：计算第一个窗口的 sum 和 sum_sq
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..period {
            let val = input[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // 第一个输出
        let sma = sum * inv_period;
        let variance = (sum_sq * inv_period) - (sma * sma);
        let std = variance.max(0.0).sqrt(); // 避免负数（浮点精度问题）
        upper_out[lookback] = sma + std_dev * std;
        middle_out[lookback] = sma;
        lower_out[lookback] = sma - std_dev * std;
        
        // 滑动窗口
        for i in period..len {
            let old_val = input[i - period];
            let new_val = input[i];
            
            // 更新 sum 和 sum_sq
            sum = sum - old_val + new_val;
            sum_sq = sum_sq - old_val * old_val + new_val * new_val;
            
            // 计算 SMA 和标准差
            let sma = sum * inv_period;
            let variance = (sum_sq * inv_period) - (sma * sma);
            let std = variance.max(0.0).sqrt();
            
            upper_out[i] = sma + std_dev * std;
            middle_out[i] = sma;
            lower_out[i] = sma - std_dev * std;
        }
    } else {
        // Fallback: 非连续数组
        let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        let lookback = period - 1;
        let period_f64 = period as f64;
        let inv_period = 1.0 / period_f64;
        
        // 初始化：计算第一个窗口的 sum 和 sum_sq
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..period {
            let val = vec_values[i];
            sum += val;
            sum_sq += val * val;
        }
        
        // 第一个输出
        let sma = sum * inv_period;
        let variance = (sum_sq * inv_period) - (sma * sma);
        let std = variance.max(0.0).sqrt();
        upper_out[lookback] = sma + std_dev * std;
        middle_out[lookback] = sma;
        lower_out[lookback] = sma - std_dev * std;
        
        // 滑动窗口
        for i in period..len {
            let old_val = vec_values[i - period];
            let new_val = vec_values[i];
            
            sum = sum - old_val + new_val;
            sum_sq = sum_sq - old_val * old_val + new_val * new_val;
            
            let sma = sum * inv_period;
            let variance = (sum_sq * inv_period) - (sma * sma);
            let std = variance.max(0.0).sqrt();
            
            upper_out[i] = sma + std_dev * std;
            middle_out[i] = sma;
            lower_out[i] = sma - std_dev * std;
        }
    }
    
    let base_name = s.name();
    let upper_series = Series::new(PlSmallStr::from_str(&format!("{}_bb_upper", base_name)), upper_out);
    let middle_series = Series::new(PlSmallStr::from_str(&format!("{}_bb_middle", base_name)), middle_out);
    let lower_series = Series::new(PlSmallStr::from_str(&format!("{}_bb_lower", base_name)), lower_out);
    
    Ok((PySeries(upper_series), PySeries(middle_series), PySeries(lower_series)))
}

/// 双指数移动平均线 (DEMA)
#[pyfunction] 
#[pyo3(signature = (series, period=20))]
pub fn dema(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    // 计算第一次EMA
    let ema1_values = calculate_ma(&vec_values, period, MAType::EMA);
    let ema1_f64: Vec<f64> = ema1_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
    
    // 计算第二次EMA
    let ema2_values = calculate_ma(&ema1_f64, period, MAType::EMA);
    let ema2_f64: Vec<f64> = ema2_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
    
    // DEMA = 2 * EMA1 - EMA2
    let dema_values: Vec<f64> = ema1_f64.iter().zip(ema2_f64.iter())
        .map(|(&ema1, &ema2)| 2.0 * ema1 - ema2)
        .collect();
    
    let result = Series::new(s.name().clone(), dema_values);
    Ok(PySeries(result))
}

/// 指数移动平均线 (EMA)
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn ema(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    if period == 0 || period > len {
        return Ok(PySeries(Series::new(s.name().clone(), output)));
    }
    
    // 快速路径：连续内存访问
    if let Ok(input) = values.cont_slice() {
        let alpha = 2.0 / (period + 1) as f64;
        let one_minus_alpha = 1.0 - alpha;
        
        // 使用 SMA 作为初始值
        let mut sum = 0.0;
        for i in 0..period {
            sum += input[i];
        }
        let mut ema = sum / period as f64;
        output[period - 1] = ema;
        
        // 指数平滑计算
        for i in period..len {
            ema = alpha * input[i] + one_minus_alpha * ema;
            output[i] = ema;
        }
    } else {
        // Fallback: 处理非连续数组
        let vec_values: Vec<f64> = values.into_iter()
            .map(|opt| opt.unwrap_or(0.0))
            .collect();
        
        let alpha = 2.0 / (period + 1) as f64;
        let one_minus_alpha = 1.0 - alpha;
        
        let mut sum = 0.0;
        for i in 0..period {
            sum += vec_values[i];
        }
        let mut ema = sum / period as f64;
        output[period - 1] = ema;
        
        for i in period..len {
            ema = alpha * vec_values[i] + one_minus_alpha * ema;
            output[i] = ema;
        }
    }
    
    Ok(PySeries(Series::new(s.name().clone(), output)))
}

/// 考夫曼自适应移动平均 (KAMA)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn kama(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut result = vec![None; len];
    
    if len < period + 1 {
        let result_series = Series::new(s.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // Zero-copy optimization
    if let Ok(arr) = values.cont_slice() {
        // Pre-compute all price differences (avoid repeated calculation)
        let mut diffs = vec![0.0; len];
        for i in 1..len {
            diffs[i] = (arr[i] - arr[i - 1]).abs();
        }
        
        // Initialize first volatility window
        let mut volatility = 0.0;
        for i in 1..=period {
            volatility += diffs[i];
        }
        
        // First KAMA value
        let change = (arr[period] - arr[0]).abs();
        let er = if volatility != 0.0 { change / volatility } else { 0.0 };
        
        const FASTEST_SC: f64 = 2.0 / 3.0;   // EMA period 2
        const SLOWEST_SC: f64 = 2.0 / 31.0;  // EMA period 30
        const SC_RANGE: f64 = FASTEST_SC - SLOWEST_SC;
        
        let _sc = (er * SC_RANGE + SLOWEST_SC).powi(2);
        let mut kama = arr[period];
        result[period] = Some(kama);
        
        // Sliding window for volatility
        for i in (period + 1)..len {
            // Update volatility: subtract oldest, add newest
            volatility = volatility - diffs[i - period] + diffs[i];
            
            // Calculate ER with new volatility
            let change = (arr[i] - arr[i - period]).abs();
            let er = if volatility != 0.0 { change / volatility } else { 0.0 };
            
            // Calculate SC and update KAMA
            let sc = (er * SC_RANGE + SLOWEST_SC).powi(2);
            kama = kama + sc * (arr[i] - kama);
            result[i] = Some(kama);
        }
    } else {
        // Fallback
        let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        
        let mut diffs = vec![0.0; len];
        for i in 1..len {
            diffs[i] = (vec_values[i] - vec_values[i - 1]).abs();
        }
        
        let mut volatility = 0.0;
        for i in 1..=period {
            volatility += diffs[i];
        }
        
        let change = (vec_values[period] - vec_values[0]).abs();
        let er = if volatility != 0.0 { change / volatility } else { 0.0 };
        
        const FASTEST_SC: f64 = 2.0 / 3.0;
        const SLOWEST_SC: f64 = 2.0 / 31.0;
        const SC_RANGE: f64 = FASTEST_SC - SLOWEST_SC;
        
        let _sc = (er * SC_RANGE + SLOWEST_SC).powi(2);
        let mut kama = vec_values[period];
        result[period] = Some(kama);
        
        for i in (period + 1)..len {
            volatility = volatility - diffs[i - period] + diffs[i];
            
            let change = (vec_values[i] - vec_values[i - period]).abs();
            let er = if volatility != 0.0 { change / volatility } else { 0.0 };
            
            let sc = (er * SC_RANGE + SLOWEST_SC).powi(2);
            kama = kama + sc * (vec_values[i] - kama);
            result[i] = Some(kama);
        }
    }
    
    let result_series = Series::new(s.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 移动平均 (MA) - 支持 SMA、EMA、WMA
#[pyfunction]
#[pyo3(signature = (series, period=20, ma_type="SMA"))]
pub fn ma(series: PySeries, period: usize, ma_type: &str) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("输入必须是数值类型: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect();
    
    // 解析 MA 类型
    let ma_type_enum = match ma_type.to_uppercase().as_str() {
        "SMA" => MAType::SMA,
        "EMA" => MAType::EMA,
        "WMA" => MAType::WMA,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("不支持的 MA 类型: {}。支持的类型: SMA, EMA, WMA", ma_type)
        )),
    };
    
    let ma_values = calculate_ma(&vec_values, period, ma_type_enum);
    let result = Series::new(s.name().clone(), ma_values);
    
    Ok(PySeries(result))
}

/// MESA 自适应移动平均 (MAMA) - 完整版本
#[pyfunction]
#[pyo3(signature = (series, fast_limit=0.5, slow_limit=0.05))]
pub fn mama(series: PySeries, fast_limit: f64, slow_limit: f64) -> PyResult<(PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    let n = vec_values.len();
    
    let (mama_values, fama_values) = {
        let mut mama = vec![None; n];
        let mut fama = vec![None; n];
        let mut period = vec![0.0; n];
        let mut smooth = vec![0.0; n];
        let mut detrender = vec![0.0; n];
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];
        let mut i2 = vec![0.0; n];
        let mut q2 = vec![0.0; n];
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        let mut phase = vec![0.0; n];
        let mut delta_phase = vec![0.0; n];
        
        if n < 7 {
            // 数据不足，返回原始值
            for i in 0..n {
                mama[i] = Some(vec_values[i]);
                fama[i] = Some(vec_values[i]);
            }
        } else {
        
        // 初始化前7个值
        for i in 0..7 {
            mama[i] = Some(vec_values[i]);
            fama[i] = Some(vec_values[i]);
        }
        
        // MESA算法主循环
        for i in 7..n {
            // 平滑价格数据 (4阶数字滤波器)
            smooth[i] = (4.0 * vec_values[i] + 3.0 * vec_values[i-1] + 2.0 * vec_values[i-2] + vec_values[i-3]) / 10.0;
            
            // 计算周期测量的去趋势化
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54);
            
            // 计算同相和正交分量
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54);
            i1[i] = detrender[i-3];
            
            // 提前Hilbert变换器的输入
            let j_i = (0.0962 * i1[i] + 0.5769 * i1[i-2] - 0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54);
            let j_q = (0.0962 * q1[i] + 0.5769 * q1[i-2] - 0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54);
            
            // 引导帧的同相和正交分量
            i2[i] = i1[i] - j_q;
            q2[i] = q1[i] + j_i;
            
            // 平滑I和Q分量
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1];
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1];
            
            // 霍模德调制鉴别器
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1];
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1];
            
            re[i] = 0.2 * re[i] + 0.8 * re[i-1];
            im[i] = 0.2 * im[i] + 0.8 * im[i-1];
            
            // 计算瞬时周期
            if i1[i] != 0.0 && q1[i] != 0.0 {
                period[i] = 6.28318 / im[i].atan2(re[i]);
            }
            
            if period[i] > 1.5 * period[i-1] {
                period[i] = 1.5 * period[i-1];
            }
            if period[i] < 0.67 * period[i-1] {
                period[i] = 0.67 * period[i-1];
            }
            if period[i] < 6.0 {
                period[i] = 6.0;
            }
            if period[i] > 50.0 {
                period[i] = 50.0;
            }
            
            period[i] = 0.2 * period[i] + 0.8 * period[i-1];
            
            // 计算相位
            if i1[i] != 0.0 {
                phase[i] = (q1[i] / i1[i]).atan() * 57.2958; // 转换为度数
            }
            
            delta_phase[i] = phase[i-1] - phase[i];
            if delta_phase[i] < 1.0 {
                delta_phase[i] = 1.0;
            }
            
            // 计算自适应因子
            let alpha = fast_limit / delta_phase[i];
            let alpha = if alpha < slow_limit { slow_limit } else { alpha };
            let alpha = if alpha > fast_limit { fast_limit } else { alpha };
            
            // 计算MAMA和FAMA
            let prev_mama = mama[i-1].unwrap_or(vec_values[i]);
            let prev_fama = fama[i-1].unwrap_or(vec_values[i]);
            
            let new_mama = alpha * vec_values[i] + (1.0 - alpha) * prev_mama;
            let new_fama = 0.5 * alpha * new_mama + (1.0 - 0.5 * alpha) * prev_fama;
            
            mama[i] = Some(new_mama);
            fama[i] = Some(new_fama);
        }
        }
        
        (mama, fama)
    };
    
    let base_name = s.name();
    let mama_series = Series::new(PlSmallStr::from_str(&format!("{}_mama", base_name)), mama_values);
    let fama_series = Series::new(PlSmallStr::from_str(&format!("{}_fama", base_name)), fama_values);
    
    Ok((PySeries(mama_series), PySeries(fama_series)))
}

/// 变周期移动平均 (MAVP)
#[pyfunction]
#[pyo3(signature = (series, periods, min_period=2, max_period=30))]
pub fn mavp(series: PySeries, periods: PySeries, min_period: usize, max_period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let p: Series = periods.into();
    
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Values must be numeric: {}", e)))?;
    let period_values = p.i64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Periods must be integer: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    let vec_periods: Vec<usize> = period_values.into_iter()
        .map(|opt| opt.unwrap_or(10) as usize)
        .map(|p| p.clamp(min_period, max_period))
        .collect();
    
    let mavp_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in 0..vec_values.len() {
            let period = vec_periods[i];
            if i + 1 >= period {
                let sum: f64 = vec_values[i + 1 - period..=i].iter().sum();
                result[i] = Some(sum / period as f64);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), mavp_values);
    Ok(PySeries(result))
}

/// 简单移动平均线 (SMA)
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn sma(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    if period == 0 || period > len {
        return Ok(PySeries(Series::new(s.name().clone(), output)));
    }
    
    // 快速路径：连续内存访问
    if let Ok(input) = values.cont_slice() {
        // 初始化累加和
        let mut period_total = 0.0;
        let lookback = period - 1;
        let inv_period = 1.0 / period as f64;
        
        for i in 0..lookback {
            period_total += input[i];
        }
        
        // 滑动窗口增量计算
        let mut trailing_idx = 0;
        for i in lookback..len {
            period_total += input[i];              // 加入新值
            output[i] = period_total * inv_period; // 乘法比除法快
            period_total -= input[trailing_idx];   // 移除旧值
            trailing_idx += 1;
        }
    } else {
        // Fallback: 处理非连续数组（带 null 检查）
        let vec_values: Vec<f64> = values.into_iter()
            .map(|opt| opt.unwrap_or(0.0))
            .collect();
        
        let mut period_total = 0.0;
        let lookback = period - 1;
        let inv_period = 1.0 / period as f64;
        
        for i in 0..lookback {
            period_total += vec_values[i];
        }
        
        let mut trailing_idx = 0;
        for i in lookback..len {
            period_total += vec_values[i];
            output[i] = period_total * inv_period;
            period_total -= vec_values[trailing_idx];
            trailing_idx += 1;
        }
    }
    
    Ok(PySeries(Series::new(s.name().clone(), output)))
}

/// T3移动平均 (T3)
#[pyfunction]
#[pyo3(signature = (series, period=14, volume_factor=0.7))]
pub fn t3(series: PySeries, period: usize, volume_factor: f64) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let t3_values: Vec<Option<f64>> = {
        // T3 需要6次EMA计算
        let c1 = -volume_factor.powi(3);
        let c2 = 3.0 * volume_factor.powi(2) + 3.0 * volume_factor.powi(3);
        let c3 = -6.0 * volume_factor.powi(2) - 3.0 * volume_factor - 3.0 * volume_factor.powi(3);
        let c4 = 1.0 + 3.0 * volume_factor + volume_factor.powi(3) + 3.0 * volume_factor.powi(2);
        
        // 第一次EMA
        let ema1_values = calculate_ma(&vec_values, period, MAType::EMA);
        let ema1: Vec<f64> = ema1_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        // 第二次EMA
        let ema2_values = calculate_ma(&ema1, period, MAType::EMA);
        let ema2: Vec<f64> = ema2_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        // 第三次EMA
        let ema3_values = calculate_ma(&ema2, period, MAType::EMA);
        let ema3: Vec<f64> = ema3_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        // 第四次EMA
        let ema4_values = calculate_ma(&ema3, period, MAType::EMA);
        let ema4: Vec<f64> = ema4_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        // 第五次EMA
        let ema5_values = calculate_ma(&ema4, period, MAType::EMA);
        let ema5: Vec<f64> = ema5_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        // 第六次EMA
        let ema6_values = calculate_ma(&ema5, period, MAType::EMA);
        let ema6: Vec<f64> = ema6_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        // 计算T3 = c1*EMA6 + c2*EMA5 + c3*EMA4 + c4*EMA3
        (0..vec_values.len()).map(|i| {
            if i >= period * 6 {
                Some(c1 * ema6[i] + c2 * ema5[i] + c3 * ema4[i] + c4 * ema3[i])
            } else {
                None
            }
        }).collect()
    };
    
    let result = Series::new(s.name().clone(), t3_values);
    Ok(PySeries(result))
}

/// 三重指数移动平均 (TEMA)
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn tema(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let tema_values: Vec<f64> = {
        // 计算第一次EMA
        let ema1_values = calculate_ma(&vec_values, period, MAType::EMA);
        let ema1_f64: Vec<f64> = ema1_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
        
        // 计算第二次EMA
        let ema2_values = calculate_ma(&ema1_f64, period, MAType::EMA);
        let ema2_f64: Vec<f64> = ema2_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
        
        // 计算第三次EMA
        let ema3_values = calculate_ma(&ema2_f64, period, MAType::EMA);
        let ema3_f64: Vec<f64> = ema3_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
        
        // TEMA = 3*EMA1 - 3*EMA2 + EMA3
        ema1_f64.iter().zip(ema2_f64.iter()).zip(ema3_f64.iter())
            .map(|((&ema1, &ema2), &ema3)| 3.0 * ema1 - 3.0 * ema2 + ema3)
            .collect()
    };
    
    let result = Series::new(s.name().clone(), tema_values);
    Ok(PySeries(result))
}

/// 三角移动平均 (TRIMA)
/// TRIMA(period=4) = (1*a + 2*b + 2*c + 1*d) / 6
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn trima(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    if len < period {
        let result = Series::new(s.name().clone(), output);
        return Ok(PySeries(result));
    }
    
    if let Ok(input) = values.cont_slice() {
        let lookback = period - 1;
        
        // TA-Lib weighted calculation
        // Odd period: weights = 1,2,3,...,n,...,3,2,1
        // Even period: weights = 1,2,3,...,n,n,...,3,2,1
        let (factor, divisor) = if period % 2 == 1 {
            let f = (period + 1) / 2;
            (f, (f * f) as f64)
        } else {
            let f = period / 2 + 1;
            (f, (f * (f - 1)) as f64)
        };
        
        // Initialize first TRIMA with weighted sum
        let mut numerator = 0.0;
        for i in 0..period {
            let weight = if i < factor {
                (i + 1) as f64
            } else {
                (period - i) as f64
            };
            numerator += input[i] * weight;
        }
        
        output[lookback] = numerator / divisor;
        
        // Slide window: TA-Lib uses 4 adjustments per iteration
        let mut trailing_idx = 0usize;
        let middle_idx = (period - 1) / 2;
        
        for today_idx in (lookback + 1)..len {
            let numerator_sub = input[trailing_idx];
            trailing_idx += 1;
            let numerator_add = input[today_idx];
            
            // Core sliding window logic
            numerator += numerator_add - numerator_sub;
            
            // Extra adjustment for triangular weighting
            numerator -= input[trailing_idx + middle_idx];
            if period % 2 == 0 {
                numerator += input[trailing_idx + middle_idx - 1];
            } else {
                numerator += input[trailing_idx + middle_idx];
            }
            
            output[today_idx] = numerator / divisor;
        }
    } else {
        // Fallback
        let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        let sma1_values = calculate_ma(&vec_values, period, MAType::SMA);
        let sma1_f64: Vec<f64> = sma1_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
        let sma2_period = if period % 2 == 1 { (period + 1) / 2 } else { period / 2 + 1 };
        let sma2_values = calculate_ma(&sma1_f64, sma2_period, MAType::SMA);
        for (i, val) in sma2_values.iter().enumerate() {
            output[i] = val.unwrap_or(f64::NAN);
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 加权移动平均 (WMA)
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn wma(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    if len < period {
        let result = Series::new(s.name().clone(), output);
        return Ok(PySeries(result));
    }
    
    let weight_sum = (period * (period + 1) / 2) as f64;
    let inv_weight_sum = 1.0 / weight_sum;
    let period_f64 = period as f64;
    
    // 连续内存访问
    if let Ok(input) = values.cont_slice() {
        // 第一个窗口
        let mut weighted_sum = 0.0;
        for j in 0..period {
            weighted_sum += input[j] * (j + 1) as f64;
        }
        output[period - 1] = weighted_sum * inv_weight_sum;
        
        // 滑动窗口
        let mut sum_values: f64 = input[..period].iter().sum();
        
        for i in period..len {
            weighted_sum = weighted_sum - sum_values + input[i] * period_f64;
            sum_values = sum_values - input[i - period] + input[i];
            output[i] = weighted_sum * inv_weight_sum;
        }
    } else {
        // fallback
        let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        
        let mut weighted_sum = 0.0;
        for j in 0..period {
            weighted_sum += vec_values[j] * (j + 1) as f64;
        }
        output[period - 1] = weighted_sum * inv_weight_sum;
        
        let mut sum_values = vec_values[..period].iter().sum::<f64>();
        
        for i in period..len {
            weighted_sum = weighted_sum - sum_values + vec_values[i] * period_f64;
            sum_values = sum_values - vec_values[i - period] + vec_values[i];
            output[i] = weighted_sum * inv_weight_sum;
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 中点 (MIDPOINT)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn midpoint(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period > len {
        let result_series = Series::new(s.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    let inv_2 = 0.5;
    
    // 连续内存访问
    if let Ok(arr) = values.cont_slice() {
        // 初始化窗口的最大值和最小值
        let window = &arr[0..period];
        let mut max_val = window[0];
        let mut min_val = window[0];
        
        // 初始化第一个窗口的max/min
        for &v in &window[1..] {
            if v > max_val { max_val = v; }
            if v < min_val { min_val = v; }
        }
        result[period - 1] = (max_val + min_val) * inv_2;
        
        // 滑动窗口
        for i in period..len {
            let new_val = arr[i];
            let old_val = arr[i - period];
            
            // 如果新值可能改变max/min,更新它们
            if new_val > max_val || old_val == max_val {
                // 需要重新扫描窗口
                max_val = arr[i - period + 1];
                for j in (i - period + 2)..=i {
                    if arr[j] > max_val { max_val = arr[j]; }
                }
            } else if new_val > max_val {
                max_val = new_val;
            }
            
            if new_val < min_val || old_val == min_val {
                // 需要重新扫描窗口
                min_val = arr[i - period + 1];
                for j in (i - period + 2)..=i {
                    if arr[j] < min_val { min_val = arr[j]; }
                }
            } else if new_val < min_val {
                min_val = new_val;
            }
            
            result[i] = (max_val + min_val) * inv_2;
        }
    } else {
        // 后备路径: 使用Vec
        let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
        
        for i in (period - 1)..len {
            let start = i + 1 - period;
            let mut max_val = vec_values[start];
            let mut min_val = vec_values[start];
            
            for j in (start + 1)..=i {
                if vec_values[j] > max_val { max_val = vec_values[j]; }
                if vec_values[j] < min_val { min_val = vec_values[j]; }
            }
            
            result[i] = (max_val + min_val) * inv_2;
        }
    }
    
    let result_series = Series::new(s.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 中间价格 (MIDPRICE) - 针对high/low序列
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn midprice_hl(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    
    let len = high_vals.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period > len {
        let result_series = Series::new(h.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        // 在窗口中搜索最高价和最低价
        for i in (period - 1)..len {
            let start = i + 1 - period;
            let mut max_high = h_arr[start];
            let mut min_low = l_arr[start];
            
            // 紧密循环查找窗口内的最高价和最低价
            for j in (start + 1)..=i {
                if h_arr[j] > max_high { max_high = h_arr[j]; }
                if l_arr[j] < min_low { min_low = l_arr[j]; }
            }
            
            result[i] = (max_high + min_low) / 2.0;
        }
    } else {
        // 后备路径: 使用Vec
        let high_vals_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vals_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        for i in (period - 1)..len {
            let start = i + 1 - period;
            let mut max_high = high_vals_vec[start];
            let mut min_low = low_vals_vec[start];
            
            for j in (start + 1)..=i {
                if high_vals_vec[j] > max_high { max_high = high_vals_vec[j]; }
                if low_vals_vec[j] < min_low { min_low = low_vals_vec[j]; }
            }
            
            result[i] = (max_high + min_low) / 2.0;
        }
    }
    
    let result_series = Series::new(h.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 抛物线SAR (SAR)
#[pyfunction]
#[pyo3(signature = (high, low, acceleration=0.02, maximum=0.2))]
pub fn sar(high: PySeries, low: PySeries, acceleration: f64, maximum: f64) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let len = high_vals.len();
    
    let mut sar_values = vec![None; len];
    
    if len < 2 {
        let result = Series::new(h.name().clone(), sar_values);
        return Ok(PySeries(result));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        // 初始化SAR参数
        let mut sar = l_arr[0];
        let mut ep = h_arr[0]; // 极值点
        let mut af = acceleration; // 加速因子
        let mut is_bull = true; // 是否为上升趋势
        
        sar_values[0] = Some(sar);
        
        for i in 1..len {
            // 更新SAR
            sar = sar + af * (ep - sar);
            
            if is_bull {
                // 上升趋势
                if l_arr[i] <= sar {
                    // 趋势反转
                    is_bull = false;
                    sar = ep;
                    ep = l_arr[i];
                    af = acceleration;
                } else {
                    // 继续上升趋势
                    if h_arr[i] > ep {
                        ep = h_arr[i];
                        af = (af + acceleration).min(maximum);
                    }
                }
            } else {
                // 下降趋势
                if h_arr[i] >= sar {
                    // 趋势反转
                    is_bull = true;
                    sar = ep;
                    ep = h_arr[i];
                    af = acceleration;
                } else {
                    // 继续下降趋势
                    if l_arr[i] < ep {
                        ep = l_arr[i];
                        af = (af + acceleration).min(maximum);
                    }
                }
            }
            
            sar_values[i] = Some(sar);
        }
    } else {
        // Fallback
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut sar = low_vec[0];
        let mut ep = high_vec[0];
        let mut af = acceleration;
        let mut is_bull = true;
        
        sar_values[0] = Some(sar);
        
        for i in 1..len {
            sar = sar + af * (ep - sar);
            
            if is_bull {
                if low_vec[i] <= sar {
                    is_bull = false;
                    sar = ep;
                    ep = low_vec[i];
                    af = acceleration;
                } else {
                    if high_vec[i] > ep {
                        ep = high_vec[i];
                        af = (af + acceleration).min(maximum);
                    }
                }
            } else {
                if high_vec[i] >= sar {
                    is_bull = true;
                    sar = ep;
                    ep = high_vec[i];
                    af = acceleration;
                } else {
                    if low_vec[i] < ep {
                        ep = low_vec[i];
                        af = (af + acceleration).min(maximum);
                    }
                }
            }
            
            sar_values[i] = Some(sar);
        }
    }
    
    let result = Series::new(h.name().clone(), sar_values);
    Ok(PySeries(result))
}

// 抛物线SAR扩展版本
#[pyfunction]
#[pyo3(signature = (high, low, startvalue=None, offsetonreverse=None, accelerationinitlong=None, accelerationlong=None, accelerationmaxlong=None, accelerationinitshort=None, accelerationshort=None, accelerationmaxshort=None))]
pub fn sarext(
    high: PySeries, 
    low: PySeries, 
    startvalue: Option<f64>, 
    offsetonreverse: Option<f64>,
    accelerationinitlong: Option<f64>,
    accelerationlong: Option<f64>,
    accelerationmaxlong: Option<f64>,
    accelerationinitshort: Option<f64>,
    accelerationshort: Option<f64>,
    accelerationmaxshort: Option<f64>
) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let len = high_vals.len();
    
    // 设置默认参数
    let start_value = startvalue.unwrap_or(0.0);
    let offset_on_reverse = offsetonreverse.unwrap_or(0.0);
    let accel_init_long = accelerationinitlong.unwrap_or(0.02);
    let accel_long = accelerationlong.unwrap_or(0.02);
    let accel_max_long = accelerationmaxlong.unwrap_or(0.2);
    let accel_init_short = accelerationinitshort.unwrap_or(0.02);
    let accel_short = accelerationshort.unwrap_or(0.02);
    let accel_max_short = accelerationmaxshort.unwrap_or(0.2);
    
    let mut sar_values = vec![None; len];
    
    if len < 2 {
        let result = Series::new(h.name().clone(), sar_values);
        return Ok(PySeries(result));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        // 初始化SAR参数
        let initial_sar = if start_value == 0.0 {
            l_arr[0]
        } else {
            start_value
        };
        let mut sar = initial_sar;
        let mut ep = h_arr[0]; // 极值点
        let mut af = accel_init_long; // 加速因子
        let mut is_bull = true; // 是否为上升趋势
        
        sar_values[0] = Some(sar);
        
        for i in 1..len {
            // 计算新的SAR
            sar = sar + af * (ep - sar);
            
            if is_bull {
                // 上升趋势 (多头)
                if l_arr[i] <= sar {
                    // 趋势反转到下降
                    is_bull = false;
                    sar = ep + offset_on_reverse;
                    ep = l_arr[i];
                    af = accel_init_short;
                } else {
                    // 继续上升趋势
                    if h_arr[i] > ep {
                        ep = h_arr[i];
                        af = (af + accel_long).min(accel_max_long);
                    }
                    // 确保SAR不超过前两个周期的低点
                    if i >= 2 {
                        sar = sar.min(l_arr[i-1]).min(l_arr[i-2]);
                    } else if i >= 1 {
                        sar = sar.min(l_arr[i-1]);
                    }
                }
            } else {
                // 下降趋势 (空头)
                if h_arr[i] >= sar {
                    // 趋势反转到上升
                    is_bull = true;
                    sar = ep - offset_on_reverse;
                    ep = h_arr[i];
                    af = accel_init_long;
                } else {
                    // 继续下降趋势
                    if l_arr[i] < ep {
                        ep = l_arr[i];
                        af = (af + accel_short).min(accel_max_short);
                    }
                    // 确保SAR不低于前两个周期的高点
                    if i >= 2 {
                        sar = sar.max(h_arr[i-1]).max(h_arr[i-2]);
                    } else if i >= 1 {
                        sar = sar.max(h_arr[i-1]);
                    }
                }
            }
            
            sar_values[i] = Some(sar);
        }
    } else {
        // Fallback
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let initial_sar = if start_value == 0.0 {
            low_vec[0]
        } else {
            start_value
        };
        let mut sar = initial_sar;
        let mut ep = high_vec[0];
        let mut af = accel_init_long;
        let mut is_bull = true;
        
        sar_values[0] = Some(sar);
        
        for i in 1..len {
            sar = sar + af * (ep - sar);
            
            if is_bull {
                if low_vec[i] <= sar {
                    is_bull = false;
                    sar = ep + offset_on_reverse;
                    ep = low_vec[i];
                    af = accel_init_short;
                } else {
                    if high_vec[i] > ep {
                        ep = high_vec[i];
                        af = (af + accel_long).min(accel_max_long);
                    }
                    if i >= 2 {
                        sar = sar.min(low_vec[i-1]).min(low_vec[i-2]);
                    } else if i >= 1 {
                        sar = sar.min(low_vec[i-1]);
                    }
                }
            } else {
                if high_vec[i] >= sar {
                    is_bull = true;
                    sar = ep - offset_on_reverse;
                    ep = high_vec[i];
                    af = accel_init_long;
                } else {
                    if low_vec[i] < ep {
                        ep = low_vec[i];
                        af = (af + accel_short).min(accel_max_short);
                    }
                    if i >= 2 {
                        sar = sar.max(high_vec[i-1]).max(high_vec[i-2]);
                    } else if i >= 1 {
                        sar = sar.max(high_vec[i-1]);
                    }
                }
            }
            
            sar_values[i] = Some(sar);
        }
    }
    
    let result = Series::new(h.name().clone(), sar_values);
    Ok(PySeries(result))
}

// ====================================================================
// 动量指标 (Momentum Indicators) - 趋势强度和方向指标
// ====================================================================

/// 平均趋向指标 (ADX)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn adx(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![None; len];
    
    if len < period * 2 || period == 0 {
        return Ok(PySeries(Series::new(c.name().clone(), result)));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        // 预计算所有TR, +DM, -DM
        let mut tr = vec![0.0; len];
        let mut plus_dm = vec![0.0; len];
        let mut minus_dm = vec![0.0; len];
        
        for i in 1..len {
            let hl = h_arr[i] - l_arr[i];
            let hc = (h_arr[i] - c_arr[i - 1]).abs();
            let lc = (l_arr[i] - c_arr[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            let high_diff = h_arr[i] - h_arr[i - 1];
            let low_diff = l_arr[i - 1] - l_arr[i];
            
            if high_diff > low_diff && high_diff > 0.0 {
                plus_dm[i] = high_diff;
            }
            if low_diff > high_diff && low_diff > 0.0 {
                minus_dm[i] = low_diff;
            }
        }
        
        // 初始求和
        let mut smooth_tr: f64 = tr[1..=period].iter().sum();
        let mut smooth_plus_dm: f64 = plus_dm[1..=period].iter().sum();
        let mut smooth_minus_dm: f64 = minus_dm[1..=period].iter().sum();
        
        let inv_period = 1.0 / period as f64;
        
        // 预分配DX数组
        let mut dx = vec![0.0; len];
        
        // Wilder平滑 + 计算DI和DX
        for i in (period + 1)..len {
            smooth_tr = smooth_tr - smooth_tr * inv_period + tr[i];
            smooth_plus_dm = smooth_plus_dm - smooth_plus_dm * inv_period + plus_dm[i];
            smooth_minus_dm = smooth_minus_dm - smooth_minus_dm * inv_period + minus_dm[i];
            
            if smooth_tr > 0.0 {
                let plus_di = 100.0 * smooth_plus_dm / smooth_tr;
                let minus_di = 100.0 * smooth_minus_dm / smooth_tr;
                let sum_di = plus_di + minus_di;
                
                if sum_di > 0.0 {
                    dx[i] = 100.0 * (plus_di - minus_di).abs() / sum_di;
                }
            }
        }
        
        // ADX: DX的Wilder平滑
        let start_idx = period * 2;
        let mut adx: f64 = dx[(period + 1)..=start_idx].iter().sum::<f64>() / period as f64;
        result[start_idx] = Some(adx);
        
        for i in (start_idx + 1)..len {
            adx = adx - adx * inv_period + dx[i];
            result[i] = Some(adx);
        }
    } else {
        // Fallback
        let h_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let l_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let c_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut tr = vec![0.0; len];
        let mut plus_dm = vec![0.0; len];
        let mut minus_dm = vec![0.0; len];
        
        for i in 1..len {
            let hl = h_vec[i] - l_vec[i];
            let hc = (h_vec[i] - c_vec[i - 1]).abs();
            let lc = (l_vec[i] - c_vec[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            let high_diff = h_vec[i] - h_vec[i - 1];
            let low_diff = l_vec[i - 1] - l_vec[i];
            
            if high_diff > low_diff && high_diff > 0.0 {
                plus_dm[i] = high_diff;
            }
            if low_diff > high_diff && low_diff > 0.0 {
                minus_dm[i] = low_diff;
            }
        }
        
        let mut smooth_tr: f64 = tr[1..=period].iter().sum();
        let mut smooth_plus_dm: f64 = plus_dm[1..=period].iter().sum();
        let mut smooth_minus_dm: f64 = minus_dm[1..=period].iter().sum();
        
        let inv_period = 1.0 / period as f64;
        let mut dx = vec![0.0; len];
        
        for i in (period + 1)..len {
            smooth_tr = smooth_tr - smooth_tr * inv_period + tr[i];
            smooth_plus_dm = smooth_plus_dm - smooth_plus_dm * inv_period + plus_dm[i];
            smooth_minus_dm = smooth_minus_dm - smooth_minus_dm * inv_period + minus_dm[i];
            
            if smooth_tr > 0.0 {
                let plus_di = 100.0 * smooth_plus_dm / smooth_tr;
                let minus_di = 100.0 * smooth_minus_dm / smooth_tr;
                let sum_di = plus_di + minus_di;
                
                if sum_di > 0.0 {
                    dx[i] = 100.0 * (plus_di - minus_di).abs() / sum_di;
                }
            }
        }
        
        let start_idx = period * 2;
        let mut adx: f64 = dx[(period + 1)..=start_idx].iter().sum::<f64>() / period as f64;
        result[start_idx] = Some(adx);
        
        for i in (start_idx + 1)..len {
            adx = adx - adx * inv_period + dx[i];
            result[i] = Some(adx);
        }
    }
    
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

/// 平均趋向指标评级 (ADXR)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn adxr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    // 保存 name 后再 move (避免不必要的 clone)
    let name = c.name().clone();
    
    // 先计算ADX值
    let adx_result = adx(PySeries(h), PySeries(l), PySeries(c), period)?;
    let adx_series: Series = adx_result.into();
    let adx_vals = adx_series.f64().unwrap();
    let len = adx_vals.len();
    
    let mut adxr_values = vec![None; len];
    
    // 连续内存访问
    if let Ok(adx_arr) = adx_vals.cont_slice() {
        // ADXR = (当前ADX + period周期前的ADX) / 2
        for i in period..len {
            if i >= period * 3 { // 确保有足够的ADX数据
                let current_adx = adx_arr[i];
                let past_adx = adx_arr[i - period];
                if current_adx > 0.0 && past_adx > 0.0 {
                    adxr_values[i] = Some((current_adx + past_adx) * 0.5);
                }
            }
        }
    } else {
        // Fallback
        let adx_vec: Vec<f64> = adx_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        for i in period..len {
            if i >= period * 3 {
                let current_adx = adx_vec[i];
                let past_adx = adx_vec[i - period];
                if current_adx > 0.0 && past_adx > 0.0 {
                    adxr_values[i] = Some((current_adx + past_adx) * 0.5);
                }
            }
        }
    }
    
    let result = Series::new(name, adxr_values);
    Ok(PySeries(result))
}

/// 绝对价格摆动指标 (APO)
#[pyfunction]
#[pyo3(signature = (series, fast_period=12, slow_period=26))]
pub fn apo(series: PySeries, fast_period: usize, slow_period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let fast_ema_values = calculate_ma(&vec_values, fast_period, MAType::EMA);
    let slow_ema_values = calculate_ma(&vec_values, slow_period, MAType::EMA);
    
    let apo_values: Vec<Option<f64>> = fast_ema_values.iter().zip(slow_ema_values.iter())
        .map(|(&fast, &slow)| {
            match (fast, slow) {
                (Some(f), Some(s)) => Some(f - s),
                _ => None,
            }
        })
        .collect();
    
    let result = Series::new(s.name().clone(), apo_values);
    Ok(PySeries(result))
}

/// 阿隆指标 (AROON)
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn aroon(high: PySeries, low: PySeries, period: usize) -> PyResult<(PySeries, PySeries)> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    
    let len = high_vals.len();
    let mut up_out = vec![f64::NAN; len];
    let mut down_out = vec![f64::NAN; len];
    
    // TA-Lib style: Track indices of extremes, only rescan when necessary
    if let (Ok(high_input), Ok(low_input)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        // Initialize: find first window's extremes
        let mut highest_idx = 0;
        let mut lowest_idx = 0;
        let mut highest = high_input[0];
        let mut lowest = low_input[0];
        
        for i in 1..period {
            if high_input[i] >= highest {
                highest = high_input[i];
                highest_idx = i;
            }
            if low_input[i] <= lowest {
                lowest = low_input[i];
                lowest_idx = i;
            }
        }
        
        let factor = 100.0 / period as f64;
        
        // Slide window with index tracking
        for today in period..len {
            let trailing_idx = today - period;
            
            // Update if new extreme found
            if high_input[today] >= highest {
                highest = high_input[today];
                highest_idx = today;
            }
            if low_input[today] <= lowest {
                lowest = low_input[today];
                lowest_idx = today;
            }
            
            // Calculate Aroon based on distance from extreme
            up_out[today] = ((period - (today - highest_idx)) as f64) * factor;
            down_out[today] = ((period - (today - lowest_idx)) as f64) * factor;
            
            // Check if old extreme exited window - rescan if necessary
            if highest_idx <= trailing_idx {
                highest = high_input[trailing_idx + 1];
                highest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if high_input[i] >= highest {
                        highest = high_input[i];
                        highest_idx = i;
                    }
                }
            }
            
            if lowest_idx <= trailing_idx {
                lowest = low_input[trailing_idx + 1];
                lowest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if low_input[i] <= lowest {
                        lowest = low_input[i];
                        lowest_idx = i;
                    }
                }
            }
        }
    } else {
        // fallback
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut highest_idx = 0;
        let mut lowest_idx = 0;
        let mut highest = high_vec[0];
        let mut lowest = low_vec[0];
        
        for i in 1..period {
            if high_vec[i] >= highest {
                highest = high_vec[i];
                highest_idx = i;
            }
            if low_vec[i] <= lowest {
                lowest = low_vec[i];
                lowest_idx = i;
            }
        }
        
        let factor = 100.0 / period as f64;
        
        for today in period..len {
            let trailing_idx = today - period;
            
            if high_vec[today] >= highest {
                highest = high_vec[today];
                highest_idx = today;
            }
            if low_vec[today] <= lowest {
                lowest = low_vec[today];
                lowest_idx = today;
            }
            
            up_out[today] = ((period - (today - highest_idx)) as f64) * factor;
            down_out[today] = ((period - (today - lowest_idx)) as f64) * factor;
            
            if highest_idx <= trailing_idx {
                highest = high_vec[trailing_idx + 1];
                highest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if high_vec[i] >= highest {
                        highest = high_vec[i];
                        highest_idx = i;
                    }
                }
            }
            
            if lowest_idx <= trailing_idx {
                lowest = low_vec[trailing_idx + 1];
                lowest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if low_vec[i] <= lowest {
                        lowest = low_vec[i];
                        lowest_idx = i;
                    }
                }
            }
        }
    }
    
    let base_name = h.name();
    let up_series = Series::new(PlSmallStr::from_str(&format!("{}_aroon_up", base_name)), up_out);
    let down_series = Series::new(PlSmallStr::from_str(&format!("{}_aroon_down", base_name)), down_out);
    
    Ok((PySeries(up_series), PySeries(down_series)))
}

/// 阿隆摆动指标 (AROONOSC)
/// Simply reuse optimized AROON and subtract
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn aroonosc(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    
    let len = high_vals.len();
    let mut output = vec![f64::NAN; len];
    
    // Reuse AROON's optimized index tracking algorithm
    if let (Ok(high_input), Ok(low_input)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        let mut highest_idx = 0;
        let mut lowest_idx = 0;
        let mut highest = high_input[0];
        let mut lowest = low_input[0];
        
        for i in 1..period {
            if high_input[i] >= highest {
                highest = high_input[i];
                highest_idx = i;
            }
            if low_input[i] <= lowest {
                lowest = low_input[i];
                lowest_idx = i;
            }
        }
        
        let factor = 100.0 / period as f64;
        
        for today in period..len {
            let trailing_idx = today - period;
            
            if high_input[today] >= highest {
                highest = high_input[today];
                highest_idx = today;
            }
            if low_input[today] <= lowest {
                lowest = low_input[today];
                lowest_idx = today;
            }
            
            let aroon_up = ((period - (today - highest_idx)) as f64) * factor;
            let aroon_down = ((period - (today - lowest_idx)) as f64) * factor;
            output[today] = aroon_up - aroon_down;
            
            if highest_idx <= trailing_idx {
                highest = high_input[trailing_idx + 1];
                highest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if high_input[i] >= highest {
                        highest = high_input[i];
                        highest_idx = i;
                    }
                }
            }
            
            if lowest_idx <= trailing_idx {
                lowest = low_input[trailing_idx + 1];
                lowest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if low_input[i] <= lowest {
                        lowest = low_input[i];
                        lowest_idx = i;
                    }
                }
            }
        }
    } else {
        // fallback with optimized algorithm too
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut highest_idx = 0;
        let mut lowest_idx = 0;
        let mut highest = high_vec[0];
        let mut lowest = low_vec[0];
        
        for i in 1..period {
            if high_vec[i] >= highest {
                highest = high_vec[i];
                highest_idx = i;
            }
            if low_vec[i] <= lowest {
                lowest = low_vec[i];
                lowest_idx = i;
            }
        }
        
        let factor = 100.0 / period as f64;
        
        for today in period..len {
            let trailing_idx = today - period;
            
            if high_vec[today] >= highest {
                highest = high_vec[today];
                highest_idx = today;
            }
            if low_vec[today] <= lowest {
                lowest = low_vec[today];
                lowest_idx = today;
            }
            
            let aroon_up = ((period - (today - highest_idx)) as f64) * factor;
            let aroon_down = ((period - (today - lowest_idx)) as f64) * factor;
            output[today] = aroon_up - aroon_down;
            
            if highest_idx <= trailing_idx {
                highest = high_vec[trailing_idx + 1];
                highest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if high_vec[i] >= highest {
                        highest = high_vec[i];
                        highest_idx = i;
                    }
                }
            }
            
            if lowest_idx <= trailing_idx {
                lowest = low_vec[trailing_idx + 1];
                lowest_idx = trailing_idx + 1;
                for i in (trailing_idx + 2)..=today {
                    if low_vec[i] <= lowest {
                        lowest = low_vec[i];
                        lowest_idx = i;
                    }
                }
            }
        }
    }
    
    let result = Series::new(h.name().clone(), output);
    Ok(PySeries(result))
}

/// 正方向指标 (PLUS_DI)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn plus_di(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![None; len];
    
    if len < period + 1 {
        return Ok(PySeries(Series::new(c.name().clone(), result)));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        // 预计算所有 TR 和 +DM
        let mut tr = vec![0.0; len];
        let mut plus_dm = vec![0.0; len];
        
        for i in 1..len {
            // 内联 TR 计算
            let hl = h_arr[i] - l_arr[i];
            let hc = (h_arr[i] - c_arr[i - 1]).abs();
            let lc = (l_arr[i] - c_arr[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            // 内联 +DM 计算
            let high_diff = h_arr[i] - h_arr[i - 1];
            let low_diff = l_arr[i - 1] - l_arr[i];
            plus_dm[i] = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
        }
        
        // Wilder 平滑初始化
        let mut smooth_tr: f64 = (1..=period).map(|i| tr[i]).sum();
        let mut smooth_plus_dm: f64 = (1..=period).map(|i| plus_dm[i]).sum();
        
        result[period] = if smooth_tr > 0.0 { 
            Some(100.0 * smooth_plus_dm / smooth_tr) 
        } else { 
            Some(0.0) 
        };
        
        // 预计算 Wilder 平滑系数
        let inv_period = 1.0 / period as f64;
        
        for i in (period + 1)..len {
            // Wilder's 平滑: smooth = smooth - smooth/period + new_value
            smooth_tr = smooth_tr - (smooth_tr * inv_period) + tr[i];
            smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm * inv_period) + plus_dm[i];
            
            result[i] = if smooth_tr > 0.0 { 
                Some(100.0 * smooth_plus_dm / smooth_tr) 
            } else { 
                Some(0.0) 
            };
        }
    } else {
        // Fallback
        let high_vals: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut tr = vec![0.0; len];
        let mut plus_dm = vec![0.0; len];
        
        for i in 1..len {
            let hl = high_vals[i] - low_vals[i];
            let hc = (high_vals[i] - close_vals[i - 1]).abs();
            let lc = (low_vals[i] - close_vals[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            let high_diff = high_vals[i] - high_vals[i - 1];
            let low_diff = low_vals[i - 1] - low_vals[i];
            plus_dm[i] = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
        }
        
        let mut smooth_tr: f64 = (1..=period).map(|i| tr[i]).sum();
        let mut smooth_plus_dm: f64 = (1..=period).map(|i| plus_dm[i]).sum();
        
        result[period] = if smooth_tr > 0.0 { Some(100.0 * smooth_plus_dm / smooth_tr) } else { Some(0.0) };
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth_tr = smooth_tr - (smooth_tr * inv_period) + tr[i];
            smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm * inv_period) + plus_dm[i];
            result[i] = if smooth_tr > 0.0 { Some(100.0 * smooth_plus_dm / smooth_tr) } else { Some(0.0) };
        }
    }
    
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

/// 负方向指标 (MINUS_DI)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn minus_di(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![None; len];
    
    if len < period + 1 {
        return Ok(PySeries(Series::new(c.name().clone(), result)));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        // 预计算所有 TR 和 -DM
        let mut tr = vec![0.0; len];
        let mut minus_dm = vec![0.0; len];
        
        for i in 1..len {
            // 内联 TR 计算
            let hl = h_arr[i] - l_arr[i];
            let hc = (h_arr[i] - c_arr[i - 1]).abs();
            let lc = (l_arr[i] - c_arr[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            // 内联 -DM 计算
            let high_diff = h_arr[i] - h_arr[i - 1];
            let low_diff = l_arr[i - 1] - l_arr[i];
            minus_dm[i] = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
        }
        
        // Wilder 平滑初始化
        let mut smooth_tr: f64 = (1..=period).map(|i| tr[i]).sum();
        let mut smooth_minus_dm: f64 = (1..=period).map(|i| minus_dm[i]).sum();
        
        result[period] = if smooth_tr > 0.0 { 
            Some(100.0 * smooth_minus_dm / smooth_tr) 
        } else { 
            Some(0.0) 
        };
        
        // 预计算 Wilder 平滑系数
        let inv_period = 1.0 / period as f64;
        
        for i in (period + 1)..len {
            // Wilder's 平滑
            smooth_tr = smooth_tr - (smooth_tr * inv_period) + tr[i];
            smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm * inv_period) + minus_dm[i];
            
            result[i] = if smooth_tr > 0.0 { 
                Some(100.0 * smooth_minus_dm / smooth_tr) 
            } else { 
                Some(0.0) 
            };
        }
    } else {
        // Fallback
        let high_vals: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut tr = vec![0.0; len];
        let mut minus_dm = vec![0.0; len];
        
        for i in 1..len {
            let hl = high_vals[i] - low_vals[i];
            let hc = (high_vals[i] - close_vals[i - 1]).abs();
            let lc = (low_vals[i] - close_vals[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            let high_diff = high_vals[i] - high_vals[i - 1];
            let low_diff = low_vals[i - 1] - low_vals[i];
            minus_dm[i] = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
        }
        
        let mut smooth_tr: f64 = (1..=period).map(|i| tr[i]).sum();
        let mut smooth_minus_dm: f64 = (1..=period).map(|i| minus_dm[i]).sum();
        
        result[period] = if smooth_tr > 0.0 { Some(100.0 * smooth_minus_dm / smooth_tr) } else { Some(0.0) };
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth_tr = smooth_tr - (smooth_tr * inv_period) + tr[i];
            smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm * inv_period) + minus_dm[i];
            result[i] = if smooth_tr > 0.0 { Some(100.0 * smooth_minus_dm / smooth_tr) } else { Some(0.0) };
        }
    }
    
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

/// 正方向移动 (PLUS_DM)
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn plus_dm(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    
    let len = high_vals.len();
    let mut result = vec![None; len];
    
    if len < period + 1 {
        let result_series = Series::new(h.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // Zero-copy optimization
    if let (Ok(h_arr), Ok(l_arr)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        // Pre-compute all +DM values
        let mut plus_dm = vec![0.0; len];
        for i in 1..len {
            let high_diff = h_arr[i] - h_arr[i - 1];
            let low_diff = l_arr[i - 1] - l_arr[i];
            plus_dm[i] = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
        }
        
        // Wilder smoothing
        let mut smooth: f64 = (1..=period).map(|i| plus_dm[i]).sum();
        result[period] = Some(smooth);
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth = smooth - (smooth * inv_period) + plus_dm[i];
            result[i] = Some(smooth);
        }
    } else {
        // Fallback
        let h_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let l_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut plus_dm = vec![0.0; len];
        for i in 1..len {
            let high_diff = h_vec[i] - h_vec[i - 1];
            let low_diff = l_vec[i - 1] - l_vec[i];
            plus_dm[i] = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
        }
        
        let mut smooth: f64 = (1..=period).map(|i| plus_dm[i]).sum();
        result[period] = Some(smooth);
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth = smooth - (smooth * inv_period) + plus_dm[i];
            result[i] = Some(smooth);
        }
    }
    
    let result_series = Series::new(h.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 负方向移动 (MINUS_DM)
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn minus_dm(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    
    let len = high_vals.len();
    let mut result = vec![None; len];
    
    if len < period + 1 {
        let result_series = Series::new(l.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // Zero-copy optimization
    if let (Ok(h_arr), Ok(l_arr)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        // Pre-compute all -DM values
        let mut minus_dm = vec![0.0; len];
        for i in 1..len {
            let high_diff = h_arr[i] - h_arr[i - 1];
            let low_diff = l_arr[i - 1] - l_arr[i];
            minus_dm[i] = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
        }
        
        // Wilder smoothing
        let mut smooth: f64 = (1..=period).map(|i| minus_dm[i]).sum();
        result[period] = Some(smooth);
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth = smooth - (smooth * inv_period) + minus_dm[i];
            result[i] = Some(smooth);
        }
    } else {
        // Fallback
        let h_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let l_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut minus_dm = vec![0.0; len];
        for i in 1..len {
            let high_diff = h_vec[i] - h_vec[i - 1];
            let low_diff = l_vec[i - 1] - l_vec[i];
            minus_dm[i] = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
        }
        
        // Wilder smoothing
        let mut smooth: f64 = (1..=period).map(|i| minus_dm[i]).sum();
        result[period] = Some(smooth);
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth = smooth - (smooth * inv_period) + minus_dm[i];
            result[i] = Some(smooth);
        }
    }
    
    let result_series = Series::new(l.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 均势指标 (BOP)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn bop(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals = o.f64().unwrap();
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![0.0; len];
    
    // 连续内存访问
    if let (Ok(o_arr), Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (open_vals.cont_slice(), high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        for i in 0..len {
            let range = h_arr[i] - l_arr[i];
            result[i] = if range > 0.0 {
                (c_arr[i] - o_arr[i]) / range
            } else {
                0.0
            };
        }
    } else {
        // 后备路径
        for i in 0..len {
            let o = open_vals.get(i).unwrap_or(0.0);
            let h = high_vals.get(i).unwrap_or(0.0);
            let l = low_vals.get(i).unwrap_or(0.0);
            let c = close_vals.get(i).unwrap_or(0.0);
            
            let range = h - l;
            result[i] = if range > 0.0 {
                (c - o) / range
            } else {
                0.0
            };
        }
    }
    
    let result_series = Series::new(c.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 商品通道指数 (CCI)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn cci(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let c_name = c.name().clone();
    
    // 向量化计算典型价格: TP = (H + L + C) / 3
    let h_plus_l = (&h + &l)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("h+l failed: {}", e)))?;
    let tp = (h_plus_l + c)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("tp failed: {}", e)))?
        .cast(&DataType::Float64)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cast failed: {}", e)))?
        / 3.0;
    
    // 使用 rolling_mean 计算 SMA(TP)
    let rolling_opts = RollingOptionsFixedWindow {
        window_size: period,
        min_periods: period,
        center: false,
        ..Default::default()
    };
    
    let sma_tp = tp.rolling_mean(rolling_opts.clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("rolling_mean failed: {}", e)))?;
    
    // 计算平均绝对偏差：需要手动循环（无原生rolling MAD）
    let tp_vals: Vec<f64> = tp.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let sma_vals: Vec<f64> = sma_tp.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let cci_values = {
        let mut result = vec![None; tp_vals.len()];
        
        for i in period - 1..tp_vals.len() {
            let sma = sma_vals[i];
            if sma == 0.0 {
                continue;
            }
            
            // 计算窗口内的平均绝对偏差
            let mut mad_sum = 0.0;
            for j in (i + 1 - period)..=i {
                mad_sum += (tp_vals[j] - sma).abs();
            }
            let mean_deviation = mad_sum / period as f64;
            
            if mean_deviation > 0.0 {
                let cci = (tp_vals[i] - sma) / (0.015 * mean_deviation);
                result[i] = Some(cci);
            }
        }
        
        result
    };
    
    let result = Series::new(c_name, cci_values);
    Ok(PySeries(result))
}

/// 钱德动量摆动指标 (CMO)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn cmo(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        let result_series = Series::new(s.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // 连续内存访问
    if let Ok(arr) = values.cont_slice() {
        // 滑动窗口计算
        let mut up_sum = 0.0;
        let mut down_sum = 0.0;
        
        // 初始化第一个窗口
        for i in 1..=period {
            let change = arr[i] - arr[i - 1];
            if change > 0.0 {
                up_sum += change;
            } else if change < 0.0 {
                down_sum += -change;
            }
        }
        
        let total = up_sum + down_sum;
        if total > 0.0 {
            result[period] = 100.0 * (up_sum - down_sum) / total;
        }
        
        // 滑动窗口
        for i in (period + 1)..len {
            let old_change = arr[i - period] - arr[i - period - 1];
            let new_change = arr[i] - arr[i - 1];
            
            // 移除窗口左侧的值
            if old_change > 0.0 {
                up_sum -= old_change;
            } else if old_change < 0.0 {
                down_sum -= -old_change;
            }
            
            // 添加窗口右侧的值
            if new_change > 0.0 {
                up_sum += new_change;
            } else if new_change < 0.0 {
                down_sum += -new_change;
            }
            
            let total = up_sum + down_sum;
            if total > 0.0 {
                result[i] = 100.0 * (up_sum - down_sum) / total;
            }
        }
    } else {
        // 后备路径
        for i in period..len {
            let mut up_sum = 0.0;
            let mut down_sum = 0.0;
            
            for j in (i + 1 - period)..=i {
                if j > 0 {
                    let curr = values.get(j).unwrap_or(0.0);
                    let prev = values.get(j - 1).unwrap_or(0.0);
                    let change = curr - prev;
                    if change > 0.0 {
                        up_sum += change;
                    } else if change < 0.0 {
                        down_sum += -change;
                    }
                }
            }
            
            let total = up_sum + down_sum;
            if total > 0.0 {
                result[i] = 100.0 * (up_sum - down_sum) / total;
            }
        }
    }
    
    let result_series = Series::new(s.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 方向性指标 (DX)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn dx(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![None; len];
    
    if len < period + 1 {
        return Ok(PySeries(Series::new(c.name().clone(), result)));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        // 预计算所有 TR, +DM, -DM
        let mut tr = vec![0.0; len];
        let mut plus_dm = vec![0.0; len];
        let mut minus_dm = vec![0.0; len];
        
        for i in 1..len {
            // 内联 TR 计算
            let hl = h_arr[i] - l_arr[i];
            let hc = (h_arr[i] - c_arr[i - 1]).abs();
            let lc = (l_arr[i] - c_arr[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            // 内联 DM 计算
            let up_move = h_arr[i] - h_arr[i - 1];
            let down_move = l_arr[i - 1] - l_arr[i];
            plus_dm[i] = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            minus_dm[i] = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        }
        
        // Wilder 平滑初始化
        let mut smooth_tr: f64 = (1..=period).map(|i| tr[i]).sum();
        let mut smooth_plus_dm: f64 = (1..=period).map(|i| plus_dm[i]).sum();
        let mut smooth_minus_dm: f64 = (1..=period).map(|i| minus_dm[i]).sum();
        
        // 计算第一个 DX 值
        if smooth_tr > 0.0 {
            let plus_di = 100.0 * smooth_plus_dm / smooth_tr;
            let minus_di = 100.0 * smooth_minus_dm / smooth_tr;
            let di_sum = plus_di + minus_di;
            result[period] = if di_sum > 0.0 {
                Some(100.0 * (plus_di - minus_di).abs() / di_sum)
            } else {
                Some(0.0)
            };
        }
        
        // 预计算 Wilder 平滑系数
        let inv_period = 1.0 / period as f64;
        
        for i in (period + 1)..len {
            // Wilder's 平滑
            smooth_tr = smooth_tr - (smooth_tr * inv_period) + tr[i];
            smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm * inv_period) + plus_dm[i];
            smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm * inv_period) + minus_dm[i];
            
            if smooth_tr > 0.0 {
                let plus_di = 100.0 * smooth_plus_dm / smooth_tr;
                let minus_di = 100.0 * smooth_minus_dm / smooth_tr;
                let di_sum = plus_di + minus_di;
                result[i] = if di_sum > 0.0 {
                    Some(100.0 * (plus_di - minus_di).abs() / di_sum)
                } else {
                    Some(0.0)
                };
            }
        }
    } else {
        // Fallback
        let high_vals: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut tr = vec![0.0; len];
        let mut plus_dm = vec![0.0; len];
        let mut minus_dm = vec![0.0; len];
        
        for i in 1..len {
            let hl = high_vals[i] - low_vals[i];
            let hc = (high_vals[i] - close_vals[i - 1]).abs();
            let lc = (low_vals[i] - close_vals[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
            
            let up_move = high_vals[i] - high_vals[i - 1];
            let down_move = low_vals[i - 1] - low_vals[i];
            plus_dm[i] = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            minus_dm[i] = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        }
        
        let mut smooth_tr: f64 = (1..=period).map(|i| tr[i]).sum();
        let mut smooth_plus_dm: f64 = (1..=period).map(|i| plus_dm[i]).sum();
        let mut smooth_minus_dm: f64 = (1..=period).map(|i| minus_dm[i]).sum();
        
        if smooth_tr > 0.0 {
            let plus_di = 100.0 * smooth_plus_dm / smooth_tr;
            let minus_di = 100.0 * smooth_minus_dm / smooth_tr;
            let di_sum = plus_di + minus_di;
            result[period] = if di_sum > 0.0 { Some(100.0 * (plus_di - minus_di).abs() / di_sum) } else { Some(0.0) };
        }
        
        let inv_period = 1.0 / period as f64;
        for i in (period + 1)..len {
            smooth_tr = smooth_tr - (smooth_tr * inv_period) + tr[i];
            smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm * inv_period) + plus_dm[i];
            smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm * inv_period) + minus_dm[i];
            
            if smooth_tr > 0.0 {
                let plus_di = 100.0 * smooth_plus_dm / smooth_tr;
                let minus_di = 100.0 * smooth_minus_dm / smooth_tr;
                let di_sum = plus_di + minus_di;
                result[i] = if di_sum > 0.0 { Some(100.0 * (plus_di - minus_di).abs() / di_sum) } else { Some(0.0) };
            }
        }
    }
    
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

/// MACD (移动平均收敛发散指标)
#[pyfunction]
#[pyo3(signature = (series, fast=12, slow=26, signal=9))]
pub fn macd(series: PySeries, fast: usize, slow: usize, signal: usize) -> PyResult<(PySeries, PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect();
    
    // 计算快线EMA
    let fast_ema_values = calculate_ma(&vec_values, fast, MAType::EMA);
    let fast_ema: Vec<f64> = fast_ema_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算慢线EMA
    let slow_ema_values = calculate_ma(&vec_values, slow, MAType::EMA);
    let slow_ema: Vec<f64> = slow_ema_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算DIF (MACD线)
    let dif: Vec<f64> = fast_ema.iter().zip(slow_ema.iter())
        .map(|(&fast, &slow)| fast - slow)
        .collect();
    
    // 计算DEA (信号线)
    let dea_values = calculate_ma(&dif, signal, MAType::EMA);
    let dea: Vec<f64> = dea_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算MACD柱状图
    let macd_hist: Vec<f64> = dif.iter().zip(dea.iter())
        .map(|(&dif_val, &dea_val)| 2.0 * (dif_val - dea_val))
        .collect();
    
    let base_name = s.name();
    let dif_series = Series::new(PlSmallStr::from_str(&format!("{}_macd_dif", base_name)), dif);
    let dea_series = Series::new(PlSmallStr::from_str(&format!("{}_macd_dea", base_name)), dea);
    let macd_series = Series::new(PlSmallStr::from_str(&format!("{}_macd_hist", base_name)), macd_hist);
    
    Ok((PySeries(dif_series), PySeries(dea_series), PySeries(macd_series)))
}

/// MACD扩展 (MACDEXT) - 支持不同类型的移动平均
#[pyfunction]
#[pyo3(signature = (series, fast=12, slow=26, signal=9, fast_matype="EMA", slow_matype="EMA", signal_matype="EMA"))]
pub fn macdext(series: PySeries, fast: usize, slow: usize, signal: usize, fast_matype: &str, slow_matype: &str, signal_matype: &str) -> PyResult<(PySeries, PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    // 解析移动平均类型
    let fast_ma_type = match fast_matype {
        "SMA" => MAType::SMA,
        "EMA" | _ => MAType::EMA,
    };
    let slow_ma_type = match slow_matype {
        "SMA" => MAType::SMA, 
        "EMA" | _ => MAType::EMA,
    };
    let signal_ma_type = match signal_matype {
        "SMA" => MAType::SMA,
        "EMA" | _ => MAType::EMA,
    };
    
    // 计算快线MA
    let fast_ma_values = calculate_ma(&vec_values, fast, fast_ma_type);
    let fast_ma: Vec<f64> = fast_ma_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算慢线MA
    let slow_ma_values = calculate_ma(&vec_values, slow, slow_ma_type);
    let slow_ma: Vec<f64> = slow_ma_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算DIF (MACD线)
    let dif: Vec<f64> = fast_ma.iter().zip(slow_ma.iter())
        .map(|(&fast, &slow)| fast - slow)
        .collect();
    
    // 计算DEA (信号线)
    let dea_values = calculate_ma(&dif, signal, signal_ma_type);
    let dea: Vec<f64> = dea_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算MACD柱状图
    let macd_hist: Vec<f64> = dif.iter().zip(dea.iter())
        .map(|(&dif_val, &dea_val)| 2.0 * (dif_val - dea_val))
        .collect();
    
    let base_name = s.name();
    let dif_series = Series::new(PlSmallStr::from_str(&format!("{}_macdext_dif", base_name)), dif);
    let dea_series = Series::new(PlSmallStr::from_str(&format!("{}_macdext_dea", base_name)), dea);
    let macd_series = Series::new(PlSmallStr::from_str(&format!("{}_macdext_hist", base_name)), macd_hist);
    
    Ok((PySeries(dif_series), PySeries(dea_series), PySeries(macd_series)))
}

/// MACD固定 (MACDFIX) - 固定快线周期为12
#[pyfunction]
#[pyo3(signature = (series, signal=9))]
pub fn macdfix(series: PySeries, signal: usize) -> PyResult<(PySeries, PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    // 固定参数：快线12，慢线26
    let fast = 12;
    let slow = 26;
    
    // 计算快线EMA
    let fast_ema_values = calculate_ma(&vec_values, fast, MAType::EMA);
    let fast_ema: Vec<f64> = fast_ema_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算慢线EMA
    let slow_ema_values = calculate_ma(&vec_values, slow, MAType::EMA);
    let slow_ema: Vec<f64> = slow_ema_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算DIF (MACD线)
    let dif: Vec<f64> = fast_ema.iter().zip(slow_ema.iter())
        .map(|(&fast, &slow)| fast - slow)
        .collect();
    
    // 计算DEA (信号线)
    let dea_values = calculate_ma(&dif, signal, MAType::EMA);
    let dea: Vec<f64> = dea_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算MACD柱状图
    let macd_hist: Vec<f64> = dif.iter().zip(dea.iter())
        .map(|(&dif_val, &dea_val)| 2.0 * (dif_val - dea_val))
        .collect();
    
    let base_name = s.name();
    let dif_series = Series::new(PlSmallStr::from_str(&format!("{}_macdfix_dif", base_name)), dif);
    let dea_series = Series::new(PlSmallStr::from_str(&format!("{}_macdfix_dea", base_name)), dea);
    let macd_series = Series::new(PlSmallStr::from_str(&format!("{}_macdfix_hist", base_name)), macd_hist);
    
    Ok((PySeries(dif_series), PySeries(dea_series), PySeries(macd_series)))
}

/// 资金流量指标 (MFI)
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, period=14))]
pub fn mfi(high: PySeries, low: PySeries, close: PySeries, volume: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    let v: Series = volume.into();
    
    // 将 volume 转换为 f64 类型
    let v_f64 = if v.dtype() == &DataType::Int64 {
        v.cast(&DataType::Float64).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to cast volume to Float64: {}", e)))?
    } else {
        v
    };
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    let volume_vals = v_f64.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        let result_series = Series::new(c.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr), Ok(v_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice(), volume_vals.cont_slice()) {
        
        // 预计算所有典型价格和资金流
        let mut tp = vec![0.0; len];
        let mut money_flow = vec![0.0; len];
        let mut flow_direction = vec![0i8; len]; // 1: positive, -1: negative, 0: no change
        
        let one_third = 1.0 / 3.0;
        for i in 0..len {
            tp[i] = (h_arr[i] + l_arr[i] + c_arr[i]) * one_third;
            money_flow[i] = tp[i] * v_arr[i];
            
            if i > 0 {
                if tp[i] > tp[i - 1] {
                    flow_direction[i] = 1;
                } else if tp[i] < tp[i - 1] {
                    flow_direction[i] = -1;
                }
            }
        }
        
        // 初始化第一个窗口
        let mut positive_flow = 0.0;
        let mut negative_flow = 0.0;
        
        for i in 1..=period {
            if flow_direction[i] == 1 {
                positive_flow += money_flow[i];
            } else if flow_direction[i] == -1 {
                negative_flow += money_flow[i];
            }
        }
        
        // 计算第一个MFI值
        if negative_flow > 0.0 {
            let mfi_ratio = positive_flow / negative_flow;
            result[period] = 100.0 - (100.0 / (1.0 + mfi_ratio));
        } else if positive_flow > 0.0 {
            result[period] = 100.0;
        }
        
        // 滑动窗口计算后续MFI值
        for i in (period + 1)..len {
            // 移除最老的bar
            let old_idx = i - period;
            if flow_direction[old_idx] == 1 {
                positive_flow -= money_flow[old_idx];
            } else if flow_direction[old_idx] == -1 {
                negative_flow -= money_flow[old_idx];
            }
            
            // 添加最新的bar
            if flow_direction[i] == 1 {
                positive_flow += money_flow[i];
            } else if flow_direction[i] == -1 {
                negative_flow += money_flow[i];
            }
            
            // 计算MFI
            if negative_flow > 0.0 {
                let mfi_ratio = positive_flow / negative_flow;
                result[i] = 100.0 - (100.0 / (1.0 + mfi_ratio));
            } else if positive_flow > 0.0 {
                result[i] = 100.0;
            }
        }
    } else {
        // 后备路径
        for i in period..len {
            let mut positive_flow = 0.0;
            let mut negative_flow = 0.0;
            
            for j in (i + 1 - period)..=i {
                if j > 0 {
                    let h = high_vals.get(j).unwrap_or(0.0);
                    let l = low_vals.get(j).unwrap_or(0.0);
                    let c = close_vals.get(j).unwrap_or(0.0);
                    let v = volume_vals.get(j).unwrap_or(0.0);
                    
                    let h_prev = high_vals.get(j - 1).unwrap_or(0.0);
                    let l_prev = low_vals.get(j - 1).unwrap_or(0.0);
                    let c_prev = close_vals.get(j - 1).unwrap_or(0.0);
                    
                    let tp = (h + l + c) / 3.0;
                    let prev_tp = (h_prev + l_prev + c_prev) / 3.0;
                    let money_flow = tp * v;
                    
                    if tp > prev_tp {
                        positive_flow += money_flow;
                    } else if tp < prev_tp {
                        negative_flow += money_flow;
                    }
                }
            }
            
            if negative_flow > 0.0 {
                let mfi_ratio = positive_flow / negative_flow;
                result[i] = 100.0 - (100.0 / (1.0 + mfi_ratio));
            } else if positive_flow > 0.0 {
                result[i] = 100.0;
            }
        }
    }
    
    let result_series = Series::new(c.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 动量指标 (MOM) - 使用 Polars 原生操作优化
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn mom(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert series to f64: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    // 零拷贝访问 + tight loop
    if let Ok(input) = values.cont_slice() {
        for i in period..len {
            output[i] = input[i] - input[i - period];
        }
    } else {
        // fallback: 标准访问
        for i in period..len {
            if let (Some(curr), Some(prev)) = (values.get(i), values.get(i - period)) {
                output[i] = curr - prev;
            }
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 价格摆动指标百分比 (PPO)
#[pyfunction]
#[pyo3(signature = (series, fast_period=12, slow_period=26))]
pub fn ppo(series: PySeries, fast_period: usize, slow_period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let fast_ema_values = calculate_ma(&vec_values, fast_period, MAType::EMA);
    let slow_ema_values = calculate_ma(&vec_values, slow_period, MAType::EMA);
    
    let ppo_values: Vec<Option<f64>> = fast_ema_values.iter().zip(slow_ema_values.iter())
        .map(|(&fast, &slow)| {
            match (fast, slow) {
                (Some(f), Some(s)) if s != 0.0 => Some(((f - s) / s) * 100.0),
                _ => None,
            }
        })
        .collect();
    
    let result = Series::new(s.name().clone(), ppo_values);
    Ok(PySeries(result))
}

/// 变化率指标 (ROC)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn roc(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop
    if let Ok(input) = values.cont_slice() {
        for i in period..len {
            let prev = input[i - period];
            if prev != 0.0 {
                output[i] = ((input[i] - prev) / prev) * 100.0;
            }
        }
    } else {
        for i in period..len {
            if let (Some(curr), Some(prev)) = (values.get(i), values.get(i - period)) {
                if prev != 0.0 {
                    output[i] = ((curr - prev) / prev) * 100.0;
                }
            }
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 变化率百分比 (ROCP)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn rocp(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop
    if let Ok(input) = values.cont_slice() {
        for i in period..len {
            let prev = input[i - period];
            if prev != 0.0 {
                output[i] = (input[i] - prev) / prev;
            }
        }
    } else {
        for i in period..len {
            if let (Some(curr), Some(prev)) = (values.get(i), values.get(i - period)) {
                if prev != 0.0 {
                    output[i] = (curr - prev) / prev;
                }
            }
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 变化率比率 (ROCR)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn rocr(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop
    if let Ok(input) = values.cont_slice() {
        for i in period..len {
            let prev = input[i - period];
            if prev != 0.0 {
                output[i] = input[i] / prev;
            }
        }
    } else {
        for i in period..len {
            if let (Some(curr), Some(prev)) = (values.get(i), values.get(i - period)) {
                if prev != 0.0 {
                    output[i] = curr / prev;
                }
            }
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 变化率比率100 (ROCR100)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn rocr100(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let len = values.len();
    let mut output = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop
    if let Ok(input) = values.cont_slice() {
        for i in period..len {
            let prev = input[i - period];
            if prev != 0.0 {
                output[i] = (input[i] / prev) * 100.0;
            }
        }
    } else {
        for i in period..len {
            if let (Some(curr), Some(prev)) = (values.get(i), values.get(i - period)) {
                if prev != 0.0 {
                    output[i] = (curr / prev) * 100.0;
                }
            }
        }
    }
    
    let result = Series::new(s.name().clone(), output);
    Ok(PySeries(result))
}

/// 相对强弱指标 (RSI)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn rsi(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let rsi_values = calculate_rsi(&vec_values, period);
    let result = Series::new(s.name().clone(), rsi_values);
    
    Ok(PySeries(result))
}

/// 随机指标 (STOCH)
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period=14, d_period=3))]
pub fn stoch(high: PySeries, low: PySeries, close: PySeries, k_period: usize, d_period: usize) -> PyResult<(PySeries, PySeries)> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut k_values = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop 计算%K
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        for i in k_period - 1..len {
            let start = i + 1 - k_period;
            
            let mut highest = h_arr[start];
            let mut lowest = l_arr[start];
            for j in start + 1..=i {
                if h_arr[j] > highest { highest = h_arr[j]; }
                if l_arr[j] < lowest { lowest = l_arr[j]; }
            }
            
            let range = highest - lowest;
            if range != 0.0 {
                k_values[i] = ((c_arr[i] - lowest) / range) * 100.0;
            }
        }
    } else {
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        for i in k_period - 1..len {
            let start = i + 1 - k_period;
            let highest = high_vec[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = low_vec[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            if highest != lowest {
                k_values[i] = ((close_vec[i] - lowest) / (highest - lowest)) * 100.0;
            }
        }
    }
    
    // 计算%D (K的移动平均)
    let d_values = calculate_ma(&k_values, d_period, MAType::SMA);
    let d_vec: Vec<f64> = d_values.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
    
    let base_name = h.name();
    let k_series = Series::new(PlSmallStr::from_str(&format!("{}_stoch_k", base_name)), k_values);
    let d_series = Series::new(PlSmallStr::from_str(&format!("{}_stoch_d", base_name)), d_vec);
    
    Ok((PySeries(k_series), PySeries(d_series)))
}

/// 快速随机指标 (STOCHF)
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period=14, d_period=3))]
pub fn stochf(high: PySeries, low: PySeries, close: PySeries, k_period: usize, d_period: usize) -> PyResult<(PySeries, PySeries)> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut fastk_values = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop 计算FastK
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        for i in k_period - 1..len {
            let start = i + 1 - k_period;
            
            let mut highest = h_arr[start];
            let mut lowest = l_arr[start];
            for j in start + 1..=i {
                if h_arr[j] > highest { highest = h_arr[j]; }
                if l_arr[j] < lowest { lowest = l_arr[j]; }
            }
            
            let range = highest - lowest;
            if range != 0.0 {
                fastk_values[i] = ((c_arr[i] - lowest) / range) * 100.0;
            }
        }
    } else {
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        for i in k_period - 1..len {
            let start = i + 1 - k_period;
            let highest = high_vec[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = low_vec[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            if highest != lowest {
                fastk_values[i] = ((close_vec[i] - lowest) / (highest - lowest)) * 100.0;
            }
        }
    }
    
    // 计算FastD (FastK的移动平均)
    let fastd_values = calculate_ma(&fastk_values, d_period, MAType::SMA);
    let fastd_vec: Vec<f64> = fastd_values.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
    
    let base_name = h.name();
    let fastk_series = Series::new(PlSmallStr::from_str(&format!("{}_stochf_k", base_name)), fastk_values);
    let fastd_series = Series::new(PlSmallStr::from_str(&format!("{}_stochf_d", base_name)), fastd_vec);
    
    Ok((PySeries(fastk_series), PySeries(fastd_series)))
}

/// RSI随机指标 (STOCHRSI)
#[pyfunction]
#[pyo3(signature = (series, period=14, k_period=5, d_period=3))]
pub fn stochrsi(series: PySeries, period: usize, k_period: usize, d_period: usize) -> PyResult<(PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    // 先计算RSI
    let mut rsi_values = vec![None; vec_values.len()];
    
    if vec_values.len() > period {
        let mut up_sum = 0.0;
        let mut down_sum = 0.0;
        
        for i in 1..=period {
            let change = vec_values[i] - vec_values[i - 1];
            if change > 0.0 {
                up_sum += change;
            } else {
                down_sum += -change;
            }
        }
        
        let mut avg_gain = up_sum / period as f64;
        let mut avg_loss = down_sum / period as f64;
        
        if avg_loss > 0.0 {
            rsi_values[period] = Some(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)));
        } else {
            rsi_values[period] = Some(100.0);
        }
        
        for i in (period + 1)..vec_values.len() {
            let change = vec_values[i] - vec_values[i - 1];
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };
            
            avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;
            
            if avg_loss > 0.0 {
                rsi_values[i] = Some(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)));
            } else {
                rsi_values[i] = Some(100.0);
            }
        }
    }
    
    // 计算RSI的StochRSI
    let rsi_vals: Vec<f64> = rsi_values.iter().map(|&x| x.unwrap_or(50.0)).collect();
    
    let mut stochrsi_values = vec![None; rsi_vals.len()];
    
    for i in k_period - 1..rsi_vals.len() {
        if i >= period {
            let slice_start = i + 1 - k_period;
            let rsi_slice = &rsi_vals[slice_start..=i];
            
            let highest = rsi_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = rsi_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            if highest != lowest {
                stochrsi_values[i] = Some(((rsi_vals[i] - lowest) / (highest - lowest)) * 100.0);
            }
        }
    }
    
    // 计算StochRSI的移动平均作为D值
    let stochrsi_vals_for_d: Vec<f64> = stochrsi_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
    let stochrsi_d_values = calculate_ma(&stochrsi_vals_for_d, d_period, MAType::SMA);
    
    let base_name = s.name();
    let k_series = Series::new(PlSmallStr::from_str(&format!("{}_stochrsi_k", base_name)), stochrsi_values);
    let d_series = Series::new(PlSmallStr::from_str(&format!("{}_stochrsi_d", base_name)), stochrsi_d_values);
    
    Ok((PySeries(k_series), PySeries(d_series)))
}

/// 三重指数平均线 (TRIX)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn trix(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    // 第一次EMA
    let ema1_values = calculate_ma(&vec_values, period, MAType::EMA);
    let ema1: Vec<f64> = ema1_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 第二次EMA
    let ema2_values = calculate_ma(&ema1, period, MAType::EMA);
    let ema2: Vec<f64> = ema2_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 第三次EMA
    let ema3_values = calculate_ma(&ema2, period, MAType::EMA);
    let ema3: Vec<f64> = ema3_values.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算TRIX (EMA3的变化率)
    let trix_values = {
        let mut result = vec![None; ema3.len()];
        
        for i in 1..ema3.len() {
            if ema3[i - 1] != 0.0 {
                result[i] = Some(((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 10000.0);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), trix_values);
    Ok(PySeries(result))
}

/// 终极摆动指标 (ULTOSC)
#[pyfunction]
#[pyo3(signature = (high, low, close, period1=7, period2=14, period3=28))]
pub fn ultosc(high: PySeries, low: PySeries, close: PySeries, period1: usize, period2: usize, period3: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("high must be numeric: {}", e)))?;
    let low_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("low must be numeric: {}", e)))?;
    let close_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("close must be numeric: {}", e)))?;
    
    let len = close_vals.len();
    let mut result = vec![None; len];
    
    // Sort periods: longest to shortest (TA-Lib style)
    let max_period = period1.max(period2).max(period3);
    
    if len <= max_period {
        let result_series = Series::new(c.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // Zero-copy optimization
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        // Pre-compute BP and TR for all bars (TA-Lib CALC_TERMS macro pattern)
        let mut bp = vec![0.0; len];
        let mut tr = vec![0.0; len];
        
        for i in 1..len {
            let true_low = l_arr[i].min(c_arr[i - 1]);
            bp[i] = c_arr[i] - true_low;
            
            // True Range: max(H-L, |H-prev_close|, |L-prev_close|)
            let mut true_range = h_arr[i] - l_arr[i];
            let temp = (c_arr[i - 1] - h_arr[i]).abs();
            if temp > true_range { true_range = temp; }
            let temp = (c_arr[i - 1] - l_arr[i]).abs();
            if temp > true_range { true_range = temp; }
            
            tr[i] = true_range;
        }
        
        // Initialize running totals for the longest period (TA-Lib PRIME_TOTALS pattern)
        let mut bp1_total = 0.0;
        let mut tr1_total = 0.0;
        let mut bp2_total = 0.0;
        let mut tr2_total = 0.0;
        let mut bp3_total = 0.0;
        let mut tr3_total = 0.0;
        
        let start_idx = max_period;
        
        // Prime the first windows
        for i in (start_idx - period1 + 1)..start_idx {
            bp1_total += bp[i];
            tr1_total += tr[i];
        }
        for i in (start_idx - period2 + 1)..start_idx {
            bp2_total += bp[i];
            tr2_total += tr[i];
        }
        for i in (start_idx - period3 + 1)..start_idx {
            bp3_total += bp[i];
            tr3_total += tr[i];
        }
        
        // Sliding window calculation (TA-Lib main loop pattern)
        let mut trailing_idx1 = start_idx - period1 + 1;
        let mut trailing_idx2 = start_idx - period2 + 1;
        let mut trailing_idx3 = start_idx - period3 + 1;
        
        for today in start_idx..len {
            // Add current bar to all windows
            bp1_total += bp[today];
            tr1_total += tr[today];
            bp2_total += bp[today];
            tr2_total += tr[today];
            bp3_total += bp[today];
            tr3_total += tr[today];
            
            // Calculate Ultimate Oscillator
            let mut output = 0.0;
            if tr1_total > 0.0 { output += 4.0 * (bp1_total / tr1_total); }
            if tr2_total > 0.0 { output += 2.0 * (bp2_total / tr2_total); }
            if tr3_total > 0.0 { output += bp3_total / tr3_total; }
            
            result[today] = Some(100.0 * (output / 7.0));
            
            // Remove oldest bars from each window
            bp1_total -= bp[trailing_idx1];
            tr1_total -= tr[trailing_idx1];
            bp2_total -= bp[trailing_idx2];
            tr2_total -= tr[trailing_idx2];
            bp3_total -= bp[trailing_idx3];
            tr3_total -= tr[trailing_idx3];
            
            trailing_idx1 += 1;
            trailing_idx2 += 1;
            trailing_idx3 += 1;
        }
    } else {
        // Fallback: same algorithm with Vec
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut bp = vec![0.0; len];
        let mut tr = vec![0.0; len];
        
        for i in 1..len {
            let true_low = low_vec[i].min(close_vec[i - 1]);
            bp[i] = close_vec[i] - true_low;
            
            let mut true_range = high_vec[i] - low_vec[i];
            let temp = (close_vec[i - 1] - high_vec[i]).abs();
            if temp > true_range { true_range = temp; }
            let temp = (close_vec[i - 1] - low_vec[i]).abs();
            if temp > true_range { true_range = temp; }
            
            tr[i] = true_range;
        }
        
        let mut bp1_total = 0.0;
        let mut tr1_total = 0.0;
        let mut bp2_total = 0.0;
        let mut tr2_total = 0.0;
        let mut bp3_total = 0.0;
        let mut tr3_total = 0.0;
        
        let start_idx = max_period;
        
        for i in (start_idx - period1 + 1)..start_idx {
            bp1_total += bp[i];
            tr1_total += tr[i];
        }
        for i in (start_idx - period2 + 1)..start_idx {
            bp2_total += bp[i];
            tr2_total += tr[i];
        }
        for i in (start_idx - period3 + 1)..start_idx {
            bp3_total += bp[i];
            tr3_total += tr[i];
        }
        
        let mut trailing_idx1 = start_idx - period1 + 1;
        let mut trailing_idx2 = start_idx - period2 + 1;
        let mut trailing_idx3 = start_idx - period3 + 1;
        
        for today in start_idx..len {
            bp1_total += bp[today];
            tr1_total += tr[today];
            bp2_total += bp[today];
            tr2_total += tr[today];
            bp3_total += bp[today];
            tr3_total += tr[today];
            
            let mut output = 0.0;
            if tr1_total > 0.0 { output += 4.0 * (bp1_total / tr1_total); }
            if tr2_total > 0.0 { output += 2.0 * (bp2_total / tr2_total); }
            if tr3_total > 0.0 { output += bp3_total / tr3_total; }
            
            result[today] = Some(100.0 * (output / 7.0));
            
            bp1_total -= bp[trailing_idx1];
            tr1_total -= tr[trailing_idx1];
            bp2_total -= bp[trailing_idx2];
            tr2_total -= tr[trailing_idx2];
            bp3_total -= bp[trailing_idx3];
            tr3_total -= tr[trailing_idx3];
            
            trailing_idx1 += 1;
            trailing_idx2 += 1;
            trailing_idx3 += 1;
        }
    }
    
    let result_series = Series::new(c.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 威廉指标 (WILLR)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn willr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("high must be numeric: {}", e)))?;
    let low_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("low must be numeric: {}", e)))?;
    let close_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("close must be numeric: {}", e)))?;
    
    let len = close_vals.len();
    let mut output = vec![f64::NAN; len];
    
    // 零拷贝 + tight loop
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        for i in period - 1..len {
            let start = i + 1 - period;
            
            // 查找窗口内最高/最低价
            let mut highest = h_arr[start];
            let mut lowest = l_arr[start];
            for j in start + 1..=i {
                if h_arr[j] > highest { highest = h_arr[j]; }
                if l_arr[j] < lowest { lowest = l_arr[j]; }
            }
            
            let range = highest - lowest;
            if range != 0.0 {
                output[i] = ((highest - c_arr[i]) / range) * -100.0;
            }
        }
    } else {
        // fallback
        let high_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        for i in period - 1..len {
            let start = i + 1 - period;
            let highest = high_vec[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = low_vec[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            if highest != lowest {
                output[i] = ((highest - close_vec[i]) / (highest - lowest)) * -100.0;
            }
        }
    }
    
    let result = Series::new(c.name().clone(), output);
    Ok(PySeries(result))
}

// ====================================================================
// 成交量指标 (Volume Indicators) - 成交量相关指标
// ====================================================================

/// 累积/派发线 (AD) - 零拷贝优化版本
#[pyfunction]
#[pyo3(signature = (high, low, close, volume))]
pub fn ad(high: PySeries, low: PySeries, close: PySeries, volume: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    let v: Series = volume.into();
    
    let c_name = c.name().clone();
    
    // 将 volume 转换为 f64 类型
    let v_f64 = if v.dtype() == &DataType::Int64 {
        v.cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to cast volume to Float64: {}", e)))?
    } else {
        v
    };
    
    let high_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("high must be numeric: {}", e)))?;
    let low_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("low must be numeric: {}", e)))?;
    let close_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("close must be numeric: {}", e)))?;
    let volume_vals = v_f64.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("volume must be numeric: {}", e)))?;
    
    let len = close_vals.len();
    let mut result = vec![0.0; len];
    
    // 零拷贝访问 + tight loop
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr), Ok(v_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice(), volume_vals.cont_slice()) {
        
        let mut cumsum = 0.0;
        
        // Tight loop: CLV计算 + 累加
        for i in 0..len {
            let hl_diff = h_arr[i] - l_arr[i];
            if hl_diff > 0.0 {
                // CLV = ((C - L) - (H - C)) / (H - L)
                let clv = ((c_arr[i] - l_arr[i]) - (h_arr[i] - c_arr[i])) / hl_diff;
                cumsum += clv * v_arr[i];
            }
            result[i] = cumsum;
        }
    } else {
        // Fallback
        let h_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let l_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let c_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let v_vec: Vec<f64> = volume_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut cumsum = 0.0;
        
        for i in 0..len {
            let hl_diff = h_vec[i] - l_vec[i];
            if hl_diff > 0.0 {
                let clv = ((c_vec[i] - l_vec[i]) - (h_vec[i] - c_vec[i])) / hl_diff;
                cumsum += clv * v_vec[i];
            }
            result[i] = cumsum;
        }
    }
    
    Ok(PySeries(Series::new(c_name, result)))
}

/// 累积/派发摆动指标 (ADOSC)
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, fast_period=3, slow_period=10))]
pub fn adosc(high: PySeries, low: PySeries, close: PySeries, volume: PySeries, fast_period: usize, slow_period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    let v: Series = volume.into();
    
    let c_name = c.name().clone();
    
    // 将 volume 转换为 f64 类型
    let v_f64 = if v.dtype() == &DataType::Int64 {
        v.cast(&DataType::Float64).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to cast volume to Float64: {}", e)))?
    } else {
        v
    };
    
    let high_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("high must be numeric: {}", e)))?;
    let low_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("low must be numeric: {}", e)))?;
    let close_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("close must be numeric: {}", e)))?;
    let volume_vals = v_f64.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("volume must be numeric: {}", e)))?;
    
    let len = close_vals.len();
    let mut result = vec![None; len];
    
    // Zero-copy optimization
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr), Ok(v_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice(), volume_vals.cont_slice()) {
        
        // Step 1: 计算AD线（内联，避免额外内存）
        let mut ad_values = vec![0.0; len];
        let mut cumsum = 0.0;
        
        for i in 0..len {
            let hl_diff = h_arr[i] - l_arr[i];
            if hl_diff > 0.0 {
                let clv = ((c_arr[i] - l_arr[i]) - (h_arr[i] - c_arr[i])) / hl_diff;
                cumsum += clv * v_arr[i];
            }
            ad_values[i] = cumsum;
        }
        
        // Step 2: 计算EMA（内联优化）
        let fast_k = 2.0 / (fast_period as f64 + 1.0);
        let slow_k = 2.0 / (slow_period as f64 + 1.0);
        
        let mut fast_ema = ad_values[0];
        let mut slow_ema = ad_values[0];
        
        for i in 1..len {
            fast_ema = ad_values[i] * fast_k + fast_ema * (1.0 - fast_k);
            slow_ema = ad_values[i] * slow_k + slow_ema * (1.0 - slow_k);
            
            // 从慢线周期开始输出
            if i >= slow_period - 1 {
                result[i] = Some(fast_ema - slow_ema);
            }
        }
    } else {
        // Fallback
        let h_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let l_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let c_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let v_vec: Vec<f64> = volume_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        let mut ad_values = vec![0.0; len];
        let mut cumsum = 0.0;
        
        for i in 0..len {
            let hl_diff = h_vec[i] - l_vec[i];
            if hl_diff > 0.0 {
                let clv = ((c_vec[i] - l_vec[i]) - (h_vec[i] - c_vec[i])) / hl_diff;
                cumsum += clv * v_vec[i];
            }
            ad_values[i] = cumsum;
        }
        
        let fast_k = 2.0 / (fast_period as f64 + 1.0);
        let slow_k = 2.0 / (slow_period as f64 + 1.0);
        
        let mut fast_ema = ad_values[0];
        let mut slow_ema = ad_values[0];
        
        for i in 1..len {
            fast_ema = ad_values[i] * fast_k + fast_ema * (1.0 - fast_k);
            slow_ema = ad_values[i] * slow_k + slow_ema * (1.0 - slow_k);
            
            if i >= slow_period - 1 {
                result[i] = Some(fast_ema - slow_ema);
            }
        }
    }
    
    Ok(PySeries(Series::new(c_name, result)))
}

/// 能量潮指标 (OBV) - 零拷贝优化版本
#[pyfunction]
#[pyo3(signature = (close, volume))]
pub fn obv(close: PySeries, volume: PySeries) -> PyResult<PySeries> {
    let c: Series = close.into();
    let v: Series = volume.into();
    
    // 将 volume 转换为 f64 类型
    let v_f64 = if v.dtype() == &DataType::Int64 {
        v.cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to cast volume to Float64: {}", e)))?
    } else {
        v
    };
    
    let close_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("close must be numeric: {}", e)))?;
    let volume_vals = v_f64.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("volume must be numeric: {}", e)))?;
    
    let len = close_vals.len();
    let mut result = vec![0.0; len];
    
    if len == 0 {
        return Ok(PySeries(Series::new(c.name().clone(), result)));
    }
    
    // 零拷贝访问 + 无分支优化
    if let (Ok(c_arr), Ok(v_arr)) = (close_vals.cont_slice(), volume_vals.cont_slice()) {
        result[0] = v_arr[0];
        let mut obv = result[0];
        
        // 使用符号函数
        for i in 1..len {
            let diff = c_arr[i] - c_arr[i - 1];
            // signum: 1.0 if diff > 0, -1.0 if diff < 0, 0.0 if diff == 0
            let sign = if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
            obv += sign * v_arr[i];
            result[i] = obv;
        }
    } else {
        // Fallback
        let close_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let volume_vec: Vec<f64> = volume_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        
        result[0] = volume_vec[0];
        let mut obv = result[0];
        
        for i in 1..len {
            let diff = close_vec[i] - close_vec[i - 1];
            let sign = if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
            obv += sign * volume_vec[i];
            result[i] = obv;
        }
    }
    
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

// ============================================================================
// 波动性指标 (Volatility Indicators)
// ============================================================================

/// 真实波幅 (TRANGE) - SIMD优化版本
#[pyfunction]
#[pyo3(signature = (high, low, close))]
pub fn trange(high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = close_vals.len();
    let mut result = vec![f64::NAN; len];
    
    if len == 0 {
        let result_series = Series::new(c.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // 连续内存访问
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        // 第一个值就是 high - low
        result[0] = h_arr[0] - l_arr[0];
        
        // 紧密循环: 内联abs和max运算
        for i in 1..len {
            let h = h_arr[i];
            let l = l_arr[i];
            let c_prev = c_arr[i - 1];
            
            let hl = h - l;
            let hc_diff = h - c_prev;
            let lc_diff = l - c_prev;
            
            // 内联abs: 使用位运算或条件移动
            let hc = if hc_diff >= 0.0 { hc_diff } else { -hc_diff };
            let lc = if lc_diff >= 0.0 { lc_diff } else { -lc_diff };
            
            // 三者取最大
            let max_hl_hc = if hl > hc { hl } else { hc };
            result[i] = if max_hl_hc > lc { max_hl_hc } else { lc };
        }
    } else {
        // 后备路径
        result[0] = high_vals.get(0).unwrap_or(0.0) - low_vals.get(0).unwrap_or(0.0);
        
        for i in 1..len {
            let h = high_vals.get(i).unwrap_or(0.0);
            let l = low_vals.get(i).unwrap_or(0.0);
            let c_prev = close_vals.get(i - 1).unwrap_or(0.0);
            
            let hl = h - l;
            let hc_diff = h - c_prev;
            let lc_diff = l - c_prev;
            
            let hc = if hc_diff >= 0.0 { hc_diff } else { -hc_diff };
            let lc = if lc_diff >= 0.0 { lc_diff } else { -lc_diff };
            
            let max_hl_hc = if hl > hc { hl } else { hc };
            result[i] = if max_hl_hc > lc { max_hl_hc } else { lc };
        }
    }
    
    let result_series = Series::new(c.name().clone(), result);
    Ok(PySeries(result_series))
}

// ====================================================================
// 波动率指标 (Volatility Indicators) - 价格波动测量
// ====================================================================

/// 平均真实波幅 (ATR) - SIMD优化版本
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn atr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    // 连续内存访问
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = high_vals.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        let result_series = Series::new(c.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // 尝试零拷贝路径
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        // 直接在一次遍历中计算TR和累积ATR
        let first_tr = h_arr[0] - l_arr[0];
        let mut tr_sum = first_tr;
        
        // 累积前period个TR值
        for i in 1..period {
            let tr1 = h_arr[i] - l_arr[i];
            let tr2 = (h_arr[i] - c_arr[i - 1]).abs();
            let tr3 = (l_arr[i] - c_arr[i - 1]).abs();
            tr_sum += tr1.max(tr2).max(tr3);
        }
        
        result[period - 1] = tr_sum / period as f64;
        
        // Wilder平滑
        let smoothing_factor = 1.0 / period as f64;
        let retention_factor = (period - 1) as f64 / period as f64;
        
        for i in period..len {
            let tr1 = h_arr[i] - l_arr[i];
            let tr2 = (h_arr[i] - c_arr[i - 1]).abs();
            let tr3 = (l_arr[i] - c_arr[i - 1]).abs();
            let tr = tr1.max(tr2).max(tr3);
            result[i] = result[i - 1] * retention_factor + tr * smoothing_factor;
        }
    } else {
        // 后备路径: 使用Vec
        let high_vals_vec: Vec<f64> = high_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let low_vals_vec: Vec<f64> = low_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        let close_vals_vec: Vec<f64> = close_vals.into_iter().map(|x| x.unwrap_or(0.0)).collect();
        result = calculate_atr(&high_vals_vec, &low_vals_vec, &close_vals_vec, period);
    }
    
    let result_series = Series::new(c.name().clone(), result);
    Ok(PySeries(result_series))
}

/// 标准化平均真实波幅 (NATR)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn natr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    // 直接计算ATR和NATR
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    
    let len = high_vals.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        let result_series = Series::new(c.name().clone(), result);
        return Ok(PySeries(result_series));
    }
    
    // 尝试零拷贝路径
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        
        // 计算ATR值(与atr函数相同的逻辑)
        let first_tr = h_arr[0] - l_arr[0];
        let mut tr_sum = first_tr;
        
        for i in 1..period {
            let tr1 = h_arr[i] - l_arr[i];
            let tr2 = (h_arr[i] - c_arr[i - 1]).abs();
            let tr3 = (l_arr[i] - c_arr[i - 1]).abs();
            tr_sum += tr1.max(tr2).max(tr3);
        }
        
        let mut atr = tr_sum / period as f64;
        if c_arr[period - 1] > 0.0 {
            result[period - 1] = (atr / c_arr[period - 1]) * 100.0;
        }
        
        // Wilder平滑 + 立即计算NATR
        let smoothing_factor = 1.0 / period as f64;
        let retention_factor = (period - 1) as f64 / period as f64;
        
        for i in period..len {
            let tr1 = h_arr[i] - l_arr[i];
            let tr2 = (h_arr[i] - c_arr[i - 1]).abs();
            let tr3 = (l_arr[i] - c_arr[i - 1]).abs();
            let tr = tr1.max(tr2).max(tr3);
            atr = atr * retention_factor + tr * smoothing_factor;
            
            // 立即计算NATR: (ATR / Close) * 100
            if c_arr[i] > 0.0 {
                result[i] = (atr / c_arr[i]) * 100.0;
            }
        }
    } else {
        // 后备路径: 先计算ATR,再计算NATR
        let atr_result = atr(PySeries(h.clone()), PySeries(l.clone()), PySeries(c.clone()), period)?;
        let atr_series: Series = atr_result.into();
        let atr_vals = atr_series.f64().unwrap();
        
        for i in 0..len {
            if let (Some(atr_val), Some(close_val)) = (atr_vals.get(i), close_vals.get(i)) {
                if !atr_val.is_nan() && close_val > 0.0 {
                    result[i] = (atr_val / close_val) * 100.0;
                }
            }
        }
    }
    
    let result_series = Series::new(c.name().clone(), result);
    Ok(PySeries(result_series))
}

// ============================================================================
// 价格变换函数 (Price Transform Functions)
// ====================================================================
// 价格变换函数 (Price Transform) - 价格数据变换
// ====================================================================

/// 平均价格 (AVGPRICE) - SIMD优化版本
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn avgprice(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals = o.f64().unwrap();
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    let len = close_vals.len();
    let mut result = vec![0.0; len];
    
    if let (Ok(o_arr), Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (open_vals.cont_slice(), high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        for i in 0..len {
            result[i] = (o_arr[i] + h_arr[i] + l_arr[i] + c_arr[i]) * 0.25;
        }
    } else {
        for i in 0..len {
            result[i] = (open_vals.get(i).unwrap_or(0.0) + high_vals.get(i).unwrap_or(0.0) + 
                        low_vals.get(i).unwrap_or(0.0) + close_vals.get(i).unwrap_or(0.0)) * 0.25;
        }
    }
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

/// 中间价格 (MEDPRICE) - 简洁优化
#[pyfunction]
#[pyo3(signature = (high, low))]
pub fn medprice(high: PySeries, low: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let len = high_vals.len();
    let mut result = vec![0.0; len];
    
    if let (Ok(h_arr), Ok(l_arr)) = (high_vals.cont_slice(), low_vals.cont_slice()) {
        for i in 0..len {
            result[i] = (h_arr[i] + l_arr[i]) * 0.5;
        }
    } else {
        for i in 0..len {
            result[i] = (high_vals.get(i).unwrap_or(0.0) + low_vals.get(i).unwrap_or(0.0)) * 0.5;
        }
    }
    Ok(PySeries(Series::new(h.name().clone(), result)))
}

/// 典型价格 (TYPPRICE) - 简洁优化
#[pyfunction]
#[pyo3(signature = (high, low, close))]
pub fn typprice(high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    let len = close_vals.len();
    let mut result = vec![0.0; len];
    let one_third = 1.0 / 3.0;
    
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        for i in 0..len {
            result[i] = (h_arr[i] + l_arr[i] + c_arr[i]) * one_third;
        }
    } else {
        for i in 0..len {
            result[i] = (high_vals.get(i).unwrap_or(0.0) + low_vals.get(i).unwrap_or(0.0) + 
                        close_vals.get(i).unwrap_or(0.0)) * one_third;
        }
    }
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

/// 加权收盘价 (WCLPRICE) - 简洁优化
#[pyfunction]
#[pyo3(signature = (high, low, close))]
pub fn wclprice(high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals = h.f64().unwrap();
    let low_vals = l.f64().unwrap();
    let close_vals = c.f64().unwrap();
    let len = close_vals.len();
    let mut result = vec![0.0; len];
    
    if let (Ok(h_arr), Ok(l_arr), Ok(c_arr)) = 
        (high_vals.cont_slice(), low_vals.cont_slice(), close_vals.cont_slice()) {
        for i in 0..len {
            result[i] = (h_arr[i] + l_arr[i] + c_arr[i] * 2.0) * 0.25;
        }
    } else {
        for i in 0..len {
            let c_val = close_vals.get(i).unwrap_or(0.0);
            result[i] = (high_vals.get(i).unwrap_or(0.0) + low_vals.get(i).unwrap_or(0.0) + c_val * 2.0) * 0.25;
        }
    }
    Ok(PySeries(Series::new(c.name().clone(), result)))
}

// ============================================================================
// 周期指标 (Cycle Indicators) - 希尔伯特变换系列
// ====================================================================
// 周期指标 (Cycle Indicators) - 希尔伯特变换系列
// ====================================================================

/// 希尔伯特变换 - 瞬时趋势线 (HT_TRENDLINE)
#[pyfunction]
#[pyo3(signature = (series))]
pub fn ht_trendline(series: PySeries) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let ht_trendline_values = {
        let mut result = vec![None; vec_values.len()];
        
        // 简化的希尔伯特变换实现 - 使用数字滤波器方法
        if vec_values.len() >= 7 {
            for i in 6..vec_values.len() {
                // 使用7点希尔伯特变换滤波器
                let ht_value = (vec_values[i] + 2.0 * vec_values[i-2] + 3.0 * vec_values[i-4] + 3.0 * vec_values[i-6]) / 10.5;
                result[i] = Some(ht_value);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), ht_trendline_values);
    Ok(PySeries(result))
}

/// 希尔伯特变换 - 主导周期 (HT_DCPERIOD)
#[pyfunction]
#[pyo3(signature = (series))]
pub fn ht_dcperiod(series: PySeries) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let ht_dcperiod_values = {
        let mut result = vec![None; vec_values.len()];
        
        // 简化的主导周期计算
        if vec_values.len() >= 50 {
            for i in 49..vec_values.len() {
                // 分析最近50个数据点的周期性
                let window = &vec_values[i-49..=i];
                
                // 简单的周期检测 - 寻找重复模式
                let mut best_period = 15.0; // 默认周期
                let mut max_correlation = 0.0;
                
                for period in 8..=48 {
                    if i >= period * 2 {
                        let mut correlation = 0.0;
                        let mut count = 0;
                        
                        for j in 0..(window.len() - period) {
                            correlation += (window[j] - window[j + period]).abs();
                            count += 1;
                        }
                        
                        if count > 0 {
                            correlation = 1.0 / (1.0 + correlation / count as f64);
                            if correlation > max_correlation {
                                max_correlation = correlation;
                                best_period = period as f64;
                            }
                        }
                    }
                }
                
                result[i] = Some(best_period);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), ht_dcperiod_values);
    Ok(PySeries(result))
}

/// 希尔伯特变换 - 主导周期相位 (HT_DCPHASE)
#[pyfunction]
#[pyo3(signature = (series))]
pub fn ht_dcphase(series: PySeries) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let ht_dcphase_values = {
        let mut result = vec![None; vec_values.len()];
        
        if vec_values.len() >= 10 {
            for i in 9..vec_values.len() {
                // 简化的相位计算 - 基于价格变化
                let mut phase_sum = 0.0;
                for j in 1..=5 {
                    let change = vec_values[i - j + 1] - vec_values[i - j];
                    phase_sum += change * (j as f64);
                }
                
                // 将相位归一化到0-360度
                let phase = (phase_sum.atan2(1.0) * 180.0 / std::f64::consts::PI + 360.0) % 360.0;
                result[i] = Some(phase);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), ht_dcphase_values);
    Ok(PySeries(result))
}

/// 希尔伯特变换 - 相量组件 (HT_PHASOR)
#[pyfunction]
#[pyo3(signature = (series))]
pub fn ht_phasor(series: PySeries) -> PyResult<(PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let (inphase_values, quadrature_values) = {
        let mut inphase = vec![None; vec_values.len()];
        let mut quadrature = vec![None; vec_values.len()];
        
        if vec_values.len() >= 7 {
            for i in 6..vec_values.len() {
                // 计算实部（同相分量）
                let i_component = vec_values[i] - vec_values[i-6];
                
                // 计算虚部（正交分量）- 使用希尔伯特变换
                let q_component = (vec_values[i-2] - vec_values[i-4]) * 0.5;
                
                inphase[i] = Some(i_component);
                quadrature[i] = Some(q_component);
            }
        }
        
        (inphase, quadrature)
    };
    
    let base_name = s.name();
    let inphase_series = Series::new(PlSmallStr::from_str(&format!("{}_ht_inphase", base_name)), inphase_values);
    let quadrature_series = Series::new(PlSmallStr::from_str(&format!("{}_ht_quadrature", base_name)), quadrature_values);
    
    Ok((PySeries(inphase_series), PySeries(quadrature_series)))
}

/// 希尔伯特变换 - 正弦波 (HT_SINE)
#[pyfunction]
#[pyo3(signature = (series))]
pub fn ht_sine(series: PySeries) -> PyResult<(PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let (sine_values, lead_sine_values) = {
        let mut sine = vec![None; vec_values.len()];
        let mut lead_sine = vec![None; vec_values.len()];
        
        if vec_values.len() >= 10 {
            for i in 9..vec_values.len() {
                // 计算相位
                let mut phase = 0.0;
                for j in 1..=5 {
                    phase += (vec_values[i - j + 1] - vec_values[i - j]) * (j as f64);
                }
                phase = phase / 15.0; // 归一化
                
                // 计算正弦波
                let sine_val = phase.sin();
                let lead_sine_val = (phase + std::f64::consts::PI / 4.0).sin(); // 领先45度
                
                sine[i] = Some(sine_val);
                lead_sine[i] = Some(lead_sine_val);
            }
        }
        
        (sine, lead_sine)
    };
    
    let base_name = s.name();
    let sine_series = Series::new(PlSmallStr::from_str(&format!("{}_ht_sine", base_name)), sine_values);
    let lead_sine_series = Series::new(PlSmallStr::from_str(&format!("{}_ht_leadsine", base_name)), lead_sine_values);
    
    Ok((PySeries(sine_series), PySeries(lead_sine_series)))
}

/// 希尔伯特变换 - 趋势vs周期模式 (HT_TRENDMODE)
#[pyfunction]
#[pyo3(signature = (series))]
pub fn ht_trendmode(series: PySeries) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let ht_trendmode_values = {
        let mut result = vec![None; vec_values.len()];
        
        if vec_values.len() >= 20 {
            for i in 19..vec_values.len() {
                // 分析趋势强度 vs 周期性
                let window = &vec_values[i-19..=i];
                
                // 计算线性趋势强度
                let n = window.len() as f64;
                let sum_x: f64 = (0..window.len()).map(|x| x as f64).sum();
                let sum_y: f64 = window.iter().sum();
                let sum_xy: f64 = window.iter().enumerate().map(|(x, &y)| x as f64 * y).sum();
                let sum_x2: f64 = (0..window.len()).map(|x| (x as f64).powi(2)).sum();
                
                let trend_strength = if n * sum_x2 - sum_x * sum_x != 0.0 {
                    ((n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)).abs()
                } else {
                    0.0
                };
                
                // 计算价格波动性（周期性指标）
                let mean = sum_y / n;
                let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
                let volatility = variance.sqrt();
                
                // 趋势模式：1表示趋势，0表示周期
                let trend_mode = if trend_strength > volatility * 0.1 { 1.0 } else { 0.0 };
                
                result[i] = Some(trend_mode);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), ht_trendmode_values);
    Ok(PySeries(result))
}

// ======================== 蜡烛图模式识别函数 ========================

// ====================================================================
// 蜡烛图模式识别 (Candlestick Pattern Recognition) - K线形态
// ====================================================================

// CDL2CROWS - 两只乌鸦 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl2crows(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    // 连续内存访问
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(),
        h_vals.cont_slice(),
        l_vals.cont_slice(),
        c_vals.cont_slice()
    ) {
        // Tight loop: 所有计算内联，避免函数调用
        for i in 2..len {
            // 第一根蜡烛：长阳线
            let o0 = o_slice[i-2];
            let h0 = h_slice[i-2];
            let l0 = l_slice[i-2];
            let c0 = c_slice[i-2];
            let body0 = (c0 - o0).abs();
            let range0 = h0 - l0;
            let day1_bullish = c0 > o0;
            let is_long_body1 = body0 > range0 * 0.6;
            
            // 第二根蜡烛：小阴线
            let o1 = o_slice[i-1];
            let c1 = c_slice[i-1];
            let day2_bearish = c1 <= o1;
            let gap_up = o1 > c0;
            let close_in_body1 = c1 < c0 && c1 > o0;
            
            // 第三根蜡烛：阴线
            let o2 = o_slice[i];
            let c2 = c_slice[i];
            let day3_bearish = c2 <= o2;
            let gap_up2 = o2 > c1;
            let lower_close = c2 < c1;
            
            result[i] = if day1_bullish && is_long_body1 && day2_bearish && gap_up && close_in_body1 &&
                           day3_bearish && gap_up2 && lower_close { -100 } else { 0 };
        }
    } else {
        // Fallback: 非连续内存
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0);
            let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0);
            let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let body0 = (c0 - o0).abs();
            let range0 = h0 - l0;
            let day1_bullish = c0 > o0;
            let is_long_body1 = body0 > range0 * 0.6;
            
            let o1 = o_vals.get(i-1).unwrap_or(0.0);
            let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let day2_bearish = c1 <= o1;
            let gap_up = o1 > c0;
            let close_in_body1 = c1 < c0 && c1 > o0;
            
            let o2 = o_vals.get(i).unwrap_or(0.0);
            let c2 = c_vals.get(i).unwrap_or(0.0);
            let day3_bearish = c2 <= o2;
            let gap_up2 = o2 > c1;
            let lower_close = c2 < c1;
            
            result[i] = if day1_bullish && is_long_body1 && day2_bearish && gap_up && close_in_body1 &&
                           day3_bearish && gap_up2 && lower_close { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3BLACKCROWS - 三只黑乌鸦 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3blackcrows(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let o1 = o_slice[i-1]; let h1 = h_slice[i-1]; let l1 = l_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let h2 = h_slice[i]; let l2 = l_slice[i]; let c2 = c_slice[i];
            
            let day1_bearish = c0 <= o0;
            let day2_bearish = c1 <= o1;
            let day3_bearish = c2 <= o2;
            
            let body0 = (c0 - o0).abs(); let range0 = h0 - l0;
            let body1 = (c1 - o1).abs(); let range1 = h1 - l1;
            let body2 = (c2 - o2).abs(); let range2 = h2 - l2;
            
            let long_body1 = body0 > range0 * 0.6;
            let long_body2 = body1 > range1 * 0.6;
            let long_body3 = body2 > range2 * 0.6;
            
            let decreasing_opens = o1 < o0 && o2 < o1;
            let decreasing_closes = c1 < c0 && c2 < c1;
            let open_in_body1 = o1 < o0 && o1 > c0;
            let open_in_body2 = o2 < o1 && o2 > c1;
            
            result[i] = if day1_bearish && day2_bearish && day3_bearish &&
                           long_body1 && long_body2 && long_body3 &&
                           decreasing_opens && decreasing_closes &&
                           open_in_body1 && open_in_body2 { -100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let o1 = o_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0);
            let l1 = l_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let h2 = h_vals.get(i).unwrap_or(0.0);
            let l2 = l_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bearish = c0 <= o0;
            let day2_bearish = c1 <= o1;
            let day3_bearish = c2 <= o2;
            
            let body0 = (c0 - o0).abs(); let range0 = h0 - l0;
            let body1 = (c1 - o1).abs(); let range1 = h1 - l1;
            let body2 = (c2 - o2).abs(); let range2 = h2 - l2;
            
            let long_body1 = body0 > range0 * 0.6;
            let long_body2 = body1 > range1 * 0.6;
            let long_body3 = body2 > range2 * 0.6;
            
            let decreasing_opens = o1 < o0 && o2 < o1;
            let decreasing_closes = c1 < c0 && c2 < c1;
            let open_in_body1 = o1 < o0 && o1 > c0;
            let open_in_body2 = o2 < o1 && o2 > c1;
            
            result[i] = if day1_bearish && day2_bearish && day3_bearish &&
                           long_body1 && long_body2 && long_body3 &&
                           decreasing_opens && decreasing_closes &&
                           open_in_body1 && open_in_body2 { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3WHITESOLDIERS - 三个白兵 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3whitesoldiers(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let o1 = o_slice[i-1]; let h1 = h_slice[i-1]; let l1 = l_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let h2 = h_slice[i]; let l2 = l_slice[i]; let c2 = c_slice[i];
            
            let day1_bullish = c0 > o0;
            let day2_bullish = c1 > o1;
            let day3_bullish = c2 > o2;
            
            let body0 = (c0 - o0).abs(); let range0 = h0 - l0;
            let body1 = (c1 - o1).abs(); let range1 = h1 - l1;
            let body2 = (c2 - o2).abs(); let range2 = h2 - l2;
            
            let long_body1 = body0 > range0 * 0.6;
            let long_body2 = body1 > range1 * 0.6;
            let long_body3 = body2 > range2 * 0.6;
            
            let increasing_opens = o1 > o0 && o2 > o1;
            let increasing_closes = c1 > c0 && c2 > c1;
            let open_in_body1 = o1 > o0 && o1 < c0;
            let open_in_body2 = o2 > o1 && o2 < c1;
            
            let upper_shadow0 = h0 - c0.max(o0);
            let upper_shadow1 = h1 - c1.max(o1);
            let upper_shadow2 = h2 - c2.max(o2);
            let short_shadows = upper_shadow0 < body0 * 0.3 && upper_shadow1 < body1 * 0.3 && upper_shadow2 < body2 * 0.3;
            
            result[i] = if day1_bullish && day2_bullish && day3_bullish &&
                           long_body1 && long_body2 && long_body3 &&
                           increasing_opens && increasing_closes &&
                           open_in_body1 && open_in_body2 && short_shadows { 100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let o1 = o_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0);
            let l1 = l_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let h2 = h_vals.get(i).unwrap_or(0.0);
            let l2 = l_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bullish = c0 > o0;
            let day2_bullish = c1 > o1;
            let day3_bullish = c2 > o2;
            
            let body0 = (c0 - o0).abs(); let range0 = h0 - l0;
            let body1 = (c1 - o1).abs(); let range1 = h1 - l1;
            let body2 = (c2 - o2).abs(); let range2 = h2 - l2;
            
            let long_body1 = body0 > range0 * 0.6;
            let long_body2 = body1 > range1 * 0.6;
            let long_body3 = body2 > range2 * 0.6;
            
            let increasing_opens = o1 > o0 && o2 > o1;
            let increasing_closes = c1 > c0 && c2 > c1;
            let open_in_body1 = o1 > o0 && o1 < c0;
            let open_in_body2 = o2 > o1 && o2 < c1;
            
            let upper_shadow0 = h0 - c0.max(o0);
            let upper_shadow1 = h1 - c1.max(o1);
            let upper_shadow2 = h2 - c2.max(o2);
            let short_shadows = upper_shadow0 < body0 * 0.3 && upper_shadow1 < body1 * 0.3 && upper_shadow2 < body2 * 0.3;
            
            result[i] = if day1_bullish && day2_bullish && day3_bullish &&
                           long_body1 && long_body2 && long_body3 &&
                           increasing_opens && increasing_closes &&
                           open_in_body1 && open_in_body2 && short_shadows { 100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLDOJI - 十字星 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdldoji(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];  // 直接使用 i32，不用 Option
    
    // 连续内存访问
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(),
        h_vals.cont_slice(),
        l_vals.cont_slice(),
        c_vals.cont_slice()
    ) {
        // Tight loop: 内联所有计算，避免函数调用
        for i in 0..len {
            let open = o_slice[i];
            let high = h_slice[i];
            let low = l_slice[i];
            let close = c_slice[i];
            
            // 内联 candle_metrics 计算
            let body = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            // 内联 is_doji_body 和条件判断
            let is_doji = body < range * 0.1;
            let has_shadows = upper_shadow > body && lower_shadow > body;
            
            result[i] = if is_doji && has_shadows { 100 } else { 0 };
        }
    } else {
        // Fallback: 非连续内存路径
        for i in 0..len {
            let open = o_vals.get(i).unwrap_or(0.0);
            let high = h_vals.get(i).unwrap_or(0.0);
            let low = l_vals.get(i).unwrap_or(0.0);
            let close = c_vals.get(i).unwrap_or(0.0);
            
            let body = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            let is_doji = body < range * 0.1;
            let has_shadows = upper_shadow > body && lower_shadow > body;
            
            result[i] = if is_doji && has_shadows { 100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHAMMER - 锤子线 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlhammer(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    // 连续内存访问
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(),
        h_vals.cont_slice(),
        l_vals.cont_slice(),
        c_vals.cont_slice()
    ) {
        // Tight loop: 所有计算内联
        for i in 0..len {
            let open = o_slice[i];
            let high = h_slice[i];
            let low = l_slice[i];
            let close = c_slice[i];
            
            // 内联计算
            let body = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            // 锤子线特征：小实体，短上影线，长下影线
            let small_body = body < range * 0.3;
            let short_upper_shadow = upper_shadow < body * 0.5;
            let long_lower_shadow = lower_shadow > body * 2.0;
            
            result[i] = if small_body && short_upper_shadow && long_lower_shadow { 100 } else { 0 };
        }
    } else {
        // Fallback: 非连续内存
        for i in 0..len {
            let open = o_vals.get(i).unwrap_or(0.0);
            let high = h_vals.get(i).unwrap_or(0.0);
            let low = l_vals.get(i).unwrap_or(0.0);
            let close = c_vals.get(i).unwrap_or(0.0);
            
            let body = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            let small_body = body < range * 0.3;
            let short_upper_shadow = upper_shadow < body * 0.5;
            let long_lower_shadow = lower_shadow > body * 2.0;
            
            result[i] = if small_body && short_upper_shadow && long_lower_shadow { 100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHANGINGMAN - 上吊线 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlhangingman(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 0..len {
            let open = o_slice[i]; let high = h_slice[i]; let low = l_slice[i]; let close = c_slice[i];
            let body_size = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            let small_body = body_size < range * 0.3;
            let short_upper_shadow = upper_shadow < body_size * 0.5;
            let long_lower_shadow = lower_shadow > body_size * 2.0;
            
            result[i] = if small_body && short_upper_shadow && long_lower_shadow { -100 } else { 0 };
        }
    } else {
        for i in 0..len {
            let open = o_vals.get(i).unwrap_or(0.0); let high = h_vals.get(i).unwrap_or(0.0);
            let low = l_vals.get(i).unwrap_or(0.0); let close = c_vals.get(i).unwrap_or(0.0);
            let body_size = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            let small_body = body_size < range * 0.3;
            let short_upper_shadow = upper_shadow < body_size * 0.5;
            let long_lower_shadow = lower_shadow > body_size * 2.0;
            
            result[i] = if small_body && short_upper_shadow && long_lower_shadow { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSHOOTINGSTAR - 流星线 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlshootingstar(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 0..len {
            let open = o_slice[i]; let high = h_slice[i]; let low = l_slice[i]; let close = c_slice[i];
            let body_size = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            let small_body = body_size < range * 0.3;
            let long_upper_shadow = upper_shadow > body_size * 2.0;
            let short_lower_shadow = lower_shadow < body_size * 0.5;
            
            result[i] = if small_body && long_upper_shadow && short_lower_shadow { -100 } else { 0 };
        }
    } else {
        for i in 0..len {
            let open = o_vals.get(i).unwrap_or(0.0); let high = h_vals.get(i).unwrap_or(0.0);
            let low = l_vals.get(i).unwrap_or(0.0); let close = c_vals.get(i).unwrap_or(0.0);
            let body_size = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let range = high - low;
            
            let small_body = body_size < range * 0.3;
            let long_upper_shadow = upper_shadow > body_size * 2.0;
            let short_lower_shadow = lower_shadow < body_size * 0.5;
            
            result[i] = if small_body && long_upper_shadow && short_lower_shadow { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLENGULFING - 吞没模式 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlengulfing(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 2 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    if let (Ok(o_slice), Ok(_h_slice), Ok(_l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 1..len {
            let o0 = o_slice[i-1]; let c0 = c_slice[i-1];
            let o1 = o_slice[i]; let c1 = c_slice[i];
            
            let day1_bullish = c0 > o0;
            let day2_bullish = c1 > o1;
            let body1 = (c0 - o0).abs();
            let body2 = (c1 - o1).abs();
            
            let bullish_engulfing = !day1_bullish && day2_bullish &&
                                   o1 < c0 && c1 > o0 && body2 > body1;
            
            let bearish_engulfing = day1_bullish && !day2_bullish &&
                                   o1 > c0 && c1 < o0 && body2 > body1;
            
            result[i] = if bullish_engulfing { 100 } else if bearish_engulfing { -100 } else { 0 };
        }
    } else {
        for i in 1..len {
            let o0 = o_vals.get(i-1).unwrap_or(0.0); let c0 = c_vals.get(i-1).unwrap_or(0.0);
            let o1 = o_vals.get(i).unwrap_or(0.0); let c1 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bullish = c0 > o0;
            let day2_bullish = c1 > o1;
            let body1 = (c0 - o0).abs();
            let body2 = (c1 - o1).abs();
            
            let bullish_engulfing = !day1_bullish && day2_bullish &&
                                   o1 < c0 && c1 > o0 && body2 > body1;
            
            let bearish_engulfing = day1_bullish && !day2_bullish &&
                                   o1 > c0 && c1 < o0 && body2 > body1;
            
            result[i] = if bullish_engulfing { 100 } else if bearish_engulfing { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHARAMI - 孕育线 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlharami(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 2 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 1..len {
            let o0 = o_slice[i-1]; let h0 = h_slice[i-1]; let l0 = l_slice[i-1]; let c0 = c_slice[i-1];
            let o1 = o_slice[i]; let h1 = h_slice[i]; let l1 = l_slice[i]; let c1 = c_slice[i];
            
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let long_body1 = body1 > range1 * 0.6;
            let small_body2 = body2 < range2 * 0.3;
            
            let prev_body_high = o0.max(c0); let prev_body_low = o0.min(c0);
            let curr_body_high = o1.max(c1); let curr_body_low = o1.min(c1);
            let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
            
            result[i] = if long_body1 && small_body2 && inside_body {
                let day1_bullish = c0 > o0; let day2_bullish = c1 > o1;
                if !day1_bullish && day2_bullish { 100 } else if day1_bullish && !day2_bullish { -100 } else { 50 }
            } else { 0 };
        }
    } else {
        for i in 1..len {
            let o0 = o_vals.get(i-1).unwrap_or(0.0); let h0 = h_vals.get(i-1).unwrap_or(0.0);
            let l0 = l_vals.get(i-1).unwrap_or(0.0); let c0 = c_vals.get(i-1).unwrap_or(0.0);
            let o1 = o_vals.get(i).unwrap_or(0.0); let h1 = h_vals.get(i).unwrap_or(0.0);
            let l1 = l_vals.get(i).unwrap_or(0.0); let c1 = c_vals.get(i).unwrap_or(0.0);
            
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let long_body1 = body1 > range1 * 0.6;
            let small_body2 = body2 < range2 * 0.3;
            
            let prev_body_high = o0.max(c0); let prev_body_low = o0.min(c0);
            let curr_body_high = o1.max(c1); let curr_body_low = o1.min(c1);
            let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
            
            result[i] = if long_body1 && small_body2 && inside_body {
                let day1_bullish = c0 > o0; let day2_bullish = c1 > o1;
                if !day1_bullish && day2_bullish { 100 } else if day1_bullish && !day2_bullish { -100 } else { 50 }
            } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLMORNINGSTAR - 启明星 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdlmorningstar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let h1 = h_slice[i-1]; let l1 = l_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let c2 = c_slice[i];
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_slice[i-1]).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            let gap_down = h1 < c0;
            
            let day3_bullish = c2 > o2;
            let penetrates = c2 > (o0 + c0) / 2.0 * (1.0 + penetration);
            
            result[i] = if day1_bearish && long_body1 && small_body2 && gap_down && day3_bullish && penetrates { 100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let h1 = h_vals.get(i-1).unwrap_or(0.0); let l1 = l_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_vals.get(i-1).unwrap_or(0.0)).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            let gap_down = h1 < c0;
            
            let day3_bullish = c2 > o2;
            let penetrates = c2 > (o0 + c0) / 2.0 * (1.0 + penetration);
            
            result[i] = if day1_bearish && long_body1 && small_body2 && gap_down && day3_bullish && penetrates { 100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLEVENINGSTAR - 黄昏星 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdleveningstar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        let series_result = Series::new(o.name().clone(), result);
        return Ok(PySeries(series_result));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let l1 = l_slice[i-1]; let h1 = h_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let c2 = c_slice[i];
            
            let day1_bullish = c0 > o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_slice[i-1]).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            let gap_up = l1 > c0;
            
            let day3_bearish = c2 <= o2;
            let penetrates = c2 < (o0 + c0) / 2.0 * (1.0 - penetration);
            
            result[i] = if day1_bullish && long_body1 && small_body2 && gap_up && day3_bearish && penetrates { -100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let l1 = l_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bullish = c0 > o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_vals.get(i-1).unwrap_or(0.0)).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            let gap_up = l1 > c0;
            
            let day3_bearish = c2 <= o2;
            let penetrates = c2 < (o0 + c0) / 2.0 * (1.0 - penetration);
            
            result[i] = if day1_bullish && long_body1 && small_body2 && gap_up && day3_bearish && penetrates { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLPIERCING - 穿刺线
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlpiercing(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 2 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(_h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 1..len {
            let o0 = o_slice[i-1]; let l0 = l_slice[i-1]; let c0 = c_slice[i-1];
            let o1 = o_slice[i]; let c1 = c_slice[i];
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h_vals.get(i-1).unwrap_or(0.0) - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let day2_bullish = c1 > o1;
            let opens_lower = o1 < l0;
            let midpoint = (o0 + c0) / 2.0;
            let closes_above_midpoint = c1 > midpoint;
            let closes_below_open = c1 < o0;
            
            result[i] = if day1_bearish && long_body1 && day2_bullish && opens_lower && 
                           closes_above_midpoint && closes_below_open { 100 } else { 0 };
        }
    } else {
        for i in 1..len {
            let o0 = o_vals.get(i-1).unwrap_or(0.0); let l0 = l_vals.get(i-1).unwrap_or(0.0); let c0 = c_vals.get(i-1).unwrap_or(0.0);
            let o1 = o_vals.get(i).unwrap_or(0.0); let c1 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h_vals.get(i-1).unwrap_or(0.0) - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let day2_bullish = c1 > o1;
            let opens_lower = o1 < l0;
            let midpoint = (o0 + c0) / 2.0;
            let closes_above_midpoint = c1 > midpoint;
            let closes_below_open = c1 < o0;
            
            result[i] = if day1_bearish && long_body1 && day2_bullish && opens_lower && 
                           closes_above_midpoint && closes_below_open { 100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLDARKCLOUDCOVER - 乌云盖顶
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.5))]
pub fn cdldarkcloudcover(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 2 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(_l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 1..len {
            let o0 = o_slice[i-1]; let h0 = h_slice[i-1]; let c0 = c_slice[i-1];
            let o1 = o_slice[i]; let c1 = c_slice[i];
            
            let day1_bullish = c0 > o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l_vals.get(i-1).unwrap_or(0.0);
            let long_body1 = body1 > range1 * 0.6;
            
            let day2_bearish = c1 <= o1;
            let opens_higher = o1 > h0;
            let midpoint = (o0 + c0) / 2.0;
            let closes_below_midpoint = c1 < midpoint;
            let closes_above_close = c1 > c0 * (1.0 - penetration);
            
            result[i] = if day1_bullish && long_body1 && day2_bearish && opens_higher && 
                           closes_below_midpoint && closes_above_close { -100 } else { 0 };
        }
    } else {
        for i in 1..len {
            let o0 = o_vals.get(i-1).unwrap_or(0.0); let h0 = h_vals.get(i-1).unwrap_or(0.0); let c0 = c_vals.get(i-1).unwrap_or(0.0);
            let o1 = o_vals.get(i).unwrap_or(0.0); let c1 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bullish = c0 > o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l_vals.get(i-1).unwrap_or(0.0);
            let long_body1 = body1 > range1 * 0.6;
            
            let day2_bearish = c1 <= o1;
            let opens_higher = o1 > h0;
            let midpoint = (o0 + c0) / 2.0;
            let closes_below_midpoint = c1 < midpoint;
            let closes_above_close = c1 > c0 * (1.0 - penetration);
            
            result[i] = if day1_bullish && long_body1 && day2_bearish && opens_higher && 
                           closes_below_midpoint && closes_above_close { -100 } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHARAMICROSS - 十字孕育线
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlharamicross(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open must be numeric: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High must be numeric: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low must be numeric: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close must be numeric: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 2 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 1..len {
            let o0 = o_slice[i-1]; let h0 = h_slice[i-1]; let l0 = l_slice[i-1]; let c0 = c_slice[i-1];
            let o1 = o_slice[i]; let h1 = h_slice[i]; let l1 = l_slice[i]; let c1 = c_slice[i];
            
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let is_doji2 = body2 < range2 * 0.1;
            
            let prev_body_high = o0.max(c0); let prev_body_low = o0.min(c0);
            let curr_body_high = o1.max(c1); let curr_body_low = o1.min(c1);
            let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
            
            result[i] = if long_body1 && is_doji2 && inside_body {
                if c0 <= o0 { 100 } else { -100 }
            } else { 0 };
        }
    } else {
        for i in 1..len {
            let o0 = o_vals.get(i-1).unwrap_or(0.0); let h0 = h_vals.get(i-1).unwrap_or(0.0);
            let l0 = l_vals.get(i-1).unwrap_or(0.0); let c0 = c_vals.get(i-1).unwrap_or(0.0);
            let o1 = o_vals.get(i).unwrap_or(0.0); let h1 = h_vals.get(i).unwrap_or(0.0);
            let l1 = l_vals.get(i).unwrap_or(0.0); let c1 = c_vals.get(i).unwrap_or(0.0);
            
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let is_doji2 = body2 < range2 * 0.1;
            
            let prev_body_high = o0.max(c0); let prev_body_low = o0.min(c0);
            let curr_body_high = o1.max(c1); let curr_body_low = o1.min(c1);
            let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
            
            result[i] = if long_body1 && is_doji2 && inside_body {
                if c0 <= o0 { 100 } else { -100 }
            } else { 0 };
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLMORNINGDOJISTAR - 启明十字星
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdlmorningdojistar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let h1 = h_slice[i-1]; let l1 = l_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let c2 = c_slice[i];
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_slice[i-1]).abs(); let range2 = h1 - l1;
            let is_doji2 = body2 < range2 * 0.1;
            let gap_down = h1 < c0;
            
            let day3_bullish = c2 > o2;
            let penetrates = c2 > (o0 + c0) / 2.0 * (1.0 + penetration);
            
            result[i] = if day1_bearish && long_body1 && is_doji2 && gap_down && day3_bullish && penetrates { 100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let h1 = h_vals.get(i-1).unwrap_or(0.0); let l1 = l_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_vals.get(i-1).unwrap_or(0.0)).abs(); let range2 = h1 - l1;
            let is_doji2 = body2 < range2 * 0.1;
            let gap_down = h1 < c0;
            
            let day3_bullish = c2 > o2;
            let penetrates = c2 > (o0 + c0) / 2.0 * (1.0 + penetration);
            
            result[i] = if day1_bearish && long_body1 && is_doji2 && gap_down && day3_bullish && penetrates { 100 } else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}

// CDLEVENINGDOJISTAR - 黄昏十字星
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdleveningdojistar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let l1 = l_slice[i-1]; let h1 = h_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let c2 = c_slice[i];
            
            let day1_bullish = c0 > o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_slice[i-1]).abs(); let range2 = h1 - l1;
            let is_doji2 = body2 < range2 * 0.1;
            let gap_up = l1 > c0;
            
            let day3_bearish = c2 <= o2;
            let penetrates = c2 < (o0 + c0) / 2.0 * (1.0 - penetration);
            
            result[i] = if day1_bullish && long_body1 && is_doji2 && gap_up && day3_bearish && penetrates { -100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let l1 = l_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bullish = c0 > o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            
            let body2 = (c1 - o_vals.get(i-1).unwrap_or(0.0)).abs(); let range2 = h1 - l1;
            let is_doji2 = body2 < range2 * 0.1;
            let gap_up = l1 > c0;
            
            let day3_bearish = c2 <= o2;
            let penetrates = c2 < (o0 + c0) / 2.0 * (1.0 - penetration);
            
            result[i] = if day1_bullish && long_body1 && is_doji2 && gap_up && day3_bearish && penetrates { -100 } else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}

// CDL3INSIDE - 三内部上升/下降
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3inside(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let o1 = o_slice[i-1]; let h1 = h_slice[i-1]; let l1 = l_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let c2 = c_slice[i];
            
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            
            let prev_body_high = o0.max(c0); let prev_body_low = o0.min(c0);
            let curr_body_high = o1.max(c1); let curr_body_low = o1.min(c1);
            let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
            
            let day1_bearish = c0 <= o0; let day2_bullish = c1 > o1;
            let day3_bullish = c2 > o2; let day3_bearish = c2 <= o2;
            
            result[i] = if day1_bearish && day2_bullish && long_body1 && small_body2 && inside_body && day3_bullish && c2 > o0 {
                100
            } else if !day1_bearish && !day2_bullish && long_body1 && small_body2 && inside_body && day3_bearish && c2 < o0 {
                -100
            } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let o1 = o_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0);
            let l1 = l_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let long_body1 = body1 > range1 * 0.6;
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            
            let prev_body_high = o0.max(c0); let prev_body_low = o0.min(c0);
            let curr_body_high = o1.max(c1); let curr_body_low = o1.min(c1);
            let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
            
            let day1_bearish = c0 <= o0; let day2_bullish = c1 > o1;
            let day3_bullish = c2 > o2; let day3_bearish = c2 <= o2;
            
            result[i] = if day1_bearish && day2_bullish && long_body1 && small_body2 && inside_body && day3_bullish && c2 > o0 {
                100
            } else if !day1_bearish && !day2_bullish && long_body1 && small_body2 && inside_body && day3_bearish && c2 < o0 {
                -100
            } else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}

// CDL3OUTSIDE - 三外部上升/下降 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3outside(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(_h_slice), Ok(_l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let c0 = c_slice[i-2];
            let o1 = o_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let c2 = c_slice[i];
            
            let day1_bearish = c0 <= o0; let day2_bullish = c1 > o1;
            let body1 = (c0 - o0).abs(); let body2 = (c1 - o1).abs();
            
            let bullish_engulfing = day1_bearish && day2_bullish && o1 < c0 && c1 > o0 && body2 > body1;
            let bearish_engulfing = !day1_bearish && !day2_bullish && o1 > c0 && c1 < o0 && body2 > body1;
            
            let day3_bullish = c2 > o2;
            
            result[i] = if bullish_engulfing && day3_bullish && c2 > c1 { 100 }
                       else if bearish_engulfing && !day3_bullish && c2 < c1 { -100 }
                       else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let o1 = o_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bearish = c0 <= o0; let day2_bullish = c1 > o1;
            let body1 = (c0 - o0).abs(); let body2 = (c1 - o1).abs();
            
            let bullish_engulfing = day1_bearish && day2_bullish && o1 < c0 && c1 > o0 && body2 > body1;
            let bearish_engulfing = !day1_bearish && !day2_bullish && o1 > c0 && c1 < o0 && body2 > body1;
            
            let day3_bullish = c2 > o2;
            
            result[i] = if bullish_engulfing && day3_bullish && c2 > c1 { 100 }
                       else if bearish_engulfing && !day3_bullish && c2 < c1 { -100 }
                       else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}

// CDL3LINESTRIKE - 三线攻击 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, _high, _low, close))]
pub fn cdl3linestrike(open: PySeries, _high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 4 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(c_slice)) = (o_vals.cont_slice(), c_vals.cont_slice()) {
        for i in 3..len {
            let o0 = o_slice[i-3]; let c0 = c_slice[i-3];
            let o1 = o_slice[i-2]; let c1 = c_slice[i-2];
            let o2 = o_slice[i-1]; let c2 = c_slice[i-1];
            let o3 = o_slice[i]; let c3 = c_slice[i];
            
            let day1_bullish = c0 > o0; let day2_bullish = c1 > o1; let day3_bullish = c2 > o2;
            
            let three_bulls = day1_bullish && day2_bullish && day3_bullish && c1 > c0 && c2 > c1;
            let three_bears = !day1_bullish && !day2_bullish && !day3_bullish && c1 < c0 && c2 < c1;
            
            let day4_bullish = c3 > o3;
            
            result[i] = if three_bears && day4_bullish && o3 < c2 && c3 > o0 { 100 }
                       else if three_bulls && !day4_bullish && o3 > c2 && c3 < o0 { -100 }
                       else { 0 };
        }
    } else {
        for i in 3..len {
            let o0 = o_vals.get(i-3).unwrap_or(0.0); let c0 = c_vals.get(i-3).unwrap_or(0.0);
            let o1 = o_vals.get(i-2).unwrap_or(0.0); let c1 = c_vals.get(i-2).unwrap_or(0.0);
            let o2 = o_vals.get(i-1).unwrap_or(0.0); let c2 = c_vals.get(i-1).unwrap_or(0.0);
            let o3 = o_vals.get(i).unwrap_or(0.0); let c3 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bullish = c0 > o0; let day2_bullish = c1 > o1; let day3_bullish = c2 > o2;
            
            let three_bulls = day1_bullish && day2_bullish && day3_bullish && c1 > c0 && c2 > c1;
            let three_bears = !day1_bullish && !day2_bullish && !day3_bullish && c1 < c0 && c2 < c1;
            
            let day4_bullish = c3 > o3;
            
            result[i] = if three_bears && day4_bullish && o3 < c2 && c3 > o0 { 100 }
                       else if three_bulls && !day4_bullish && o3 > c2 && c3 < o0 { -100 }
                       else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}

// CDL3STARSINSOUTH - 三星在南 (零拷贝 + tight loop 优化)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3starsinsouth(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High: {}", e)))?;
    let l_vals = l.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Low: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(l_slice), Ok(c_slice)) = (
        o_vals.cont_slice(), h_vals.cont_slice(), l_vals.cont_slice(), c_vals.cont_slice()
    ) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let l0 = l_slice[i-2]; let c0 = c_slice[i-2];
            let o1 = o_slice[i-1]; let h1 = h_slice[i-1]; let l1 = l_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let h2 = h_slice[i]; let l2 = l_slice[i]; let c2 = c_slice[i];
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let lower_shadow1 = o0.min(c0) - l0;
            let long_body1 = body1 > range1 * 0.6;
            let long_lower_shadow1 = lower_shadow1 > body1;
            
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            let opens_in_body1 = o1 <= o0 && o1 >= c0;
            let closes_above_low1 = c1 > l0;
            
            let day3_bearish = c2 <= o2;
            let body3 = (c2 - o2).abs(); let range3 = h2 - l2;
            let small_body3 = body3 < range3 * 0.3;
            let contained_in_day2 = o2 <= h1 && c2 >= l1;
            
            result[i] = if day1_bearish && long_body1 && long_lower_shadow1 &&
                           small_body2 && opens_in_body1 && closes_above_low1 &&
                           day3_bearish && small_body3 && contained_in_day2 { 100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0);
            let l0 = l_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let o1 = o_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0);
            let l1 = l_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let h2 = h_vals.get(i).unwrap_or(0.0);
            let l2 = l_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let day1_bearish = c0 <= o0;
            let body1 = (c0 - o0).abs(); let range1 = h0 - l0;
            let lower_shadow1 = o0.min(c0) - l0;
            let long_body1 = body1 > range1 * 0.6;
            let long_lower_shadow1 = lower_shadow1 > body1;
            
            let body2 = (c1 - o1).abs(); let range2 = h1 - l1;
            let small_body2 = body2 < range2 * 0.3;
            let opens_in_body1 = o1 <= o0 && o1 >= c0;
            let closes_above_low1 = c1 > l0;
            
            let day3_bearish = c2 <= o2;
            let body3 = (c2 - o2).abs(); let range3 = h2 - l2;
            let small_body3 = body3 < range3 * 0.3;
            let contained_in_day2 = o2 <= h1 && c2 >= l1;
            
            result[i] = if day1_bearish && long_body1 && long_lower_shadow1 &&
                           small_body2 && opens_in_body1 && closes_above_low1 &&
                           day3_bearish && small_body3 && contained_in_day2 { 100 } else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}
// CDLADVANCEBLOCK - 前进阻挡 (零拷贝 + tight loop 优化)
#[pyfunction]
pub fn cdladvanceblock(open: PySeries, high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let c: Series = close.into();
    
    let o_vals = o.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Open: {}", e)))?;
    let h_vals = h.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("High: {}", e)))?;
    let c_vals = c.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Close: {}", e)))?;
    
    let len = o_vals.len();
    let mut result = vec![0i32; len];
    
    if len < 3 {
        return Ok(PySeries(Series::new(o.name().clone(), result)));
    }
    
    if let (Ok(o_slice), Ok(h_slice), Ok(c_slice)) = (o_vals.cont_slice(), h_vals.cont_slice(), c_vals.cont_slice()) {
        for i in 2..len {
            let o0 = o_slice[i-2]; let h0 = h_slice[i-2]; let c0 = c_slice[i-2];
            let o1 = o_slice[i-1]; let h1 = h_slice[i-1]; let c1 = c_slice[i-1];
            let o2 = o_slice[i]; let h2 = h_slice[i]; let c2 = c_slice[i];
            
            let body1 = c0 - o0; let body2 = c1 - o1; let body3 = c2 - o2;
            
            let three_white_soldiers = body1 > 0.0 && body2 > 0.0 && body3 > 0.0;
            let ascending_closes = c0 < c1 && c1 < c2;
            let opens_within_bodies = o1 > o0 && o1 < c0 && o2 > o1 && o2 < c1;
            let decreasing_highs = h1 >= h0 && h2 <= h1;
            let decreasing_bodies = body3 <= body2 && body2 <= body1;
            
            result[i] = if three_white_soldiers && ascending_closes && opens_within_bodies && 
                           decreasing_highs && decreasing_bodies { -100 } else { 0 };
        }
    } else {
        for i in 2..len {
            let o0 = o_vals.get(i-2).unwrap_or(0.0); let h0 = h_vals.get(i-2).unwrap_or(0.0); let c0 = c_vals.get(i-2).unwrap_or(0.0);
            let o1 = o_vals.get(i-1).unwrap_or(0.0); let h1 = h_vals.get(i-1).unwrap_or(0.0); let c1 = c_vals.get(i-1).unwrap_or(0.0);
            let o2 = o_vals.get(i).unwrap_or(0.0); let h2 = h_vals.get(i).unwrap_or(0.0); let c2 = c_vals.get(i).unwrap_or(0.0);
            
            let body1 = c0 - o0; let body2 = c1 - o1; let body3 = c2 - o2;
            
            let three_white_soldiers = body1 > 0.0 && body2 > 0.0 && body3 > 0.0;
            let ascending_closes = c0 < c1 && c1 < c2;
            let opens_within_bodies = o1 > o0 && o1 < c0 && o2 > o1 && o2 < c1;
            let decreasing_highs = h1 >= h0 && h2 <= h1;
            let decreasing_bodies = body3 <= body2 && body2 <= body1;
            
            result[i] = if three_white_soldiers && ascending_closes && opens_within_bodies && 
                           decreasing_highs && decreasing_bodies { -100 } else { 0 };
        }
    }
    
    Ok(PySeries(Series::new(o.name().clone(), result)))
}

// CDLTRISTAR - 三星
#[pyfunction]
pub fn cdltristar(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        // 三星：三根十字星蜡烛
        let (body1, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (body2, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        let (body3, _, _, range3) = candle_metrics(open_vals[2], high_vals[2], low_vals[2], close_vals[2]);
        
        let is_doji1 = is_doji_body(body1, range1);
        let is_doji2 = is_doji_body(body2, range2);
        let is_doji3 = is_doji_body(body3, range3);
        
        if is_doji1 && is_doji2 && is_doji3 {
            // 判断趋势方向决定信号
            if i >= 5 {
                // 在上升趋势中：看跌信号
                let uptrend = close.get(i-5).unwrap_or(0.0) < close.get(i-4).unwrap_or(0.0) && 
                             close.get(i-4).unwrap_or(0.0) < close.get(i-3).unwrap_or(0.0) && 
                             close.get(i-3).unwrap_or(0.0) < close_vals[0];
                             
                // 在下降趋势中：看涨信号  
                let downtrend = close.get(i-5).unwrap_or(0.0) > close.get(i-4).unwrap_or(0.0) && 
                               close.get(i-4).unwrap_or(0.0) > close.get(i-3).unwrap_or(0.0) && 
                               close.get(i-3).unwrap_or(0.0) > close_vals[0];
                
                if uptrend {
                    result[i] = Some(-100);
                } else if downtrend {
                    result[i] = Some(100);
                } else {
                    result[i] = Some(0);
                }
            } else {
                result[i] = Some(0);
            }
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLABANDONEDBABY - 弃婴
#[pyfunction]
pub fn cdlabandonedbaby(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let (body1, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (body2, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        let (body3, _, _, range3) = candle_metrics(open_vals[2], high_vals[2], low_vals[2], close_vals[2]);
        
        let is_doji_middle = is_doji_body(body2, range2);
        let long_body1 = is_long_body(body1, range1);
        let long_body3 = is_long_body(body3, range3);
        
        // 弃婴（看涨）：第一根长阴线，第二根十字星有向下跳空，第三根长阳线有向上跳空
        let bullish_abandoned_baby = 
            body1 < 0.0 && long_body1 && // 第一根长阴线
            is_doji_middle && // 第二根十字星
            high_vals[1] < low_vals[0] && // 第二根与第一根有向下跳空
            body3 > 0.0 && long_body3 && // 第三根长阳线
            low_vals[2] > high_vals[1]; // 第三根与第二根有向上跳空
            
        // 弃婴（看跌）：第一根长阳线，第二根十字星有向上跳空，第三根长阴线有向下跳空
        let bearish_abandoned_baby = 
            body1 > 0.0 && long_body1 && // 第一根长阳线
            is_doji_middle && // 第二根十字星
            low_vals[1] > high_vals[0] && // 第二根与第一根有向上跳空
            body3 < 0.0 && long_body3 && // 第三根长阴线
            high_vals[2] < low_vals[1]; // 第三根与第二根有向下跳空
        
        if bullish_abandoned_baby {
            result[i] = Some(100);
        } else if bearish_abandoned_baby {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLDOJISTAR - 十字星
#[pyfunction]
pub fn cdldojistar(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let (body1, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (body2, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        
        let long_body1 = is_long_body(body1, range1);
        let is_doji2 = is_doji_body(body2, range2);
        
        // 十字星（看涨）：强阴线后出现十字星，有向下跳空
        let bullish_doji_star = 
            body1 < 0.0 && long_body1 && is_doji2 && 
            high_vals[1] < low_vals[0];
            
        // 十字星（看跌）：强阳线后出现十字星，有向上跳空
        let bearish_doji_star = 
            body1 > 0.0 && long_body1 && is_doji2 && 
            low_vals[1] > high_vals[0];
        
        if bullish_doji_star {
            result[i] = Some(100);
        } else if bearish_doji_star {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSPINNINGTOP - 纺锤
#[pyfunction]
pub fn cdlspinningtop(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 纺锤：小实体，上下影线都较长
        let small_body = is_short_body(body, range);
        let long_upper_shadow = upper_shadow > body.abs() * 1.5;
        let long_lower_shadow = lower_shadow > body.abs() * 1.5;
        
        if small_body && long_upper_shadow && long_lower_shadow {
            result[i] = Some(100); // 不确定性信号，通常表示趋势可能反转
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSTALLEDPATTERN - 停顿形态
#[pyfunction]
pub fn cdlstalledpattern(open: PySeries, high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        
        // 停顿形态：三根白兵的变体，第三根实体明显缩小且有长上影线
        let three_white = body1 > 0.0 && body2 > 0.0 && body3 > 0.0;
        let ascending_closes = close_vals[0] < close_vals[1] && close_vals[1] < close_vals[2];
        let opens_within_bodies = open_vals[1] > open_vals[0] && open_vals[1] < close_vals[0] &&
                                  open_vals[2] > open_vals[1] && open_vals[2] < close_vals[1];
        let third_small_body = body3 < body2 * 0.5;
        let third_long_upper_shadow = (high_vals[2] - close_vals[2]) > body3 * 2.0;
        
        if three_white && ascending_closes && opens_within_bodies && 
           third_small_body && third_long_upper_shadow {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLTHRUSTING - 插入
#[pyfunction]
pub fn cdlthrusting(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        let (_, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (_, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        
        let long_body1 = is_long_body(body1, range1);
        let long_body2 = is_long_body(body2, range2);
        
        // 插入形态：第一根长阴线，第二根长阳线开盘低于第一根收盘但收盘在第一根实体中部以下
        let thrusting_pattern = 
            body1 < 0.0 && long_body1 && // 第一根长阴线
            body2 > 0.0 && long_body2 && // 第二根长阳线
            open_vals[1] < close_vals[0] && // 第二根开盘低于第一根收盘
            close_vals[1] > close_vals[0] && // 第二根收盘高于第一根收盘
            close_vals[1] < (open_vals[0] + close_vals[0]) / 2.0; // 但收盘在第一根实体中部以下
        
        if thrusting_pattern {
            result[i] = Some(-100); // 看跌信号，反弹失败
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLUPSIDEGAP2CROWS - 向上跳空的两只乌鸦
#[pyfunction]
pub fn cdlupsidegap2crows(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        
        // 向上跳空的两只乌鸦：第一根长阳线，第二根小阴线有向上跳空，第三根阴线吞没第二根
        let upside_gap_two_crows = 
            body1 > 0.0 && // 第一根阳线
            body2 < 0.0 && // 第二根小阴线
            low_vals[1] > high_vals[0] && // 第二根与第一根有向上跳空
            body3 < 0.0 && // 第三根阴线
            open_vals[2] > open_vals[1] && // 第三根开盘高于第二根开盘
            close_vals[2] < close_vals[1] && // 第三根收盘低于第二根收盘
            close_vals[2] > close_vals[0]; // 但第三根收盘仍高于第一根收盘
            
        if upside_gap_two_crows {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLGAPSIDESIDEWHITE - 向上/向下跳空并列阳线
#[pyfunction]
pub fn cdlgapsidesidewhite(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        // 向上跳空并列阳线：两根阳线，第二根有向上跳空，且长度相似
        let upward_gap_side_by_side = 
            body1 > 0.0 && body2 > 0.0 && // 两根阳线
            low_vals[1] > high_vals[0] && // 向上跳空
            (body1 - body2).abs() < body1 * 0.2; // 长度相似
            
        // 向下跳空并列阳线：在下降趋势中的两根阳线，第二根有向下跳空
        let downward_gap_side_by_side = 
            body1 > 0.0 && body2 > 0.0 && // 两根阳线
            high_vals[1] < low_vals[0] && // 向下跳空
            (body1 - body2).abs() < body1 * 0.2; // 长度相似
        
        if upward_gap_side_by_side {
            result[i] = Some(100); // 看涨继续信号
        } else if downward_gap_side_by_side {
            result[i] = Some(-100); // 看跌继续信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLIDENTICAL3CROWS - 相同三只乌鸦
#[pyfunction]
pub fn cdlidentical3crows(open: PySeries, _high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        
        // 相同三只乌鸦：三根相似的阴线，每根收盘都创新低，开盘接近前一根收盘
        let three_black_crows = 
            body1 < 0.0 && body2 < 0.0 && body3 < 0.0 && // 三根阴线
            close_vals[1] < close_vals[0] && close_vals[2] < close_vals[1] && // 收盘价递减
            (open_vals[1] - close_vals[0]).abs() < body1.abs() * 0.1 && // 第二根开盘接近第一根收盘
            (open_vals[2] - close_vals[1]).abs() < body2.abs() * 0.1 && // 第三根开盘接近第二根收盘
            (body1.abs() - body2.abs()).abs() < body1.abs() * 0.2 && // 实体大小相似
            (body2.abs() - body3.abs()).abs() < body2.abs() * 0.2;
        
        if three_black_crows {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLINNECK - 颈内线
#[pyfunction]
pub fn cdlinneck(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        let (_, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (_, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        
        let long_body1 = is_long_body(body1, range1);
        let long_body2 = is_long_body(body2, range2);
        
        // 颈内线：第一根长阴线，第二根长阳线开盘较低，收盘接近但略低于第一根收盘
        let in_neck = 
            body1 < 0.0 && long_body1 && // 第一根长阴线
            body2 > 0.0 && long_body2 && // 第二根长阳线
            open_vals[1] < low_vals[0] && // 第二根开盘低于第一根最低价
            close_vals[1] <= close_vals[0] && // 第二根收盘不高于第一根收盘
            close_vals[1] >= close_vals[0] - range1 * 0.05; // 但收盘接近第一根收盘
        
        if in_neck {
            result[i] = Some(-100); // 看跌信号，反弹失败
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLONNECK - 颈上线
#[pyfunction]
pub fn cdlonneck(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        let (_, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (_, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        
        let long_body1 = is_long_body(body1, range1);
        let long_body2 = is_long_body(body2, range2);
        
        // 颈上线：第一根长阴线，第二根长阳线开盘较低，收盘等于或略高于第一根收盘
        let on_neck = 
            body1 < 0.0 && long_body1 && // 第一根长阴线
            body2 > 0.0 && long_body2 && // 第二根长阳线
            open_vals[1] < low_vals[0] && // 第二根开盘低于第一根最低价
            close_vals[1] >= close_vals[0] && // 第二根收盘高于或等于第一根收盘
            close_vals[1] <= close_vals[0] + range1 * 0.05; // 但收盘接近第一根收盘
        
        if on_neck {
            result[i] = Some(-100); // 看跌信号，反弹仍然无力
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLRISEFALL3METHODS - 上升/下降三法
#[pyfunction]
pub fn cdlrisefall3methods(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let open_vals: Vec<f64> = (0..5).map(|j| open.get(i-4+j).unwrap_or(0.0)).collect();
        let high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = (0..5).map(|j| low.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        let body4 = close_vals[3] - open_vals[3];
        let body5 = close_vals[4] - open_vals[4];
        
        // 上升三法：长阳线 + 三根小阴线（在长阳线范围内）+ 长阳线
        let rising_three_methods = 
            body1 > 0.0 && body1 > (high_vals[0] - low_vals[0]) * 0.6 && // 第一根长阳线
            body2 < 0.0 && body3 < 0.0 && body4 < 0.0 && // 中间三根阴线
            body2.abs() < body1 * 0.5 && body3.abs() < body1 * 0.5 && body4.abs() < body1 * 0.5 && // 中间三根较小
            high_vals[1] < high_vals[0] && high_vals[2] < high_vals[0] && high_vals[3] < high_vals[0] && // 高点在第一根范围内
            low_vals[1] > low_vals[0] && low_vals[2] > low_vals[0] && low_vals[3] > low_vals[0] && // 低点在第一根范围内
            body5 > 0.0 && close_vals[4] > close_vals[0]; // 第五根长阳线突破
            
        // 下降三法：长阴线 + 三根小阳线（在长阴线范围内）+ 长阴线
        let falling_three_methods = 
            body1 < 0.0 && body1.abs() > (high_vals[0] - low_vals[0]) * 0.6 && // 第一根长阴线
            body2 > 0.0 && body3 > 0.0 && body4 > 0.0 && // 中间三根阳线
            body2 < body1.abs() * 0.5 && body3 < body1.abs() * 0.5 && body4 < body1.abs() * 0.5 && // 中间三根较小
            high_vals[1] < high_vals[0] && high_vals[2] < high_vals[0] && high_vals[3] < high_vals[0] && // 高点在第一根范围内
            low_vals[1] > low_vals[0] && low_vals[2] > low_vals[0] && low_vals[3] > low_vals[0] && // 低点在第一根范围内
            body5 < 0.0 && close_vals[4] < close_vals[0]; // 第五根长阴线突破
        
        if rising_three_methods {
            result[i] = Some(100); // 看涨信号
        } else if falling_three_methods {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLKICKING - 踢脚线
#[pyfunction]
pub fn cdlkicking(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        // 检查是否为光头光脚（Marubozu）
        let is_marubozu1 = (high_vals[0] == f64::max(open_vals[0], close_vals[0])) && 
                          (low_vals[0] == f64::min(open_vals[0], close_vals[0]));
        let is_marubozu2 = (high_vals[1] == f64::max(open_vals[1], close_vals[1])) && 
                          (low_vals[1] == f64::min(open_vals[1], close_vals[1]));
        
        // 踢脚线（看涨）：第一根黑色光头光脚，第二根白色光头光脚，有向上跳空
        let bullish_kicking = 
            close_vals[0] < open_vals[0] && is_marubozu1 && // 第一根阴线光头光脚
            close_vals[1] > open_vals[1] && is_marubozu2 && // 第二根阳线光头光脚
            low_vals[1] > high_vals[0]; // 向上跳空
            
        // 踢脚线（看跌）：第一根白色光头光脚，第二根黑色光头光脚，有向下跳空
        let bearish_kicking = 
            close_vals[0] > open_vals[0] && is_marubozu1 && // 第一根阳线光头光脚
            close_vals[1] < open_vals[1] && is_marubozu2 && // 第二根阴线光头光脚
            high_vals[1] < low_vals[0]; // 向下跳空
        
        if bullish_kicking {
            result[i] = Some(100);
        } else if bearish_kicking {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLKICKINGBYLENGTH - 按长度踢脚线
#[pyfunction]
pub fn cdlkickingbylength(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let (_, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (_, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        
        // 检查是否为光头光脚且长实体
        let is_long_marubozu1 = (high_vals[0] == f64::max(open_vals[0], close_vals[0])) && 
                               (low_vals[0] == f64::min(open_vals[0], close_vals[0])) &&
                               is_long_body(close_vals[0] - open_vals[0], range1);
        let is_long_marubozu2 = (high_vals[1] == f64::max(open_vals[1], close_vals[1])) && 
                               (low_vals[1] == f64::min(open_vals[1], close_vals[1])) &&
                               is_long_body(close_vals[1] - open_vals[1], range2);
        
        // 按长度踢脚线：与普通踢脚线类似，但要求两根蜡烛都是长实体
        let bullish_kicking_by_length = 
            close_vals[0] < open_vals[0] && is_long_marubozu1 && // 第一根长阴线光头光脚
            close_vals[1] > open_vals[1] && is_long_marubozu2 && // 第二根长阳线光头光脚
            low_vals[1] > high_vals[0]; // 向上跳空
            
        let bearish_kicking_by_length = 
            close_vals[0] > open_vals[0] && is_long_marubozu1 && // 第一根长阳线光头光脚
            close_vals[1] < open_vals[1] && is_long_marubozu2 && // 第二根长阴线光头光脚
            high_vals[1] < low_vals[0]; // 向下跳空
        
        if bullish_kicking_by_length {
            result[i] = Some(100);
        } else if bearish_kicking_by_length {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLLONGLEGGEDDOJI - 长腿十字星
#[pyfunction]
pub fn cdllongleggeddoji(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 长腿十字星：十字星且上下影线都很长
        let is_doji = is_doji_body(body, range);
        let long_upper_shadow = upper_shadow > range * 0.3;
        let long_lower_shadow = lower_shadow > range * 0.3;
        
        if is_doji && long_upper_shadow && long_lower_shadow {
            result[i] = Some(100); // 反转信号，不确定方向
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLMARUBOZU - 光头光脚
#[pyfunction]
pub fn cdlmarubozu(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let body = close_val - open_val;
        let tolerance = (high_val - low_val) * 0.01; // 1%的容差
        
        // 白色光头光脚（阳线）：开盘价等于最低价，收盘价等于最高价
        let white_marubozu = body > 0.0 &&
                            (open_val - low_val).abs() <= tolerance &&
                            (close_val - high_val).abs() <= tolerance;
                            
        // 黑色光头光脚（阴线）：开盘价等于最高价，收盘价等于最低价
        let black_marubozu = body < 0.0 &&
                            (open_val - high_val).abs() <= tolerance &&
                            (close_val - low_val).abs() <= tolerance;
        
        if white_marubozu {
            result[i] = Some(100); // 看涨信号
        } else if black_marubozu {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLMATCHINGLOW - 相同低点
#[pyfunction]
pub fn cdlmatchinglow(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        let (_, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let long_body1 = is_long_body(body1, range1);
        
        let tolerance = (high_vals[0] - low_vals[0]) * 0.02; // 2%的容差
        
        // 相同低点：第一根黑色长蜡烛，第二根短蜡烛，两根蜡烛有相同的低点
        let matching_low = 
            body1 < 0.0 && long_body1 && // 第一根长阴线
            (low_vals[1] - low_vals[0]).abs() <= tolerance && // 相同的低点
            body2.abs() < body1.abs() * 0.5; // 第二根为短蜡烛
        
        if matching_low {
            result[i] = Some(100); // 看涨信号，支撑确认
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLRICKSHAWMAN - 人力车夫
#[pyfunction]
pub fn cdlrickshawman(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 人力车夫：十字星且上下影线都很长，实体在整个K线的中间部分
        let is_doji = is_doji_body(body, range);
        let long_upper_shadow = upper_shadow > range * 0.25;
        let long_lower_shadow = lower_shadow > range * 0.25;
        let centered_body = upper_shadow > range * 0.15 && lower_shadow > range * 0.15;
        
        if is_doji && long_upper_shadow && long_lower_shadow && centered_body {
            result[i] = Some(100); // 不确定性信号，趋势可能反转
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSEPARATINGLINES - 分离线
#[pyfunction]
pub fn cdlseparatinglines(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        let tolerance = (high_vals[0] - low_vals[0]) * 0.02; // 2%的容差
        
        // 看涨分离线：第一根阴线，第二根阳线，两根蜡烛有相同的开盘价
        let bullish_separating_lines = 
            body1 < 0.0 && body2 > 0.0 && // 第一根阴线，第二根阳线
            (open_vals[1] - open_vals[0]).abs() <= tolerance; // 相同的开盘价
            
        // 看跌分离线：第一根阳线，第二根阴线，两根蜡烛有相同的开盘价
        let bearish_separating_lines = 
            body1 > 0.0 && body2 < 0.0 && // 第一根阳线，第二根阴线
            (open_vals[1] - open_vals[0]).abs() <= tolerance; // 相同的开盘价
        
        if bullish_separating_lines {
            result[i] = Some(100); // 看涨信号
        } else if bearish_separating_lines {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSHORTLINE - 短线
#[pyfunction]
pub fn cdlshortline(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, _, _, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 短线：很短的实体
        let is_short = is_short_body(body, range);
        
        if is_short {
            result[i] = Some(100); // 不确定性信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLTASUKIGAP - 上影线缺口
#[pyfunction]
pub fn cdltasukigap(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        
        // 向上跳空上影线缺口
        let upward_tasuki_gap = 
            body1 > 0.0 && body2 > 0.0 && // 前两根阳线
            low_vals[1] > high_vals[0] && // 第二根与第一根有向上跳空
            body3 < 0.0 && // 第三根阴线
            open_vals[2] > close_vals[1] && // 第三根开盘高于第二根收盘
            close_vals[2] < high_vals[0] && close_vals[2] > low_vals[1]; // 第三根收盘在跳空区间内
            
        // 向下跳空上影线缺口
        let downward_tasuki_gap = 
            body1 < 0.0 && body2 < 0.0 && // 前两根阴线
            high_vals[1] < low_vals[0] && // 第二根与第一根有向下跳空
            body3 > 0.0 && // 第三根阳线
            open_vals[2] < close_vals[1] && // 第三根开盘低于第二根收盘
            close_vals[2] > low_vals[0] && close_vals[2] < high_vals[1]; // 第三根收盘在跳空区间内
        
        if upward_tasuki_gap {
            result[i] = Some(100); // 看涨持续信号
        } else if downward_tasuki_gap {
            result[i] = Some(-100); // 看跌持续信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLUNIQUE3RIVER - 独特三江河
#[pyfunction]
pub fn cdlunique3river(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        
        let (_, _, _, range1) = candle_metrics(open_vals[0], high_vals[0], low_vals[0], close_vals[0]);
        let (_, _, _, range2) = candle_metrics(open_vals[1], high_vals[1], low_vals[1], close_vals[1]);
        let (_, _, _, range3) = candle_metrics(open_vals[2], high_vals[2], low_vals[2], close_vals[2]);
        
        let long_body1 = is_long_body(body1, range1);
        let is_doji2 = is_doji_body(body2, range2);
        let short_body3 = is_short_body(body3, range3);
        
        // 独特三江河：第一根长阴线，第二根十字星，第三根小阳线收盘低于第二根开盘
        let unique_three_river = 
            body1 < 0.0 && long_body1 && // 第一根长阴线
            is_doji2 && // 第二根十字星
            body3 > 0.0 && short_body3 && // 第三根小阳线
            close_vals[2] < open_vals[1] && // 第三根收盘低于第二根开盘
            low_vals[2] < low_vals[1] && low_vals[1] <= low_vals[0]; // 创新低
        
        if unique_three_river {
            result[i] = Some(100); // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLXSIDEGAP3METHODS - 向侧跳空三法
#[pyfunction]
pub fn cdlxsidegap3methods(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let open_vals: Vec<f64> = (0..5).map(|j| open.get(i-4+j).unwrap_or(0.0)).collect();
        let high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = (0..5).map(|j| low.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        let body4 = close_vals[3] - open_vals[3];
        let body5 = close_vals[4] - open_vals[4];
        
        // 上升向侧跳空三法：第一根阳线，第二根阳线跳空，中间三根小蜡烛，第五根阳线填补跳空
        let upside_gap_three_methods = 
            body1 > 0.0 && body2 > 0.0 && // 前两根阳线
            low_vals[1] > high_vals[0] && // 向上跳空
            body3.abs() < body1 * 0.5 && body4.abs() < body1 * 0.5 && // 中间蜡烛较小
            high_vals[2] < low_vals[1] && high_vals[3] < low_vals[1] && // 中间蜡烛在跳空下方
            body5 > 0.0 && // 第五根阳线
            close_vals[4] > high_vals[0] && open_vals[4] < low_vals[1]; // 填补跳空
            
        // 下降向侧跳空三法：第一根阴线，第二根阴线跳空，中间三根小蜡烛，第五根阴线填补跳空
        let downside_gap_three_methods = 
            body1 < 0.0 && body2 < 0.0 && // 前两根阴线
            high_vals[1] < low_vals[0] && // 向下跳空
            body3.abs() < body1.abs() * 0.5 && body4.abs() < body1.abs() * 0.5 && // 中间蜡烛较小
            low_vals[2] > high_vals[1] && low_vals[3] > high_vals[1] && // 中间蜡烛在跳空上方
            body5 < 0.0 && // 第五根阴线
            close_vals[4] < low_vals[0] && open_vals[4] > high_vals[1]; // 填补跳空
        
        if upside_gap_three_methods {
            result[i] = Some(100); // 看涨信号
        } else if downside_gap_three_methods {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLBELTHOLD - 抱线
#[pyfunction]
pub fn cdlbelthold(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        let long_body = is_long_body(body, range);
        
        // 白色抱线：长阳线，开盘价等于或接近最低价
        let white_belt_hold = body > 0.0 && long_body && 
                             lower_shadow < body * 0.1;
                             
        // 黑色抱线：长阴线，开盘价等于或接近最高价
        let black_belt_hold = body < 0.0 && long_body && 
                             upper_shadow < body.abs() * 0.1;
        
        if white_belt_hold {
            result[i] = Some(100);
        } else if black_belt_hold {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLBREAKAWAY - 脱离
#[pyfunction]
pub fn cdlbreakaway(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let open_vals: Vec<f64> = (0..5).map(|j| open.get(i-4+j).unwrap_or(0.0)).collect();
        let high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = (0..5).map(|j| low.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        let body4 = close_vals[3] - open_vals[3];
        let body5 = close_vals[4] - open_vals[4];
        
        // 看涨脱离：第一根长阴线，第二根阴线跳空，中间小实体，第五根阳线收盘在跳空内
        let bullish_breakaway = 
            body1 < 0.0 && // 第一根阴线
            body2 < 0.0 && // 第二根阴线
            high_vals[1] < low_vals[0] && // 向下跳空
            body3.abs() < body1.abs() * 0.5 && body4.abs() < body1.abs() * 0.5 && // 中间小实体
            body5 > 0.0 && // 第五根阳线
            close_vals[4] > low_vals[0] && close_vals[4] < high_vals[1]; // 收盘在跳空内
            
        // 看跌脱离：第一根长阳线，第二根阳线跳空，中间小实体，第五根阴线收盘在跳空内
        let bearish_breakaway = 
            body1 > 0.0 && // 第一根阳线
            body2 > 0.0 && // 第二根阳线
            low_vals[1] > high_vals[0] && // 向上跳空
            body3.abs() < body1 * 0.5 && body4.abs() < body1 * 0.5 && // 中间小实体
            body5 < 0.0 && // 第五根阴线
            close_vals[4] < high_vals[0] && close_vals[4] > low_vals[1]; // 收盘在跳空内
        
        if bullish_breakaway {
            result[i] = Some(100);
        } else if bearish_breakaway {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLCLOSINGMARUBOZU - 收盘光头光脚
#[pyfunction]
pub fn cdlclosingmarubozu(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let body = close_val - open_val;
        let (_, _, _, range) = candle_metrics(open_val, high_val, low_val, close_val);
        let tolerance = range * 0.01;
        
        // 收盘白色光头光脚：阳线，收盘价等于最高价
        let closing_white_marubozu = body > 0.0 &&
                                    (close_val - high_val).abs() <= tolerance;
                                    
        // 收盘黑色光头光脚：阴线，收盘价等于最低价
        let closing_black_marubozu = body < 0.0 &&
                                    (close_val - low_val).abs() <= tolerance;
        
        if closing_white_marubozu {
            result[i] = Some(100);
        } else if closing_black_marubozu {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLCONCEALBABYSWALL - 藏婴吞没
#[pyfunction]
pub fn cdlconcealbabyswall(open: PySeries, high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let open_vals: Vec<f64> = (0..5).map(|j| open.get(i-4+j).unwrap_or(0.0)).collect();
        let high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        let body4 = close_vals[3] - open_vals[3];
        let body5 = close_vals[4] - open_vals[4];
        
        // 藏婴吞没：四根阴线，第五根阳线开盘高于第四根最高价，收盘高于第一根开盘价
        let conceal_baby_swallow = 
            body1 < 0.0 && body2 < 0.0 && body3 < 0.0 && body4 < 0.0 && // 前四根阴线
            body5 > 0.0 && // 第五根阳线
            open_vals[4] > high_vals[3] && // 第五根开盘高于第四根最高价
            close_vals[4] > open_vals[0]; // 第五根收盘高于第一根开盘价
        
        if conceal_baby_swallow {
            result[i] = Some(100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLCOUNTERATTACK - 反击线
#[pyfunction]
pub fn cdlcounterattack(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let tolerance = (high_vals[0] - low_vals[0]) * 0.02;
        
        // 看涨反击：第一根阴线，第二根阳线，收盘价相等
        let bullish_counterattack = 
            body1 < 0.0 && body2 > 0.0 &&
            (close_vals[1] - close_vals[0]).abs() <= tolerance;
            
        // 看跌反击：第一根阳线，第二根阴线，收盘价相等
        let bearish_counterattack = 
            body1 > 0.0 && body2 < 0.0 &&
            (close_vals[1] - close_vals[0]).abs() <= tolerance;
        
        if bullish_counterattack {
            result[i] = Some(100);
        } else if bearish_counterattack {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLDRAGONFLYDOJI - 蜻蜓十字
#[pyfunction]
pub fn cdldragonflydoji(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 蜻蜓十字：十字星，长下影线，无或很短的上影线
        let is_doji = is_doji_body(body, range);
        let long_lower_shadow = lower_shadow > range * 0.5;
        let no_upper_shadow = upper_shadow < range * 0.1;
        
        if is_doji && long_lower_shadow && no_upper_shadow {
            result[i] = Some(100); // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLGRAVESTONEDOJI - 墓碑十字
#[pyfunction]
pub fn cdlgravestonedoji(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 墓碑十字：十字星，长上影线，无或很短的下影线
        let is_doji = is_doji_body(body, range);
        let long_upper_shadow = upper_shadow > range * 0.5;
        let no_lower_shadow = lower_shadow < range * 0.1;
        
        if is_doji && long_upper_shadow && no_lower_shadow {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHIGHWAVE - 高浪
#[pyfunction]
pub fn cdlhighwave(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 高浪：小实体，极长的上下影线
        let small_body = is_short_body(body, range);
        let very_long_upper = upper_shadow > range * 0.35;
        let very_long_lower = lower_shadow > range * 0.35;
        
        if small_body && very_long_upper && very_long_lower {
            result[i] = Some(100); // 不确定性信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHIKKAKE - 陷阱
#[pyfunction]
pub fn cdlhikkake(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open_series = open.as_ref();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = high.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        // 陷阱：第二根K线的高点和低点都在第一根K线范围内，第三根突破
        let inside_bar = high_vals[1] < high_vals[0] && low_vals[1] > low_vals[0];
        let bullish_breakout = close_vals[2] > high_vals[0];
        let bearish_breakout = close_vals[2] < low_vals[0];
        
        if inside_bar && bullish_breakout {
            result[i] = Some(100);
        } else if inside_bar && bearish_breakout {
            result[i] = Some(-100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open_series.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHIKKAKEMOD - 修正陷阱
#[pyfunction]
pub fn cdlhikkakemod(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let low_vals: Vec<f64> = (0..5).map(|j| low.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        // 修正陷阱：陷阱模式的变体，需要更多确认
        let inside_bar = high_vals[1] < high_vals[0] && low_vals[1] > low_vals[0];
        let initial_breakout = close_vals[2] > high_vals[0] || close_vals[2] < low_vals[0];
        let confirmation = (close_vals[4] > high_vals[0] && close_vals[2] > high_vals[0]) ||
                          (close_vals[4] < low_vals[0] && close_vals[2] < low_vals[0]);
        
        if inside_bar && initial_breakout && confirmation {
            if close_vals[4] > high_vals[0] {
                result[i] = Some(100);
            } else if close_vals[4] < low_vals[0] {
                result[i] = Some(-100);
            }
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHOMINGPIGEON - 信鸽
#[pyfunction]
pub fn cdlhomingpigeon(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 1..len {
        let open_vals = [
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let _high_vals = [
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let _low_vals = [
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        
        // 信鸽：第一根长阴线，第二根短阴线完全在第一根实体内
        let homing_pigeon = 
            body1 < 0.0 && body2 < 0.0 && // 两根阴线
            body2.abs() < body1.abs() * 0.5 && // 第二根较短
            open_vals[1] <= open_vals[0] && close_vals[1] >= close_vals[0]; // 第二根在第一根实体内
        
        if homing_pigeon {
            result[i] = Some(100); // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLINVERTEDHAMMER - 倒锤头
#[pyfunction]
pub fn cdlinvertedhammer(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 倒锤头：小实体在底部，长上影线，无或短下影线
        let small_body = is_short_body(body, range);
        let long_upper_shadow = upper_shadow > range * 0.6;
        let short_lower_shadow = lower_shadow < upper_shadow * 0.3;
        
        if small_body && long_upper_shadow && short_lower_shadow {
            result[i] = Some(100); // 看涨信号（在下降趋势中）
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLLADDERBOTTOM - 梯底
#[pyfunction]
pub fn cdlladderbottom(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let open_vals: Vec<f64> = (0..5).map(|j| open.get(i-4+j).unwrap_or(0.0)).collect();
        let _high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let _low_vals: Vec<f64> = (0..5).map(|j| low.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        let body4 = close_vals[3] - open_vals[3];
        let body5 = close_vals[4] - open_vals[4];
        
        // 梯底：三根阴线递减，第四根小阴线，第五根阳线开盘高于第四根收盘
        let ladder_bottom = 
            body1 < 0.0 && body2 < 0.0 && body3 < 0.0 && // 前三根阴线
            close_vals[1] < close_vals[0] && close_vals[2] < close_vals[1] && // 递减
            body4 < 0.0 && body4.abs() < body3.abs() * 0.5 && // 第四根小阴线
            body5 > 0.0 && open_vals[4] > close_vals[3]; // 第五根阳线跳空开盘
        
        if ladder_bottom {
            result[i] = Some(100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLLONGLINE - 长线
#[pyfunction]
pub fn cdllongline(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, _, _, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 长线：长实体
        let is_long = is_long_body(body, range);
        
        if is_long {
            if body > 0.0 {
                result[i] = Some(100); // 长阳线
            } else {
                result[i] = Some(-100); // 长阴线
            }
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLMATHOLD - 铺垫
#[pyfunction]
pub fn cdlmathold(open: PySeries, high: PySeries, low: PySeries, close: PySeries, _penetration: f64) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 4..len {
        let open_vals: Vec<f64> = (0..5).map(|j| open.get(i-4+j).unwrap_or(0.0)).collect();
        let _high_vals: Vec<f64> = (0..5).map(|j| high.get(i-4+j).unwrap_or(0.0)).collect();
        let _low_vals: Vec<f64> = (0..5).map(|j| low.get(i-4+j).unwrap_or(0.0)).collect();
        let close_vals: Vec<f64> = (0..5).map(|j| close.get(i-4+j).unwrap_or(0.0)).collect();
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        let body4 = close_vals[3] - open_vals[3];
        let body5 = close_vals[4] - open_vals[4];
        
        // 铺垫：第一根长阳线，中间三根小阴线，第五根阳线创新高
        let mat_hold = 
            body1 > 0.0 && // 第一根阳线
            body2 < 0.0 && body3 < 0.0 && body4 < 0.0 && // 中间三根阴线
            body2.abs() < body1 * 0.3 && body3.abs() < body1 * 0.3 && body4.abs() < body1 * 0.3 && // 小阴线
            body5 > 0.0 && close_vals[4] > close_vals[0]; // 第五根阳线创新高
        
        if mat_hold {
            result[i] = Some(100);
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSTICKSANDWICH - 条形三明治
#[pyfunction]
pub fn cdlsticksandwich(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 2..len {
        let open_vals = [
            open.get(i-2).unwrap_or(0.0),
            open.get(i-1).unwrap_or(0.0),
            open.get(i).unwrap_or(0.0)
        ];
        let high_vals = [
            high.get(i-2).unwrap_or(0.0),
            high.get(i-1).unwrap_or(0.0),
            high.get(i).unwrap_or(0.0)
        ];
        let low_vals = [
            low.get(i-2).unwrap_or(0.0),
            low.get(i-1).unwrap_or(0.0),
            low.get(i).unwrap_or(0.0)
        ];
        let close_vals = [
            close.get(i-2).unwrap_or(0.0),
            close.get(i-1).unwrap_or(0.0),
            close.get(i).unwrap_or(0.0)
        ];
        
        let body1 = close_vals[0] - open_vals[0];
        let body2 = close_vals[1] - open_vals[1];
        let body3 = close_vals[2] - open_vals[2];
        
        let tolerance = (high_vals[0] - low_vals[0]) * 0.02;
        
        // 条形三明治：第一根和第三根阴线收盘价相等，中间一根阳线
        let stick_sandwich = 
            body1 < 0.0 && body2 > 0.0 && body3 < 0.0 && // 阴阳阴
            (close_vals[2] - close_vals[0]).abs() <= tolerance; // 收盘价相等
        
        if stick_sandwich {
            result[i] = Some(100); // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLTAKURI - 探水竿
#[pyfunction]
pub fn cdltakuri(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let open = open.as_ref().f64().unwrap();
    let high = high.as_ref().f64().unwrap();
    let low = low.as_ref().f64().unwrap();
    let close = close.as_ref().f64().unwrap();
    
    let len = open.len();
    let mut result: Vec<Option<i32>> = vec![Some(0); len];
    
    for i in 0..len {
        let open_val = open.get(i).unwrap_or(0.0);
        let high_val = high.get(i).unwrap_or(0.0);
        let low_val = low.get(i).unwrap_or(0.0);
        let close_val = close.get(i).unwrap_or(0.0);
        
        let (body, upper_shadow, lower_shadow, range) = candle_metrics(open_val, high_val, low_val, close_val);
        
        // 探水竿：小实体在顶部，很长的下影线，无上影线（类似蜻蜓十字但更严格）
        let small_body = is_short_body(body, range);
        let very_long_lower = lower_shadow > range * 0.7;
        let no_upper_shadow = upper_shadow < range * 0.05;
        
        if small_body && very_long_lower && no_upper_shadow {
            result[i] = Some(100); // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
}
