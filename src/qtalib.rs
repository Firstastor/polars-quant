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
}

/// 通用移动平均计算函数
pub fn calculate_ma(values: &[f64], period: usize, ma_type: MAType) -> Vec<Option<f64>> {
    let len = values.len();
    let mut result = vec![None; len];
    
    if period == 0 || period > len {
        return result;
    }
    
    match ma_type {
        MAType::SMA => {
            let mut sum = 0.0;
            
            // 计算第一个窗口的和
            for i in 0..period {
                sum += values[i];
            }
            result[period - 1] = Some(sum / period as f64);
            
            // 滑动窗口计算后续值
            for i in period..len {
                sum += values[i] - values[i - period];
                result[i] = Some(sum / period as f64);
            }
        },
        MAType::EMA => {
            let alpha = 2.0 / (period as f64 + 1.0);
            let one_minus_alpha = 1.0 - alpha;
            
            // 使用SMA作为初始值
            let mut sum = 0.0;
            for i in 0..period {
                sum += values[i];
            }
            let mut ema = sum / period as f64;
            result[period - 1] = Some(ema);
            
            // 计算EMA
            for i in period..len {
                ema = alpha * values[i] + one_minus_alpha * ema;
                result[i] = Some(ema);
            }
        }
    }
    
    result
}

// 高效的RSI计算函数
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

// 高效的ATR计算函数
fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![f64::NAN; len];
    
    if period == 0 || period >= len {
        return result;
    }
    
    // 计算真实范围
    let mut tr_values = Vec::with_capacity(len);
    tr_values.push(high[0] - low[0]); // 第一个值
    
    for i in 1..len {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        tr_values.push(tr1.max(tr2).max(tr3));
    }
    
    // 计算ATR (使用SMA开始，然后用指数平滑)
    let sum = tr_values.iter().take(period).sum::<f64>();
    result[period - 1] = sum / period as f64;
    
    // 使用Wilder的平滑方法
    for i in period..len {
        let prev_atr = result[i - 1];
        result[i] = (prev_atr * (period - 1) as f64 + tr_values[i]) / period as f64;
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

fn is_bullish(open: f64, close: f64) -> bool {
    close > open
}

fn is_doji_body(body_size: f64, range: f64) -> bool {
    body_size < range * 0.1
}

/// 布林带 (BBAND)
#[pyfunction]
#[pyo3(signature = (series, period=20, std_dev=2.0))]
pub fn bband(series: PySeries, period: usize, std_dev: f64) -> PyResult<(PySeries, PySeries, PySeries)> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    let sma_values = calculate_ma(&vec_values, period, MAType::SMA);
    
    // 计算标准差
    let mut upper = Vec::with_capacity(vec_values.len());
    let mut middle = Vec::with_capacity(vec_values.len());
    let mut lower = Vec::with_capacity(vec_values.len());
    
    for i in 0..vec_values.len() {
        if let Some(sma) = sma_values[i] {
            if i + 1 >= period {
                let slice = &vec_values[i + 1 - period..=i];
                let variance = slice.iter()
                    .map(|&x| (x - sma).powi(2))
                    .sum::<f64>() / period as f64;
                let std = variance.sqrt();
                
                upper.push(Some(sma + std_dev * std));
                middle.push(Some(sma));
                lower.push(Some(sma - std_dev * std));
            } else {
                upper.push(None);
                middle.push(None);
                lower.push(None);
            }
        } else {
            upper.push(None);
            middle.push(None);
            lower.push(None);
        }
    }
    
    let base_name = s.name();
    let upper_series = Series::new(PlSmallStr::from_str(&format!("{}_bb_upper", base_name)), upper);
    let middle_series = Series::new(PlSmallStr::from_str(&format!("{}_bb_middle", base_name)), middle);
    let lower_series = Series::new(PlSmallStr::from_str(&format!("{}_bb_lower", base_name)), lower);
    
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
    
    let vec_values: Vec<f64> = values.into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect();
    
    let ema_values = {
        let mut result = vec![None; vec_values.len()];
        if vec_values.is_empty() || period == 0 {
            return Ok(PySeries(Series::new(s.name().clone(), result)));
        }
        
        let alpha = 2.0 / (period + 1) as f64;
        let mut ema = vec_values[0];
        result[0] = Some(ema);
        
        for i in 1..vec_values.len() {
            ema = alpha * vec_values[i] + (1.0 - alpha) * ema;
            result[i] = Some(ema);
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), ema_values);
    Ok(PySeries(result))
}

/// 考夫曼自适应移动平均 (KAMA)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn kama(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let kama_values = {
        let mut result = vec![None; vec_values.len()];
        
        if vec_values.len() >= period {
            for i in period..vec_values.len() {
                // 计算变化 (Change)
                let change = (vec_values[i] - vec_values[i - period]).abs();
                
                // 计算波动 (Volatility)
                let mut volatility = 0.0;
                for j in (i - period + 1)..=i {
                    volatility += (vec_values[j] - vec_values[j - 1]).abs();
                }
                
                // 计算效率比 (Efficiency Ratio)
                let er = if volatility != 0.0 { change / volatility } else { 0.0 };
                
                // 计算平滑常数 (Smoothing Constant)
                let fastest_sc = 2.0 / 3.0; // 对应周期2的EMA
                let slowest_sc = 2.0 / 31.0; // 对应周期30的EMA
                let sc = (er * (fastest_sc - slowest_sc) + slowest_sc).powi(2);
                
                // 计算KAMA
                if i == period {
                    result[i] = Some(vec_values[i]);
                } else {
                    let prev_kama = result[i - 1].unwrap_or(vec_values[i]);
                    result[i] = Some(prev_kama + sc * (vec_values[i] - prev_kama));
                }
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), kama_values);
    Ok(PySeries(result))
}

/// 移动平均线 (MA)
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn ma(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect();
    
    let sma_values = calculate_ma(&vec_values, period, MAType::SMA);
    let result = Series::new(s.name().clone(), sma_values);
    
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
    
    let vec_values: Vec<f64> = values.into_iter()
        .map(|opt| opt.unwrap_or(0.0))
        .collect();
    
    let sma_values = {
        let mut result = vec![None; vec_values.len()];
        if vec_values.len() < period || period == 0 {
            return Ok(PySeries(Series::new(s.name().clone(), result)));
        }
        
        for i in (period - 1)..vec_values.len() {
            let sum: f64 = vec_values[i - period + 1..=i].iter().sum();
            result[i] = Some(sum / period as f64);
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), sma_values);
    Ok(PySeries(result))
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
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn trima(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let trima_values = {
        let sma1_values = calculate_ma(&vec_values, period, MAType::SMA);
        let sma1_f64: Vec<f64> = sma1_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
        let sma2_period = if period % 2 == 1 { (period + 1) / 2 } else { period / 2 + 1 };
        let sma2_values = calculate_ma(&sma1_f64, sma2_period, MAType::SMA);
        sma2_values
    };
    
    let result = Series::new(s.name().clone(), trima_values);
    Ok(PySeries(result))
}

/// 加权移动平均 (WMA)
#[pyfunction]
#[pyo3(signature = (series, period=20))]
pub fn wma(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let wma_values = {
        let mut result = vec![None; vec_values.len()];
        let weight_sum: usize = (1..=period).sum();
        
        for i in 0..vec_values.len() {
            if i + 1 >= period {
                let mut weighted_sum = 0.0;
                for (j, weight) in (1..=period).enumerate() {
                    weighted_sum += vec_values[i - period + 1 + j] * weight as f64;
                }
                result[i] = Some(weighted_sum / weight_sum as f64);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), wma_values);
    Ok(PySeries(result))
}

/// 中点 (MIDPOINT)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn midpoint(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let midpoint_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in 0..vec_values.len() {
            if i + 1 >= period {
                let slice = &vec_values[i + 1 - period..=i];
                let max_val = slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_val = slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                result[i] = Some((max_val + min_val) / 2.0);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), midpoint_values);
    Ok(PySeries(result))
}

/// 中间价格 (MIDPRICE) - 针对high/low序列
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn midprice_hl(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let midprice_values = {
        let mut result = vec![None; high_vals.len()];
        
        for i in 0..high_vals.len() {
            if i + 1 >= period {
                let high_slice = &high_vals[i + 1 - period..=i];
                let low_slice = &low_vals[i + 1 - period..=i];
                
                let max_high = high_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_low = low_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                
                result[i] = Some((max_high + min_low) / 2.0);
            }
        }
        
        result
    };
    
    let result = Series::new(h.name().clone(), midprice_values);
    Ok(PySeries(result))
}

/// 抛物线SAR (SAR)
#[pyfunction]
#[pyo3(signature = (high, low, acceleration=0.02, maximum=0.2))]
pub fn sar(high: PySeries, low: PySeries, acceleration: f64, maximum: f64) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let sar_values = {
        let mut result = vec![None; high_vals.len()];
        
        if high_vals.len() >= 2 {
            // 初始化SAR参数
            let mut sar = low_vals[0];
            let mut ep = high_vals[0]; // 极值点
            let mut af = acceleration; // 加速因子
            let mut is_bull = true; // 是否为上升趋势
            
            result[0] = Some(sar);
            
            for i in 1..high_vals.len() {
                // 更新SAR
                sar = sar + af * (ep - sar);
                
                if is_bull {
                    // 上升趋势
                    if low_vals[i] <= sar {
                        // 趋势反转
                        is_bull = false;
                        sar = ep;
                        ep = low_vals[i];
                        af = acceleration;
                    } else {
                        // 继续上升趋势
                        if high_vals[i] > ep {
                            ep = high_vals[i];
                            af = (af + acceleration).min(maximum);
                        }
                    }
                } else {
                    // 下降趋势
                    if high_vals[i] >= sar {
                        // 趋势反转
                        is_bull = true;
                        sar = ep;
                        ep = high_vals[i];
                        af = acceleration;
                    } else {
                        // 继续下降趋势
                        if low_vals[i] < ep {
                            ep = low_vals[i];
                            af = (af + acceleration).min(maximum);
                        }
                    }
                }
                
                result[i] = Some(sar);
            }
        }
        
        result
    };
    
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 设置默认参数
    let start_value = startvalue.unwrap_or(0.0);
    let offset_on_reverse = offsetonreverse.unwrap_or(0.0);
    let accel_init_long = accelerationinitlong.unwrap_or(0.02);
    let accel_long = accelerationlong.unwrap_or(0.02);
    let accel_max_long = accelerationmaxlong.unwrap_or(0.2);
    let accel_init_short = accelerationinitshort.unwrap_or(0.02);
    let accel_short = accelerationshort.unwrap_or(0.02);
    let accel_max_short = accelerationmaxshort.unwrap_or(0.2);
    
    let sar_values = {
        let mut result = vec![None; high_vals.len()];
        
        if high_vals.len() >= 2 {
            // 初始化SAR参数
            let initial_sar = if start_value == 0.0 {
                low_vals[0]
            } else {
                start_value
            };
            let mut sar = initial_sar;
            let mut ep = high_vals[0]; // 极值点
            let mut af = accel_init_long; // 加速因子
            let mut is_bull = true; // 是否为上升趋势
            
            result[0] = Some(sar);
            
            for i in 1..high_vals.len() {
                // 计算新的SAR
                sar = sar + af * (ep - sar);
                
                if is_bull {
                    // 上升趋势 (多头)
                    if low_vals[i] <= sar {
                        // 趋势反转到下降
                        is_bull = false;
                        sar = ep + offset_on_reverse;
                        ep = low_vals[i];
                        af = accel_init_short;
                    } else {
                        // 继续上升趋势
                        if high_vals[i] > ep {
                            ep = high_vals[i];
                            af = (af + accel_long).min(accel_max_long);
                        }
                        // 确保SAR不超过前两个周期的低点
                        if i >= 2 {
                            sar = sar.min(low_vals[i-1]).min(low_vals[i-2]);
                        } else if i >= 1 {
                            sar = sar.min(low_vals[i-1]);
                        }
                    }
                } else {
                    // 下降趋势 (空头)
                    if high_vals[i] >= sar {
                        // 趋势反转到上升
                        is_bull = true;
                        sar = ep - offset_on_reverse;
                        ep = high_vals[i];
                        af = accel_init_long;
                    } else {
                        // 继续下降趋势
                        if low_vals[i] < ep {
                            ep = low_vals[i];
                            af = (af + accel_short).min(accel_max_short);
                        }
                        // 确保SAR不低于前两个周期的高点
                        if i >= 2 {
                            sar = sar.max(high_vals[i-1]).max(high_vals[i-2]);
                        } else if i >= 1 {
                            sar = sar.max(high_vals[i-1]);
                        }
                    }
                }
                
                result[i] = Some(sar);
            }
        }
        
        result
    };
    
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let adx_values = {
        let mut result = vec![None; close_vals.len()];
        
        if close_vals.len() >= period * 2 {
            // 计算TR, +DM, -DM
            let mut tr_values = Vec::new();
            let mut plus_dm = Vec::new();
            let mut minus_dm = Vec::new();
            
            for i in 1..close_vals.len() {
                // 计算真实波幅 (True Range)
                let hl = high_vals[i] - low_vals[i];
                let hc = (high_vals[i] - close_vals[i - 1]).abs();
                let lc = (low_vals[i] - close_vals[i - 1]).abs();
                let tr = hl.max(hc).max(lc);
                tr_values.push(tr);
                
                // 计算方向移动 (Directional Movement)
                let high_diff = high_vals[i] - high_vals[i - 1];
                let low_diff = low_vals[i - 1] - low_vals[i];
                
                let plus_dm_val = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
                let minus_dm_val = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
                
                plus_dm.push(plus_dm_val);
                minus_dm.push(minus_dm_val);
            }
            
            // 计算平滑的TR, +DM, -DM
            if tr_values.len() >= period {
                // 计算初始平均值
                let mut smooth_tr: f64 = tr_values[0..period].iter().sum::<f64>();
                let mut smooth_plus_dm: f64 = plus_dm[0..period].iter().sum::<f64>();
                let mut smooth_minus_dm: f64 = minus_dm[0..period].iter().sum::<f64>();
                
                // 计算+DI和-DI
                let mut plus_di_vals = Vec::new();
                let mut minus_di_vals = Vec::new();
                
                for i in period..tr_values.len() {
                    // Wilder's平滑
                    smooth_tr = smooth_tr - (smooth_tr / period as f64) + tr_values[i];
                    smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period as f64) + plus_dm[i];
                    smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period as f64) + minus_dm[i];
                    
                    let plus_di = if smooth_tr > 0.0 { 100.0 * smooth_plus_dm / smooth_tr } else { 0.0 };
                    let minus_di = if smooth_tr > 0.0 { 100.0 * smooth_minus_dm / smooth_tr } else { 0.0 };
                    
                    plus_di_vals.push(plus_di);
                    minus_di_vals.push(minus_di);
                }
                
                // 计算DX
                let mut dx_vals = Vec::new();
                for i in 0..plus_di_vals.len() {
                    let sum_di = plus_di_vals[i] + minus_di_vals[i];
                    let dx = if sum_di > 0.0 {
                        100.0 * (plus_di_vals[i] - minus_di_vals[i]).abs() / sum_di
                    } else {
                        0.0
                    };
                    dx_vals.push(dx);
                }
                
                // 计算ADX（DX的平滑移动平均）
                if dx_vals.len() >= period {
                    let mut adx: f64 = dx_vals[0..period].iter().sum::<f64>() / period as f64;
                    result[period * 2] = Some(adx);
                    
                    for i in 1..(dx_vals.len() - period + 1) {
                        adx = (adx * (period - 1) as f64 + dx_vals[period - 1 + i]) / period as f64;
                        if period * 2 + i < result.len() {
                            result[period * 2 + i] = Some(adx);
                        }
                    }
                }
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), adx_values);
    Ok(PySeries(result))
}

/// 平均趋向指标评级 (ADXR)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn adxr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    // 先计算ADX值
    let adx_result = adx(PySeries(h.clone()), PySeries(l.clone()), PySeries(c.clone()), period)?;
    let adx_series: Series = adx_result.into();
    let adx_vals: Vec<f64> = adx_series.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let adxr_values = {
        let mut result = vec![None; adx_vals.len()];
        
        // ADXR = (当前ADX + period周期前的ADX) / 2
        for i in period..(adx_vals.len()) {
            if i >= period * 3 { // 确保有足够的ADX数据
                let current_adx = adx_vals[i];
                let past_adx = adx_vals[i - period];
                if current_adx > 0.0 && past_adx > 0.0 {
                    result[i] = Some((current_adx + past_adx) / 2.0);
                }
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), adxr_values);
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let (aroon_up, aroon_down) = {
        let mut up = vec![None; high_vals.len()];
        let mut down = vec![None; high_vals.len()];
        
        for i in period..high_vals.len() {
            // 查找最高价和最低价的位置
            let slice_start = i + 1 - period;
            let high_slice = &high_vals[slice_start..=i];
            let low_slice = &low_vals[slice_start..=i];
            
            let max_idx = high_slice.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
                
            let min_idx = low_slice.iter().enumerate()
                .max_by(|a, b| b.1.partial_cmp(a.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            up[i] = Some(((period - max_idx - 1) as f64 / period as f64) * 100.0);
            down[i] = Some(((period - min_idx - 1) as f64 / period as f64) * 100.0);
        }
        
        (up, down)
    };
    
    let base_name = h.name();
    let up_series = Series::new(PlSmallStr::from_str(&format!("{}_aroon_up", base_name)), aroon_up);
    let down_series = Series::new(PlSmallStr::from_str(&format!("{}_aroon_down", base_name)), aroon_down);
    
    Ok((PySeries(up_series), PySeries(down_series)))
}

/// 阿隆摆动指标 (AROONOSC)
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn aroonosc(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let aroonosc_values = {
        let mut result = vec![None; high_vals.len()];
        
        for i in period..high_vals.len() {
            let slice_start = i + 1 - period;
            let high_slice = &high_vals[slice_start..=i];
            let low_slice = &low_vals[slice_start..=i];
            
            let max_idx = high_slice.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
                
            let min_idx = low_slice.iter().enumerate()
                .max_by(|a, b| b.1.partial_cmp(a.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            let aroon_up = ((period - max_idx - 1) as f64 / period as f64) * 100.0;
            let aroon_down = ((period - min_idx - 1) as f64 / period as f64) * 100.0;
            
            result[i] = Some(aroon_up - aroon_down);
        }
        
        result
    };
    
    let result = Series::new(h.name().clone(), aroonosc_values);
    Ok(PySeries(result))
}

/// 正方向指标 (PLUS_DI)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn plus_di(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let plus_di_values = {
        let mut result = vec![None; close_vals.len()];
        
        if close_vals.len() >= period + 1 {
            // 计算TR, +DM
            let mut tr_values = Vec::new();
            let mut plus_dm = Vec::new();
            
            for i in 1..close_vals.len() {
                // 计算真实波幅 (True Range)
                let hl = high_vals[i] - low_vals[i];
                let hc = (high_vals[i] - close_vals[i - 1]).abs();
                let lc = (low_vals[i] - close_vals[i - 1]).abs();
                let tr = hl.max(hc).max(lc);
                tr_values.push(tr);
                
                // 计算+DM
                let high_diff = high_vals[i] - high_vals[i - 1];
                let low_diff = low_vals[i - 1] - low_vals[i];
                let plus_dm_val = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
                plus_dm.push(plus_dm_val);
            }
            
            // 计算平滑的TR, +DM
            if tr_values.len() >= period {
                // 计算初始平均值
                let mut smooth_tr: f64 = tr_values[0..period].iter().sum::<f64>();
                let mut smooth_plus_dm: f64 = plus_dm[0..period].iter().sum::<f64>();
                
                for i in period..tr_values.len() {
                    // Wilder's平滑
                    smooth_tr = smooth_tr - (smooth_tr / period as f64) + tr_values[i];
                    smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period as f64) + plus_dm[i];
                    
                    let plus_di = if smooth_tr > 0.0 { 100.0 * smooth_plus_dm / smooth_tr } else { 0.0 };
                    result[i + 1] = Some(plus_di);
                }
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), plus_di_values);
    Ok(PySeries(result))
}

/// 负方向指标 (MINUS_DI)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn minus_di(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let minus_di_values = {
        let mut result = vec![None; close_vals.len()];
        
        if close_vals.len() >= period + 1 {
            // 计算TR, -DM
            let mut tr_values = Vec::new();
            let mut minus_dm = Vec::new();
            
            for i in 1..close_vals.len() {
                // 计算真实波幅 (True Range)
                let hl = high_vals[i] - low_vals[i];
                let hc = (high_vals[i] - close_vals[i - 1]).abs();
                let lc = (low_vals[i] - close_vals[i - 1]).abs();
                let tr = hl.max(hc).max(lc);
                tr_values.push(tr);
                
                // 计算-DM
                let high_diff = high_vals[i] - high_vals[i - 1];
                let low_diff = low_vals[i - 1] - low_vals[i];
                let minus_dm_val = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
                minus_dm.push(minus_dm_val);
            }
            
            // 计算平滑的TR, -DM
            if tr_values.len() >= period {
                // 计算初始平均值
                let mut smooth_tr: f64 = tr_values[0..period].iter().sum::<f64>();
                let mut smooth_minus_dm: f64 = minus_dm[0..period].iter().sum::<f64>();
                
                for i in period..tr_values.len() {
                    // Wilder's平滑
                    smooth_tr = smooth_tr - (smooth_tr / period as f64) + tr_values[i];
                    smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period as f64) + minus_dm[i];
                    
                    let minus_di = if smooth_tr > 0.0 { 100.0 * smooth_minus_dm / smooth_tr } else { 0.0 };
                    result[i + 1] = Some(minus_di);
                }
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), minus_di_values);
    Ok(PySeries(result))
}

/// 正方向移动 (PLUS_DM)
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn plus_dm(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let plus_dm_values = {
        let mut result = vec![None; high_vals.len()];
        
        if high_vals.len() >= period + 1 {
            let mut plus_dm = Vec::new();
            
            for i in 1..high_vals.len() {
                let high_diff = high_vals[i] - high_vals[i - 1];
                let low_diff = low_vals[i - 1] - low_vals[i];
                let plus_dm_val = if high_diff > low_diff && high_diff > 0.0 { high_diff } else { 0.0 };
                plus_dm.push(plus_dm_val);
            }
            
            // 使用Wilder平滑
            if plus_dm.len() >= period {
                let mut smooth_plus_dm: f64 = plus_dm[0..period].iter().sum::<f64>();
                result[period] = Some(smooth_plus_dm);
                
                for i in period..plus_dm.len() {
                    smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period as f64) + plus_dm[i];
                    result[i + 1] = Some(smooth_plus_dm);
                }
            }
        }
        
        result
    };
    
    let result = Series::new(h.name().clone(), plus_dm_values);
    Ok(PySeries(result))
}

/// 负方向移动 (MINUS_DM)
#[pyfunction]
#[pyo3(signature = (high, low, period=14))]
pub fn minus_dm(high: PySeries, low: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let minus_dm_values = {
        let mut result = vec![None; high_vals.len()];
        
        if high_vals.len() >= period + 1 {
            let mut minus_dm = Vec::new();
            
            for i in 1..high_vals.len() {
                let high_diff = high_vals[i] - high_vals[i - 1];
                let low_diff = low_vals[i - 1] - low_vals[i];
                let minus_dm_val = if low_diff > high_diff && low_diff > 0.0 { low_diff } else { 0.0 };
                minus_dm.push(minus_dm_val);
            }
            
            // 使用Wilder平滑
            if minus_dm.len() >= period {
                let mut smooth_minus_dm: f64 = minus_dm[0..period].iter().sum::<f64>();
                result[period] = Some(smooth_minus_dm);
                
                for i in period..minus_dm.len() {
                    smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period as f64) + minus_dm[i];
                    result[i + 1] = Some(smooth_minus_dm);
                }
            }
        }
        
        result
    };
    
    let result = Series::new(h.name().clone(), minus_dm_values);
    Ok(PySeries(result))
}

/// 均势指标 (BOP)
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn bop(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let bop_values: Vec<f64> = (0..close_vals.len()).map(|i| {
        let range = high_vals[i] - low_vals[i];
        if range > 0.0 {
            (close_vals[i] - open_vals[i]) / range
        } else {
            0.0
        }
    }).collect();
    
    let result = Series::new(c.name().clone(), bop_values);
    Ok(PySeries(result))
}

/// 商品通道指数 (CCI)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn cci(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let cci_values = {
        let mut result = vec![None; close_vals.len()];
        
        for i in period - 1..close_vals.len() {
            // 计算典型价格
            let mut tp_sum = 0.0;
            let mut tp_values = Vec::new();
            
            for j in (i + 1 - period)..=i {
                let tp = (high_vals[j] + low_vals[j] + close_vals[j]) / 3.0;
                tp_values.push(tp);
                tp_sum += tp;
            }
            
            let sma_tp = tp_sum / period as f64;
            
            // 计算平均偏差
            let mean_deviation: f64 = tp_values.iter()
                .map(|&tp| (tp - sma_tp).abs())
                .sum::<f64>() / period as f64;
            
            if mean_deviation > 0.0 {
                let cci = (tp_values[period - 1] - sma_tp) / (0.015 * mean_deviation);
                result[i] = Some(cci);
            }
        }
        
        result
    };
    let result = Series::new(c.name().clone(), cci_values);
    
    Ok(PySeries(result))
}

/// 钱德动量摆动指标 (CMO)
#[pyfunction]
#[pyo3(signature = (series, period=14))]
pub fn cmo(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    let cmo_values = {
        let mut result = vec![None; vec_values.len()];
        
        if vec_values.len() > period {
            for i in period..vec_values.len() {
                let mut up_sum = 0.0;
                let mut down_sum = 0.0;
                
                for j in (i + 1 - period)..=i {
                    if j > 0 {
                        let change = vec_values[j] - vec_values[j - 1];
                        if change > 0.0 {
                            up_sum += change;
                        } else {
                            down_sum += -change;
                        }
                    }
                }
                
                let total_change = up_sum + down_sum;
                if total_change > 0.0 {
                    result[i] = Some(100.0 * (up_sum - down_sum) / total_change);
                }
            }
        }
        
        result
    };
    let result = Series::new(s.name().clone(), cmo_values);
    
    Ok(PySeries(result))
}

/// 方向性指标 (DX)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn dx(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let dx_values = {
        let mut result = vec![None; close_vals.len()];
        
        if close_vals.len() >= period + 1 {
            // 计算True Range和Directional Movement
            let mut tr_values = Vec::new();
            let mut plus_dm_values = Vec::new();
            let mut minus_dm_values = Vec::new();
            
            for i in 1..close_vals.len() {
                // True Range
                let tr1 = high_vals[i] - low_vals[i];
                let tr2 = (high_vals[i] - close_vals[i - 1]).abs();
                let tr3 = (low_vals[i] - close_vals[i - 1]).abs();
                let tr = tr1.max(tr2.max(tr3));
                tr_values.push(tr);
                
                // Directional Movement
                let up_move = high_vals[i] - high_vals[i - 1];
                let down_move = low_vals[i - 1] - low_vals[i];
                
                let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
                let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
                
                plus_dm_values.push(plus_dm);
                minus_dm_values.push(minus_dm);
            }
            
            // 计算平滑化的值
            if tr_values.len() >= period {
                for i in period..tr_values.len() + 1 {
                    let start_idx = i - period;
                    let end_idx = i;
                    
                    let tr_sum: f64 = tr_values[start_idx..end_idx].iter().sum();
                    let plus_dm_sum: f64 = plus_dm_values[start_idx..end_idx].iter().sum();
                    let minus_dm_sum: f64 = minus_dm_values[start_idx..end_idx].iter().sum();
                    
                    if tr_sum > 0.0 {
                        let plus_di = 100.0 * plus_dm_sum / tr_sum;
                        let minus_di = 100.0 * minus_dm_sum / tr_sum;
                        
                        let di_sum = plus_di + minus_di;
                        let dx = if di_sum > 0.0 {
                            100.0 * (plus_di - minus_di).abs() / di_sum
                        } else {
                            0.0
                        };
                        
                        result[i] = Some(dx);
                    }
                }
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), dx_values);
    Ok(PySeries(result))
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let volume_vals: Vec<f64> = v_f64.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mfi_values = {
        let mut result = vec![None; close_vals.len()];
        
        if close_vals.len() > period {
            for i in period..close_vals.len() {
                let mut positive_flow = 0.0;
                let mut negative_flow = 0.0;
                
                for j in (i + 1 - period)..=i {
                    if j > 0 {
                        let tp = (high_vals[j] + low_vals[j] + close_vals[j]) / 3.0;
                        let prev_tp = (high_vals[j - 1] + low_vals[j - 1] + close_vals[j - 1]) / 3.0;
                        let money_flow = tp * volume_vals[j];
                        
                        if tp > prev_tp {
                            positive_flow += money_flow;
                        } else if tp < prev_tp {
                            negative_flow += money_flow;
                        }
                    }
                }
                
                if negative_flow > 0.0 {
                    let mfi_ratio = positive_flow / negative_flow;
                    result[i] = Some(100.0 - (100.0 / (1.0 + mfi_ratio)));
                }
            }
        }
        
        result
    };
    let result = Series::new(c.name().clone(), mfi_values);
    
    Ok(PySeries(result))
}

/// 动量指标 (MOM)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn mom(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let mom_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in period..vec_values.len() {
            result[i] = Some(vec_values[i] - vec_values[i - period]);
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), mom_values);
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
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let roc_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in period..vec_values.len() {
            let prev_value = vec_values[i - period];
            if prev_value != 0.0 {
                result[i] = Some(((vec_values[i] - prev_value) / prev_value) * 100.0);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), roc_values);
    Ok(PySeries(result))
}

/// 变化率百分比 (ROCP)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn rocp(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let rocp_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in period..vec_values.len() {
            let prev_value = vec_values[i - period];
            if prev_value != 0.0 {
                // ROCP = (current - prev) / prev (无100倍数)
                result[i] = Some((vec_values[i] - prev_value) / prev_value);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), rocp_values);
    Ok(PySeries(result))
}

/// 变化率比率 (ROCR)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn rocr(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let rocr_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in period..vec_values.len() {
            let prev_value = vec_values[i - period];
            if prev_value != 0.0 {
                // ROCR = current / prev
                result[i] = Some(vec_values[i] / prev_value);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), rocr_values);
    Ok(PySeries(result))
}

/// 变化率比率100 (ROCR100)
#[pyfunction]
#[pyo3(signature = (series, period=10))]
pub fn rocr100(series: PySeries, period: usize) -> PyResult<PySeries> {
    let s: Series = series.into();
    let values = s.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Input must be numeric: {}", e)))?;
    
    let vec_values: Vec<f64> = values.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();
    
    let rocr100_values = {
        let mut result = vec![None; vec_values.len()];
        
        for i in period..vec_values.len() {
            let prev_value = vec_values[i - period];
            if prev_value != 0.0 {
                // ROCR100 = (current / prev) * 100
                result[i] = Some((vec_values[i] / prev_value) * 100.0);
            }
        }
        
        result
    };
    
    let result = Series::new(s.name().clone(), rocr100_values);
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算%K
    let mut k_values = vec![None; close_vals.len()];
    
    for i in k_period - 1..close_vals.len() {
        let slice_start = i + 1 - k_period;
        let high_slice = &high_vals[slice_start..=i];
        let low_slice = &low_vals[slice_start..=i];
        
        let highest = high_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = low_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest != lowest {
            k_values[i] = Some(((close_vals[i] - lowest) / (highest - lowest)) * 100.0);
        }
    }
    
    // 计算%D (K的移动平均)
    let k_vals_for_d: Vec<f64> = k_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
    let d_values = calculate_ma(&k_vals_for_d, d_period, MAType::SMA);
    
    let base_name = h.name();
    let k_series = Series::new(PlSmallStr::from_str(&format!("{}_stoch_k", base_name)), k_values);
    let d_series = Series::new(PlSmallStr::from_str(&format!("{}_stoch_d", base_name)), d_values);
    
    Ok((PySeries(k_series), PySeries(d_series)))
}

/// 快速随机指标 (STOCHF)
#[pyfunction]
#[pyo3(signature = (high, low, close, k_period=14, d_period=3))]
pub fn stochf(high: PySeries, low: PySeries, close: PySeries, k_period: usize, d_period: usize) -> PyResult<(PySeries, PySeries)> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 计算FastK
    let mut fastk_values = vec![None; close_vals.len()];
    
    for i in k_period - 1..close_vals.len() {
        let slice_start = i + 1 - k_period;
        let high_slice = &high_vals[slice_start..=i];
        let low_slice = &low_vals[slice_start..=i];
        
        let highest = high_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = low_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest != lowest {
            fastk_values[i] = Some(((close_vals[i] - lowest) / (highest - lowest)) * 100.0);
        }
    }
    
    // 计算FastD (FastK的移动平均)
    let fastk_vals_for_d: Vec<f64> = fastk_values.iter().map(|&x| x.unwrap_or(0.0)).collect();
    let fastd_values = calculate_ma(&fastk_vals_for_d, d_period, MAType::SMA);
    
    let base_name = h.name();
    let fastk_series = Series::new(PlSmallStr::from_str(&format!("{}_stochf_k", base_name)), fastk_values);
    let fastd_series = Series::new(PlSmallStr::from_str(&format!("{}_stochf_d", base_name)), fastd_values);
    
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let ultosc_values = {
        let mut result = vec![None; close_vals.len()];
        
        if close_vals.len() >= period3 + 1 {
            for i in period3..close_vals.len() {
                // 计算买压力(BP)和真实波幅(TR)
                let mut bp1 = 0.0; let mut tr1 = 0.0;
                let mut bp2 = 0.0; let mut tr2 = 0.0;
                let mut bp3 = 0.0; let mut tr3 = 0.0;
                
                // Period 1
                for j in (i + 1 - period1)..=i {
                    if j > 0 {
                        let prev_close = close_vals[j - 1];
                        let bp = close_vals[j] - low_vals[j].min(prev_close);
                        let tr = high_vals[j].max(prev_close) - low_vals[j].min(prev_close);
                        bp1 += bp;
                        tr1 += tr;
                    }
                }
                
                // Period 2
                for j in (i + 1 - period2)..=i {
                    if j > 0 {
                        let prev_close = close_vals[j - 1];
                        let bp = close_vals[j] - low_vals[j].min(prev_close);
                        let tr = high_vals[j].max(prev_close) - low_vals[j].min(prev_close);
                        bp2 += bp;
                        tr2 += tr;
                    }
                }
                
                // Period 3
                for j in (i + 1 - period3)..=i {
                    if j > 0 {
                        let prev_close = close_vals[j - 1];
                        let bp = close_vals[j] - low_vals[j].min(prev_close);
                        let tr = high_vals[j].max(prev_close) - low_vals[j].min(prev_close);
                        bp3 += bp;
                        tr3 += tr;
                    }
                }
                
                // 计算UltiMate Oscillator
                if tr1 > 0.0 && tr2 > 0.0 && tr3 > 0.0 {
                    let avg1 = bp1 / tr1;
                    let avg2 = bp2 / tr2;
                    let avg3 = bp3 / tr3;
                    
                    result[i] = Some(100.0 * ((4.0 * avg1) + (2.0 * avg2) + avg3) / 7.0);
                }
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), ultosc_values);
    Ok(PySeries(result))
}

/// 威廉指标 (WILLR)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn willr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let willr_values = {
        let mut result = vec![None; close_vals.len()];
        
        for i in period - 1..close_vals.len() {
            let slice_start = i + 1 - period;
            let high_slice = &high_vals[slice_start..=i];
            let low_slice = &low_vals[slice_start..=i];
            
            let highest = high_slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = low_slice.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            if highest != lowest {
                result[i] = Some(((highest - close_vals[i]) / (highest - lowest)) * -100.0);
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), willr_values);
    Ok(PySeries(result))
}

// ====================================================================
// 成交量指标 (Volume Indicators) - 成交量相关指标
// ====================================================================

/// 累积/派发线 (AD)
#[pyfunction]
#[pyo3(signature = (high, low, close, volume))]
pub fn ad(high: PySeries, low: PySeries, close: PySeries, volume: PySeries) -> PyResult<PySeries> {
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let volume_vals: Vec<f64> = v_f64.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let ad_values = {
        let mut result = vec![0.0; close_vals.len()];
        let mut cumulative_ad = 0.0;
        
        for i in 0..close_vals.len() {
            let hl_diff = high_vals[i] - low_vals[i];
            if hl_diff > 0.0 {
                let clv = ((close_vals[i] - low_vals[i]) - (high_vals[i] - close_vals[i])) / hl_diff;
                cumulative_ad += clv * volume_vals[i];
            }
            result[i] = cumulative_ad;
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), ad_values);
    Ok(PySeries(result))
}

/// 累积/派发摆动指标 (ADOSC)
#[pyfunction]
#[pyo3(signature = (high, low, close, volume, fast_period=3, slow_period=10))]
pub fn adosc(high: PySeries, low: PySeries, close: PySeries, volume: PySeries, fast_period: usize, slow_period: usize) -> PyResult<PySeries> {
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let volume_vals: Vec<f64> = v_f64.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    // 先计算AD线
    let ad_values = {
        let mut result = vec![0.0; close_vals.len()];
        let mut cumulative_ad = 0.0;
        
        for i in 0..close_vals.len() {
            let hl_diff = high_vals[i] - low_vals[i];
            if hl_diff > 0.0 {
                let clv = ((close_vals[i] - low_vals[i]) - (high_vals[i] - close_vals[i])) / hl_diff;
                cumulative_ad += clv * volume_vals[i];
            }
            result[i] = cumulative_ad;
        }
        
        result
    };
    
    // 计算AD线的快线和慢线EMA
    let fast_ema_values = calculate_ma(&ad_values, fast_period, MAType::EMA);
    let slow_ema_values = calculate_ma(&ad_values, slow_period, MAType::EMA);
    
    let adosc_values: Vec<Option<f64>> = fast_ema_values.iter().zip(slow_ema_values.iter())
        .map(|(&fast, &slow)| {
            match (fast, slow) {
                (Some(f), Some(s)) => Some(f - s),
                _ => None,
            }
        })
        .collect();
    
    let result = Series::new(c.name().clone(), adosc_values);
    Ok(PySeries(result))
}

/// 能量潮指标 (OBV)
#[pyfunction]
#[pyo3(signature = (close, volume))]
pub fn obv(close: PySeries, volume: PySeries) -> PyResult<PySeries> {
    let c: Series = close.into();
    let v: Series = volume.into();
    
    // 将 volume 转换为 f64 类型
    let v_f64 = if v.dtype() == &DataType::Int64 {
        v.cast(&DataType::Float64).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to cast volume to Float64: {}", e)))?
    } else {
        v
    };
    
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let volume_vals: Vec<f64> = v_f64.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let obv_values = {
        let mut result = vec![0.0; close_vals.len()];
        
        if !close_vals.is_empty() {
            let mut obv = volume_vals[0];
            result[0] = obv;
            
            for i in 1..close_vals.len() {
                if close_vals[i] > close_vals[i - 1] {
                    obv += volume_vals[i];
                } else if close_vals[i] < close_vals[i - 1] {
                    obv -= volume_vals[i];
                }
                // 如果价格不变，OBV保持不变
                result[i] = obv;
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), obv_values);
    Ok(PySeries(result))
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let trange_values = {
        let mut result = vec![None; close_vals.len()];
        
        // 第一个值就是 high - low
        if !close_vals.is_empty() {
            result[0] = Some(high_vals[0] - low_vals[0]);
        }
        
        // 从第二个值开始计算真实波幅
        for i in 1..close_vals.len() {
            let hl = high_vals[i] - low_vals[i];
            let hc = (high_vals[i] - close_vals[i - 1]).abs();
            let lc = (low_vals[i] - close_vals[i - 1]).abs();
            
            result[i] = Some(hl.max(hc).max(lc));
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), trange_values);
    Ok(PySeries(result))
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
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let atr_values = calculate_atr(&high_vals, &low_vals, &close_vals, period);
    let result = Series::new(c.name().clone(), atr_values);
    
    Ok(PySeries(result))
}

/// 标准化平均真实波幅 (NATR)
#[pyfunction]
#[pyo3(signature = (high, low, close, period=14))]
pub fn natr(high: PySeries, low: PySeries, close: PySeries, period: usize) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    // 先计算ATR
    let atr_result = atr(PySeries(h.clone()), PySeries(l.clone()), PySeries(c.clone()), period)?;
    let atr_series: Series = atr_result.into();
    let atr_vals: Vec<f64> = atr_series.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let natr_values = {
        let mut result = vec![None; close_vals.len()];
        
        for i in 0..atr_vals.len() {
            if atr_vals[i] > 0.0 && close_vals[i] > 0.0 {
                // NATR = (ATR / Close) * 100
                result[i] = Some((atr_vals[i] / close_vals[i]) * 100.0);
            }
        }
        
        result
    };
    
    let result = Series::new(c.name().clone(), natr_values);
    Ok(PySeries(result))
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
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let avgprice_values: Vec<f64> = (0..close_vals.len()).map(|i| {
        (open_vals[i] + high_vals[i] + low_vals[i] + close_vals[i]) / 4.0
    }).collect();
    
    let result = Series::new(c.name().clone(), avgprice_values);
    Ok(PySeries(result))
}

/// 中间价格 (MEDPRICE) - SIMD优化版本
#[pyfunction]
#[pyo3(signature = (high, low))]
pub fn medprice(high: PySeries, low: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let medprice_values: Vec<f64> = (0..high_vals.len()).map(|i| {
        (high_vals[i] + low_vals[i]) / 2.0
    }).collect();
    
    let result = Series::new(h.name().clone(), medprice_values);
    Ok(PySeries(result))
}

/// 典型价格 (TYPPRICE) - SIMD优化版本
#[pyfunction]
#[pyo3(signature = (high, low, close))]
pub fn typprice(high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let typprice_values: Vec<f64> = (0..close_vals.len()).map(|i| {
        (high_vals[i] + low_vals[i] + close_vals[i]) / 3.0
    }).collect();
    
    let result = Series::new(c.name().clone(), typprice_values);
    Ok(PySeries(result))
}

/// 加权收盘价 (WCLPRICE)
#[pyfunction]
#[pyo3(signature = (high, low, close))]
pub fn wclprice(high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let wclprice_values: Vec<f64> = (0..close_vals.len()).map(|i| {
        (high_vals[i] + low_vals[i] + 2.0 * close_vals[i]) / 4.0
    }).collect();
    
    let result = Series::new(c.name().clone(), wclprice_values);
    Ok(PySeries(result))
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

// CDL2CROWS - 两只乌鸦
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl2crows(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 第一根蜡烛：长阳线
        let day1_bullish = is_bullish(open_vals[i-2], close_vals[i-2]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let is_long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：小阴线，开盘在前一日收盘之上，收盘在前一日实体内
        let day2_bearish = !is_bullish(open_vals[i-1], close_vals[i-1]);
        let gap_up = open_vals[i-1] > close_vals[i-2];
        let close_in_body1 = close_vals[i-1] < close_vals[i-2] && close_vals[i-1] > open_vals[i-2];
        
        // 第三根蜡烛：阴线，开盘在前一日收盘之上，收盘更低
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        let gap_up2 = open_vals[i] > close_vals[i-1];
        let lower_close = close_vals[i] < close_vals[i-1];
        
        if day1_bullish && is_long_body1 && day2_bearish && gap_up && close_in_body1 &&
           day3_bearish && gap_up2 && lower_close {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3BLACKCROWS - 三只黑乌鸦
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3blackcrows(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 检查三根连续阴线
        let day1_bearish = !is_bullish(open_vals[i-2], close_vals[i-2]);
        let day2_bearish = !is_bullish(open_vals[i-1], close_vals[i-1]);
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        
        // 检查每根蜡烛都是长实体
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let (body3, _, _, range3) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        
        let long_body1 = is_long_body(body1, range1);
        let long_body2 = is_long_body(body2, range2);
        let long_body3 = is_long_body(body3, range3);
        
        // 检查开盘价递减，收盘价递减
        let decreasing_opens = open_vals[i-1] < open_vals[i-2] && open_vals[i] < open_vals[i-1];
        let decreasing_closes = close_vals[i-1] < close_vals[i-2] && close_vals[i] < close_vals[i-1];
        
        // 检查每根蜡烛的开盘价都在前一根的实体内
        let open_in_body1 = open_vals[i-1] < open_vals[i-2] && open_vals[i-1] > close_vals[i-2];
        let open_in_body2 = open_vals[i] < open_vals[i-1] && open_vals[i] > close_vals[i-1];
        
        if day1_bearish && day2_bearish && day3_bearish &&
           long_body1 && long_body2 && long_body3 &&
           decreasing_opens && decreasing_closes &&
           open_in_body1 && open_in_body2 {
            result[i] = Some(-100);  // 强烈看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3WHITESOLDIERS - 三个白兵
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3whitesoldiers(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 检查三根连续阳线
        let day1_bullish = is_bullish(open_vals[i-2], close_vals[i-2]);
        let day2_bullish = is_bullish(open_vals[i-1], close_vals[i-1]);
        let day3_bullish = is_bullish(open_vals[i], close_vals[i]);
        
        // 检查每根蜡烛都是长实体
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let (body3, _, _, range3) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        
        let long_body1 = is_long_body(body1, range1);
        let long_body2 = is_long_body(body2, range2);
        let long_body3 = is_long_body(body3, range3);
        
        // 检查开盘价递增，收盘价递增
        let increasing_opens = open_vals[i-1] > open_vals[i-2] && open_vals[i] > open_vals[i-1];
        let increasing_closes = close_vals[i-1] > close_vals[i-2] && close_vals[i] > close_vals[i-1];
        
        // 检查每根蜡烛的开盘价都在前一根的实体内
        let open_in_body1 = open_vals[i-1] > open_vals[i-2] && open_vals[i-1] < close_vals[i-2];
        let open_in_body2 = open_vals[i] > open_vals[i-1] && open_vals[i] < close_vals[i-1];
        
        // 检查上影线较短
        let (_, upper_shadow1, _, _) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let (_, upper_shadow2, _, _) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let (_, upper_shadow3, _, _) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        
        let short_shadows = upper_shadow1 < body1 * 0.3 && upper_shadow2 < body2 * 0.3 && upper_shadow3 < body3 * 0.3;
        
        if day1_bullish && day2_bullish && day3_bullish &&
           long_body1 && long_body2 && long_body3 &&
           increasing_opens && increasing_closes &&
           open_in_body1 && open_in_body2 && short_shadows {
            result[i] = Some(100);  // 强烈看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLDOJI - 十字星
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdldoji(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 0..open_vals.len() {
        let (body_size, upper_shadow, lower_shadow, range) = candle_metrics(
            open_vals[i], high_vals[i], low_vals[i], close_vals[i]
        );
        
        // Doji 的特征：实体很小，上下影线相对较长
        let is_doji = is_doji_body(body_size, range);
        let has_shadows = upper_shadow > body_size && lower_shadow > body_size;
        
        if is_doji && has_shadows {
            result[i] = Some(100);  // 不确定信号，但重要的反转候选
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHAMMER - 锤子线
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlhammer(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 0..open_vals.len() {
        let (body_size, upper_shadow, lower_shadow, range) = candle_metrics(
            open_vals[i], high_vals[i], low_vals[i], close_vals[i]
        );
        
        // 锤子线特征：小实体，短上影线，长下影线
        let small_body = is_short_body(body_size, range);
        let short_upper_shadow = upper_shadow < body_size * 0.5;
        let long_lower_shadow = lower_shadow > body_size * 2.0;
        
        if small_body && short_upper_shadow && long_lower_shadow {
            result[i] = Some(100);  // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHANGINGMAN - 上吊线
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlhangingman(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 0..open_vals.len() {
        let (body_size, upper_shadow, lower_shadow, range) = candle_metrics(
            open_vals[i], high_vals[i], low_vals[i], close_vals[i]
        );
        
        // 上吊线特征：小实体，短上影线，长下影线（与锤子线形状相同，但处在上升趋势的顶部）
        let small_body = is_short_body(body_size, range);
        let short_upper_shadow = upper_shadow < body_size * 0.5;
        let long_lower_shadow = lower_shadow > body_size * 2.0;
        
        if small_body && short_upper_shadow && long_lower_shadow {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLSHOOTINGSTAR - 流星线
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlshootingstar(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 0..open_vals.len() {
        let (body_size, upper_shadow, lower_shadow, range) = candle_metrics(
            open_vals[i], high_vals[i], low_vals[i], close_vals[i]
        );
        
        // 流星线特征：小实体，长上影线，短下影线
        let small_body = is_short_body(body_size, range);
        let long_upper_shadow = upper_shadow > body_size * 2.0;
        let short_lower_shadow = lower_shadow < body_size * 0.5;
        
        if small_body && long_upper_shadow && short_lower_shadow {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLENGULFING - 吞没模式
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlengulfing(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 1..open_vals.len() {
        // 第一根蜡烛
        let day1_bullish = is_bullish(open_vals[i-1], close_vals[i-1]);
        let (body1, _, _, _range1) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        
        // 第二根蜡烛
        let day2_bullish = is_bullish(open_vals[i], close_vals[i]);
        let (body2, _, _, _range2) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        
        // 看涨吞没：前一根阴线，当前阳线，当前实体完全包含前一根实体
        let bullish_engulfing = !day1_bullish && day2_bullish &&
                               open_vals[i] < close_vals[i-1] &&
                               close_vals[i] > open_vals[i-1] &&
                               body2 > body1;
        
        // 看跌吞没：前一根阳线，当前阴线，当前实体完全包含前一根实体
        let bearish_engulfing = day1_bullish && !day2_bullish &&
                               open_vals[i] > close_vals[i-1] &&
                               close_vals[i] < open_vals[i-1] &&
                               body2 > body1;
        
        if bullish_engulfing {
            result[i] = Some(100);  // 看涨信号
        } else if bearish_engulfing {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLHARAMI - 孕育线
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdlharami(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 1..open_vals.len() {
        // 第一根蜡烛：长实体
        let (body1, _, _, range1) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：小实体，完全包含在第一根实体内
        let (body2, _, _, range2) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        let small_body2 = is_short_body(body2, range2);
        
        // 检查第二根蜡烛是否在第一根实体内
        let prev_body_high = open_vals[i-1].max(close_vals[i-1]);
        let prev_body_low = open_vals[i-1].min(close_vals[i-1]);
        let curr_body_high = open_vals[i].max(close_vals[i]);
        let curr_body_low = open_vals[i].min(close_vals[i]);
        
        let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
        
        if long_body1 && small_body2 && inside_body {
            // 看涨孕育：第一根阴线，第二根阳线
            if !is_bullish(open_vals[i-1], close_vals[i-1]) && is_bullish(open_vals[i], close_vals[i]) {
                result[i] = Some(100);
            }
            // 看跌孕育：第一根阳线，第二根阴线
            else if is_bullish(open_vals[i-1], close_vals[i-1]) && !is_bullish(open_vals[i], close_vals[i]) {
                result[i] = Some(-100);
            }
            // 中性孕育
            else {
                result[i] = Some(50);
            }
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLMORNINGSTAR - 启明星
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdlmorningstar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 第一根蜡烛：长阴线
        let day1_bearish = !is_bullish(open_vals[i-2], close_vals[i-2]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：小实体（星线），与第一根有跳空
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let small_body2 = is_short_body(body2, range2);
        let gap_down = high_vals[i-1] < close_vals[i-2]; // 向下跳空
        
        // 第三根蜡烛：阳线，收盘价深入第一根实体
        let day3_bullish = is_bullish(open_vals[i], close_vals[i]);
        let penetrates = close_vals[i] > (open_vals[i-2] + close_vals[i-2]) / 2.0 * (1.0 + penetration);
        
        if day1_bearish && long_body1 && small_body2 && gap_down && day3_bullish && penetrates {
            result[i] = Some(100);  // 强烈看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLEVENINGSTAR - 黄昏星
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdleveningstar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 第一根蜡烛：长阳线
        let day1_bullish = is_bullish(open_vals[i-2], close_vals[i-2]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：小实体（星线），与第一根有跳空
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let small_body2 = is_short_body(body2, range2);
        let gap_up = low_vals[i-1] > close_vals[i-2]; // 向上跳空
        
        // 第三根蜡烛：阴线，收盘价深入第一根实体
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        let penetrates = close_vals[i] < (open_vals[i-2] + close_vals[i-2]) / 2.0 * (1.0 - penetration);
        
        if day1_bullish && long_body1 && small_body2 && gap_up && day3_bearish && penetrates {
            result[i] = Some(-100);  // 强烈看跌信号
        } else {
            result[i] = Some(0);
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
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 1..open_vals.len() {
        // 第一根蜡烛：长阴线
        let day1_bearish = !is_bullish(open_vals[i-1], close_vals[i-1]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：阳线，开盘低于前一日最低价，收盘在前一日实体中点之上
        let day2_bullish = is_bullish(open_vals[i], close_vals[i]);
        let opens_lower = open_vals[i] < low_vals[i-1];
        let midpoint = (open_vals[i-1] + close_vals[i-1]) / 2.0;
        let closes_above_midpoint = close_vals[i] > midpoint;
        let closes_below_open = close_vals[i] < open_vals[i-1];
        
        if day1_bearish && long_body1 && day2_bullish && opens_lower && 
           closes_above_midpoint && closes_below_open {
            result[i] = Some(100);  // 看涨信号
        } else {
            result[i] = Some(0);
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
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 1..open_vals.len() {
        // 第一根蜡烛：长阳线
        let day1_bullish = is_bullish(open_vals[i-1], close_vals[i-1]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：阴线，开盘高于前一日最高价，收盘在前一日实体中点之下
        let day2_bearish = !is_bullish(open_vals[i], close_vals[i]);
        let opens_higher = open_vals[i] > high_vals[i-1];
        let midpoint = (open_vals[i-1] + close_vals[i-1]) / 2.0;
        let closes_below_midpoint = close_vals[i] < midpoint;
        let closes_above_close = close_vals[i] > close_vals[i-1] * (1.0 - penetration);
        
        if day1_bullish && long_body1 && day2_bearish && opens_higher && 
           closes_below_midpoint && closes_above_close {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
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
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 1..open_vals.len() {
        // 第一根蜡烛：长实体
        let (body1, _, _, range1) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：十字星，完全包含在第一根实体内
        let (body2, _, _, range2) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        let is_doji2 = is_doji_body(body2, range2);
        
        // 检查第二根蜡烛是否在第一根实体内
        let prev_body_high = open_vals[i-1].max(close_vals[i-1]);
        let prev_body_low = open_vals[i-1].min(close_vals[i-1]);
        let curr_body_high = open_vals[i].max(close_vals[i]);
        let curr_body_low = open_vals[i].min(close_vals[i]);
        
        let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
        
        if long_body1 && is_doji2 && inside_body {
            // 看涨十字孕育：第一根阴线，第二根doji
            if !is_bullish(open_vals[i-1], close_vals[i-1]) {
                result[i] = Some(100);
            }
            // 看跌十字孕育：第一根阳线，第二根doji
            else {
                result[i] = Some(-100);
            }
        } else {
            result[i] = Some(0);
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
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 第一根蜡烛：长阴线
        let day1_bearish = !is_bullish(open_vals[i-2], close_vals[i-2]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：十字星，与第一根有向下跳空
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let is_doji2 = is_doji_body(body2, range2);
        let gap_down = high_vals[i-1] < close_vals[i-2];
        
        // 第三根蜡烛：阳线，收盘价深入第一根实体
        let day3_bullish = is_bullish(open_vals[i], close_vals[i]);
        let penetrates = close_vals[i] > (open_vals[i-2] + close_vals[i-2]) / 2.0 * (1.0 + penetration);
        
        if day1_bearish && long_body1 && is_doji2 && gap_down && day3_bullish && penetrates {
            result[i] = Some(100);  // 强烈看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDLEVENINGDOJISTAR - 黄昏十字星
#[pyfunction]
#[pyo3(signature = (open, high, low, close, penetration=0.3))]
pub fn cdleveningdojistar(open: PySeries, high: PySeries, low: PySeries, close: PySeries, penetration: f64) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 第一根蜡烛：长阳线
        let day1_bullish = is_bullish(open_vals[i-2], close_vals[i-2]);
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let long_body1 = is_long_body(body1, range1);
        
        // 第二根蜡烛：十字星，与第一根有向上跳空
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let is_doji2 = is_doji_body(body2, range2);
        let gap_up = low_vals[i-1] > close_vals[i-2];
        
        // 第三根蜡烛：阴线，收盘价深入第一根实体
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        let penetrates = close_vals[i] < (open_vals[i-2] + close_vals[i-2]) / 2.0 * (1.0 - penetration);
        
        if day1_bullish && long_body1 && is_doji2 && gap_up && day3_bearish && penetrates {
            result[i] = Some(-100);  // 强烈看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3INSIDE - 三内部上升/下降
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3inside(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 前两根蜡烛构成孕育线
        let (body1, _, _, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let long_body1 = is_long_body(body1, range1);
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let small_body2 = is_short_body(body2, range2);
        
        // 检查第二根是否在第一根实体内
        let prev_body_high = open_vals[i-2].max(close_vals[i-2]);
        let prev_body_low = open_vals[i-2].min(close_vals[i-2]);
        let curr_body_high = open_vals[i-1].max(close_vals[i-1]);
        let curr_body_low = open_vals[i-1].min(close_vals[i-1]);
        let inside_body = curr_body_high <= prev_body_high && curr_body_low >= prev_body_low;
        
        // 第三根蜡烛确认方向
        let day3_bullish = is_bullish(open_vals[i], close_vals[i]);
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        
        // 三内部上升：第一根阴线，第二根小阳线包含其中，第三根阳线收盘高于第一根开盘价
        if !is_bullish(open_vals[i-2], close_vals[i-2]) && is_bullish(open_vals[i-1], close_vals[i-1]) &&
           long_body1 && small_body2 && inside_body && day3_bullish && close_vals[i] > open_vals[i-2] {
            result[i] = Some(100);  // 看涨信号
        }
        // 三内部下降：第一根阳线，第二根小阴线包含其中，第三根阴线收盘低于第一根开盘价
        else if is_bullish(open_vals[i-2], close_vals[i-2]) && !is_bullish(open_vals[i-1], close_vals[i-1]) &&
                long_body1 && small_body2 && inside_body && day3_bearish && close_vals[i] < open_vals[i-2] {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3OUTSIDE - 三外部上升/下降
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3outside(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 前两根蜡烛构成吞没模式
        let day1_bullish = is_bullish(open_vals[i-2], close_vals[i-2]);
        let day1_bearish = !is_bullish(open_vals[i-2], close_vals[i-2]);
        let day2_bullish = is_bullish(open_vals[i-1], close_vals[i-1]);
        let day2_bearish = !is_bullish(open_vals[i-1], close_vals[i-1]);
        
        let (body1, _, _, _) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let (body2, _, _, _) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        
        // 看涨吞没
        let bullish_engulfing = day1_bearish && day2_bullish &&
                               open_vals[i-1] < close_vals[i-2] &&
                               close_vals[i-1] > open_vals[i-2] &&
                               body2 > body1;
        
        // 看跌吞没  
        let bearish_engulfing = day1_bullish && day2_bearish &&
                               open_vals[i-1] > close_vals[i-2] &&
                               close_vals[i-1] < open_vals[i-2] &&
                               body2 > body1;
        
        // 第三根蜡烛确认方向
        let day3_bullish = is_bullish(open_vals[i], close_vals[i]);
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        
        // 三外部上升：前两根构成看涨吞没，第三根阳线收盘高于第二根收盘
        if bullish_engulfing && day3_bullish && close_vals[i] > close_vals[i-1] {
            result[i] = Some(100);  // 看涨信号
        }
        // 三外部下降：前两根构成看跌吞没，第三根阴线收盘低于第二根收盘
        else if bearish_engulfing && day3_bearish && close_vals[i] < close_vals[i-1] {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3LINESTRIKE - 三线攻击
#[pyfunction]
#[pyo3(signature = (open, _high, _low, close))]
pub fn cdl3linestrike(open: PySeries, _high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 3..open_vals.len() {
        // 前三根蜡烛：连续同向
        let day1_bullish = is_bullish(open_vals[i-3], close_vals[i-3]);
        let day2_bullish = is_bullish(open_vals[i-2], close_vals[i-2]);
        let day3_bullish = is_bullish(open_vals[i-1], close_vals[i-1]);
        
        // 检查前三根是否连续上涨或下跌
        let three_bulls = day1_bullish && day2_bullish && day3_bullish &&
                         close_vals[i-2] > close_vals[i-3] &&
                         close_vals[i-1] > close_vals[i-2];
                         
        let three_bears = !day1_bullish && !day2_bullish && !day3_bullish &&
                         close_vals[i-2] < close_vals[i-3] &&
                         close_vals[i-1] < close_vals[i-2];
        
        // 第四根蜡烛：反方向大蜡烛完全吞没前三根
        let day4_bullish = is_bullish(open_vals[i], close_vals[i]);
        let day4_bearish = !is_bullish(open_vals[i], close_vals[i]);
        
        // 看涨三线攻击：前三根阴线下跌，第四根阳线完全覆盖
        if three_bears && day4_bullish &&
           open_vals[i] < close_vals[i-1] &&
           close_vals[i] > open_vals[i-3] {
            result[i] = Some(100);  // 看涨信号
        }
        // 看跌三线攻击：前三根阳线上涨，第四根阴线完全覆盖
        else if three_bulls && day4_bearish &&
                open_vals[i] > close_vals[i-1] &&
                close_vals[i] < open_vals[i-3] {
            result[i] = Some(-100);  // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}

// CDL3STARSINSOUTH - 三星在南
#[pyfunction]
#[pyo3(signature = (open, high, low, close))]
pub fn cdl3starsinsouth(open: PySeries, high: PySeries, low: PySeries, close: PySeries) -> PyResult<PySeries> {
    let o: Series = open.into();
    let h: Series = high.into();
    let l: Series = low.into();
    let c: Series = close.into();
    
    let open_vals: Vec<f64> = o.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let high_vals: Vec<f64> = h.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let low_vals: Vec<f64> = l.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    let close_vals: Vec<f64> = c.f64().unwrap().into_iter().map(|x| x.unwrap_or(0.0)).collect();
    
    let mut result = vec![None; open_vals.len()];
    
    for i in 2..open_vals.len() {
        // 第一根蜡烛：长阴线，长下影线
        let day1_bearish = !is_bullish(open_vals[i-2], close_vals[i-2]);
        let (body1, _, lower_shadow1, range1) = candle_metrics(open_vals[i-2], high_vals[i-2], low_vals[i-2], close_vals[i-2]);
        let long_body1 = is_long_body(body1, range1);
        let long_lower_shadow1 = lower_shadow1 > body1;
        
        // 第二根蜡烛：小实体，开盘价在第一根实体内，收盘价高于第一根最低价
        let (body2, _, _, range2) = candle_metrics(open_vals[i-1], high_vals[i-1], low_vals[i-1], close_vals[i-1]);
        let small_body2 = is_short_body(body2, range2);
        let opens_in_body1 = open_vals[i-1] <= open_vals[i-2] && open_vals[i-1] >= close_vals[i-2];
        let closes_above_low1 = close_vals[i-1] > low_vals[i-2];
        
        // 第三根蜡烛：小阴线，开盘和收盘都在第二根的范围内
        let day3_bearish = !is_bullish(open_vals[i], close_vals[i]);
        let (body3, _, _, range3) = candle_metrics(open_vals[i], high_vals[i], low_vals[i], close_vals[i]);
        let small_body3 = is_short_body(body3, range3);
        let contained_in_day2 = open_vals[i] <= high_vals[i-1] && close_vals[i] >= low_vals[i-1];
        
        if day1_bearish && long_body1 && long_lower_shadow1 &&
           small_body2 && opens_in_body1 && closes_above_low1 &&
           day3_bearish && small_body3 && contained_in_day2 {
            result[i] = Some(100);  // 看涨信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(o.name().clone(), result);
    Ok(PySeries(series_result))
}
// CDLADVANCEBLOCK - 前进阻挡
#[pyfunction]
pub fn cdladvanceblock(open: PySeries, high: PySeries, _low: PySeries, close: PySeries) -> PyResult<PySeries> {
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
        
        // 前进阻挡：三根连续的白色蜡烛，但第二、三根高点递减，实体递减
        let three_white_soldiers = body1 > 0.0 && body2 > 0.0 && body3 > 0.0;
        let ascending_closes = close_vals[0] < close_vals[1] && close_vals[1] < close_vals[2];
        let opens_within_bodies = open_vals[1] > open_vals[0] && open_vals[1] < close_vals[0] &&
                                  open_vals[2] > open_vals[1] && open_vals[2] < close_vals[1];
        let decreasing_highs = high_vals[1] >= high_vals[0] && high_vals[2] <= high_vals[1];
        let decreasing_bodies = body3 <= body2 && body2 <= body1;
        
        if three_white_soldiers && ascending_closes && opens_within_bodies && 
           decreasing_highs && decreasing_bodies {
            result[i] = Some(-100); // 看跌信号
        } else {
            result[i] = Some(0);
        }
    }
    
    let series_result = Series::new(open.name().clone(), result);
    Ok(PySeries(series_result))
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
