use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

/// 因子挖掘和评估模块
/// 
/// 提供因子计算、因子评估、IC分析等功能

// ============================================================================
// 因子计算 (Factor Calculation)
// ============================================================================

#[pyclass]
pub struct Factor;

#[pymethods]
impl Factor {
    #[new]
    pub fn new() -> Self {
        Factor
    }

    // ========================================================================
    // 因子计算方法 (Factor Calculation Methods)
    // ========================================================================

    /// 动量因子（增强版）
    /// 
    /// 计算价格动量，支持多种计算方式
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名（默认"close"）
    /// - `period`: 回看周期（默认20天）
    /// - `method`: 计算方法（默认"return"）
    ///   - "return": 简单收益率 (current - past) / past
    ///   - "log": 对数收益率 ln(current / past)
    ///   - "residual": 残差动量（去除市场整体趋势）
    ///   - "acceleration": 动量加速度（动量的变化率）
    /// - `factor_col`: 因子列名（默认"momentum"）
    /// 
    /// # 返回
    /// 包含动量因子列的DataFrame
    /// 
    /// # 示例
    /// ```python
    /// # 简单收益率动量（默认）
    /// df = factor.momentum(df, period=20)
    /// 
    /// # 对数收益率动量（适合长周期）
    /// df = factor.momentum(df, period=60, method="log")
    /// 
    /// # 动量加速度（捕捉趋势变化）
    /// df = factor.momentum(df, period=20, method="acceleration")
    /// ```
    #[pyo3(signature = (df, price_col="close", period=20, method="return", factor_col="momentum"))]
    pub fn momentum(&self, df: PyDataFrame, price_col: &str, period: usize, method: &str, factor_col: &str) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut momentum = vec![f64::NAN; len];
        
        match method {
            "return" => {
                // 简单收益率
                for i in period..len {
                    if let (Some(current), Some(past)) = (price_values.get(i), price_values.get(i - period)) {
                        if past != 0.0 {
                            momentum[i] = (current - past) / past;
                        }
                    }
                }
            },
            "log" => {
                // 对数收益率
                for i in period..len {
                    if let (Some(current), Some(past)) = (price_values.get(i), price_values.get(i - period)) {
                        if past > 0.0 && current > 0.0 {
                            momentum[i] = (current / past).ln();
                        }
                    }
                }
            },
            "residual" => {
                // 残差动量：当前动量 - 平均动量（去除市场整体趋势）
                let mut raw_momentum = vec![f64::NAN; len];
                for i in period..len {
                    if let (Some(current), Some(past)) = (price_values.get(i), price_values.get(i - period)) {
                        if past != 0.0 {
                            raw_momentum[i] = (current - past) / past;
                        }
                    }
                }
                
                // 计算平均动量
                let valid_momentum: Vec<f64> = raw_momentum.iter()
                    .filter(|&&x| !x.is_nan())
                    .copied()
                    .collect();
                
                if !valid_momentum.is_empty() {
                    let avg_momentum = valid_momentum.iter().sum::<f64>() / valid_momentum.len() as f64;
                    
                    // 残差 = 实际动量 - 平均动量
                    for i in period..len {
                        if !raw_momentum[i].is_nan() {
                            momentum[i] = raw_momentum[i] - avg_momentum;
                        }
                    }
                }
            },
            "acceleration" => {
                // 动量加速度：近期动量 - 远期动量
                let short_period = period / 2;
                if short_period < 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("周期太短，无法计算加速度"));
                }
                
                for i in period..len {
                    // 近期动量（最近short_period天）
                    if let (Some(current), Some(mid)) = (price_values.get(i), price_values.get(i - short_period)) {
                        if mid != 0.0 {
                            let short_mom = (current - mid) / mid;
                            
                            // 远期动量（period到short_period之间）
                            if let Some(past) = price_values.get(i - period) {
                                if past != 0.0 {
                                    let long_mom = (mid - past) / past;
                                    // 加速度 = 近期动量 - 远期动量
                                    momentum[i] = short_mom - long_mom;
                                }
                            }
                        }
                    }
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("未知的方法: {}，支持的方法：return, log, residual, acceleration", method)
                ));
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str(factor_col), momentum))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 反转因子
    /// 
    /// 短期反转效应：过去短期表现差的股票未来可能反转
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `period`: 回看周期（默认5天）
    /// 
    /// # 返回
    /// 包含reversal因子列的DataFrame（取负号，表现差的得分高）
    #[pyo3(signature = (df, price_col="close", period=5))]
    pub fn reversal(&self, df: PyDataFrame, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut reversal = vec![f64::NAN; len];
        
        for i in period..len {
            if let (Some(current), Some(past)) = (price_values.get(i), price_values.get(i - period)) {
                if past != 0.0 {
                    // 取负号：短期跌幅大的因子值高
                    reversal[i] = -(current - past) / past;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("reversal"), reversal))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 波动率因子
    /// 
    /// 计算过去N天的价格波动率（标准差）
    /// 
    /// # 参数
    /// - `df`: 包含收益率数据的DataFrame
    /// - `return_col`: 收益率列名（如果为None则从price_col计算）
    /// - `price_col`: 价格列名（用于计算收益率）
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含volatility因子列的DataFrame
    #[pyo3(signature = (df, return_col=None, price_col="close", period=20))]
    pub fn volatility(&self, df: PyDataFrame, return_col: Option<&str>, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        // 获取或计算收益率
        let returns = if let Some(ret_col) = return_col {
            data.column(ret_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
                .as_materialized_series()
                .clone()
        } else {
            // 从价格计算收益率
            let price = data.column(price_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
                .as_materialized_series()
                .clone();
            
            let price_values = price.f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
            
            let len = price_values.len();
            let mut returns_vec = vec![f64::NAN; len];
            
            for i in 1..len {
                if let (Some(current), Some(prev)) = (price_values.get(i), price_values.get(i - 1)) {
                    if prev != 0.0 {
                        returns_vec[i] = (current - prev) / prev;
                    }
                }
            }
            
            Series::new(PlSmallStr::from_str("returns"), returns_vec)
        };
        
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = return_values.len();
        let mut volatility = vec![f64::NAN; len];
        
        for i in period..len {
            let mut sum = 0.0;
            let mut count = 0;
            
            // 计算均值
            for j in (i - period)..i {
                if let Some(ret) = return_values.get(j) {
                    if !ret.is_nan() {
                        sum += ret;
                        count += 1;
                    }
                }
            }
            
            if count > 1 {
                let mean = sum / count as f64;
                
                // 计算标准差
                let mut variance = 0.0;
                for j in (i - period)..i {
                    if let Some(ret) = return_values.get(j) {
                        if !ret.is_nan() {
                            variance += (ret - mean).powi(2);
                        }
                    }
                }
                
                volatility[i] = (variance / count as f64).sqrt();
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("volatility"), volatility))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 成交量因子
    /// 
    /// 相对成交量：当前成交量 / 过去N天平均成交量
    /// 
    /// # 参数
    /// - `df`: 包含成交量数据的DataFrame
    /// - `volume_col`: 成交量列名（默认"volume"）
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含volume_factor因子列的DataFrame
    #[pyo3(signature = (df, volume_col="volume", period=20))]
    pub fn volume_factor(&self, df: PyDataFrame, volume_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let volume = data.column(volume_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取成交量列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let volume_values = volume.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = volume_values.len();
        let mut volume_factor = vec![f64::NAN; len];
        
        for i in period..len {
            let mut sum = 0.0;
            let mut count = 0;
            
            for j in (i - period)..i {
                if let Some(vol) = volume_values.get(j) {
                    if !vol.is_nan() && vol > 0.0 {
                        sum += vol;
                        count += 1;
                    }
                }
            }
            
            if count > 0 {
                if let Some(current_vol) = volume_values.get(i) {
                    let avg_volume = sum / count as f64;
                    if avg_volume > 0.0 {
                        volume_factor[i] = current_vol / avg_volume;
                    }
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("volume_factor"), volume_factor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 量价相关性因子
    /// 
    /// 计算价格变动与成交量的相关性
    /// 
    /// # 参数
    /// - `df`: 包含价格和成交量数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `volume_col`: 成交量列名
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含price_volume_corr因子列的DataFrame
    #[pyo3(signature = (df, price_col="close", volume_col="volume", period=20))]
    pub fn price_volume_corr(&self, df: PyDataFrame, price_col: &str, volume_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let volume = data.column(volume_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取成交量列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let volume_values = volume.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut corr_factor = vec![f64::NAN; len];
        
        for i in period..len {
            // 计算价格和成交量的均值
            let mut price_sum = 0.0;
            let mut volume_sum = 0.0;
            let mut count = 0;
            
            for j in (i - period)..i {
                if let (Some(p), Some(v)) = (price_values.get(j), volume_values.get(j)) {
                    if !p.is_nan() && !v.is_nan() {
                        price_sum += p;
                        volume_sum += v;
                        count += 1;
                    }
                }
            }
            
            if count > 1 {
                let price_mean = price_sum / count as f64;
                let volume_mean = volume_sum / count as f64;
                
                // 计算协方差和标准差
                let mut cov = 0.0;
                let mut price_var = 0.0;
                let mut volume_var = 0.0;
                
                for j in (i - period)..i {
                    if let (Some(p), Some(v)) = (price_values.get(j), volume_values.get(j)) {
                        if !p.is_nan() && !v.is_nan() {
                            let price_diff = p - price_mean;
                            let volume_diff = v - volume_mean;
                            cov += price_diff * volume_diff;
                            price_var += price_diff * price_diff;
                            volume_var += volume_diff * volume_diff;
                        }
                    }
                }
                
                let price_std = price_var.sqrt();
                let volume_std = volume_var.sqrt();
                
                if price_std > 0.0 && volume_std > 0.0 {
                    corr_factor[i] = cov / (price_std * volume_std);
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("price_volume_corr"), corr_factor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 价格加速度因子
    /// 
    /// 测量价格变化的加速度（二阶导数）
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `period`: 回看周期（默认10天）
    /// 
    /// # 返回
    /// 包含price_acceleration因子列的DataFrame
    #[pyo3(signature = (df, price_col="close", period=10))]
    pub fn price_acceleration(&self, df: PyDataFrame, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut acceleration = vec![f64::NAN; len];
        
        // 需要至少2*period的数据
        for i in (2 * period)..len {
            if let (Some(p_now), Some(p_mid), Some(p_past)) = (
                price_values.get(i),
                price_values.get(i - period),
                price_values.get(i - 2 * period),
            ) {
                if p_mid != 0.0 && p_past != 0.0 {
                    // 计算两段收益率
                    let ret1 = (p_mid - p_past) / p_past;
                    let ret2 = (p_now - p_mid) / p_mid;
                    // 加速度 = 收益率的变化
                    acceleration[i] = ret2 - ret1;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("price_acceleration"), acceleration))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 偏度因子
    /// 
    /// 收益率分布的偏度，正偏表示右尾厚，负偏表示左尾厚
    /// 
    /// # 参数
    /// - `df`: 包含价格或收益率数据的DataFrame
    /// - `return_col`: 收益率列名（如果为None则从price_col计算）
    /// - `price_col`: 价格列名
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含skewness因子列的DataFrame
    #[pyo3(signature = (df, return_col=None, price_col="close", period=20))]
    pub fn skewness(&self, df: PyDataFrame, return_col: Option<&str>, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        // 获取或计算收益率
        let returns = if let Some(ret_col) = return_col {
            data.column(ret_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
                .as_materialized_series()
                .clone()
        } else {
            let price = data.column(price_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
                .as_materialized_series()
                .clone();
            
            let price_values = price.f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
            
            let len = price_values.len();
            let mut returns_vec = vec![f64::NAN; len];
            
            for i in 1..len {
                if let (Some(current), Some(prev)) = (price_values.get(i), price_values.get(i - 1)) {
                    if prev != 0.0 {
                        returns_vec[i] = (current - prev) / prev;
                    }
                }
            }
            
            Series::new(PlSmallStr::from_str("returns"), returns_vec)
        };
        
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = return_values.len();
        let mut skewness = vec![f64::NAN; len];
        
        for i in period..len {
            let mut sum = 0.0;
            let mut count = 0;
            
            // 计算均值
            for j in (i - period)..i {
                if let Some(ret) = return_values.get(j) {
                    if !ret.is_nan() {
                        sum += ret;
                        count += 1;
                    }
                }
            }
            
            if count > 2 {
                let mean = sum / count as f64;
                
                // 计算标准差和偏度
                let mut m2 = 0.0;
                let mut m3 = 0.0;
                
                for j in (i - period)..i {
                    if let Some(ret) = return_values.get(j) {
                        if !ret.is_nan() {
                            let diff = ret - mean;
                            m2 += diff * diff;
                            m3 += diff * diff * diff;
                        }
                    }
                }
                
                let variance = m2 / count as f64;
                let std = variance.sqrt();
                
                if std > 0.0 {
                    // 偏度 = E[(X-μ)³] / σ³
                    skewness[i] = (m3 / count as f64) / (std * std * std);
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("skewness"), skewness))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 峰度因子
    /// 
    /// 收益率分布的峰度，衡量尾部厚度
    /// 
    /// # 参数
    /// - `df`: 包含价格或收益率数据的DataFrame
    /// - `return_col`: 收益率列名（如果为None则从price_col计算）
    /// - `price_col`: 价格列名
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含kurtosis因子列的DataFrame（超额峰度，正态分布为0）
    #[pyo3(signature = (df, return_col=None, price_col="close", period=20))]
    pub fn kurtosis(&self, df: PyDataFrame, return_col: Option<&str>, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        // 获取或计算收益率
        let returns = if let Some(ret_col) = return_col {
            data.column(ret_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
                .as_materialized_series()
                .clone()
        } else {
            let price = data.column(price_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
                .as_materialized_series()
                .clone();
            
            let price_values = price.f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
            
            let len = price_values.len();
            let mut returns_vec = vec![f64::NAN; len];
            
            for i in 1..len {
                if let (Some(current), Some(prev)) = (price_values.get(i), price_values.get(i - 1)) {
                    if prev != 0.0 {
                        returns_vec[i] = (current - prev) / prev;
                    }
                }
            }
            
            Series::new(PlSmallStr::from_str("returns"), returns_vec)
        };
        
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = return_values.len();
        let mut kurtosis = vec![f64::NAN; len];
        
        for i in period..len {
            let mut sum = 0.0;
            let mut count = 0;
            
            // 计算均值
            for j in (i - period)..i {
                if let Some(ret) = return_values.get(j) {
                    if !ret.is_nan() {
                        sum += ret;
                        count += 1;
                    }
                }
            }
            
            if count > 3 {
                let mean = sum / count as f64;
                
                // 计算标准差和峰度
                let mut m2 = 0.0;
                let mut m4 = 0.0;
                
                for j in (i - period)..i {
                    if let Some(ret) = return_values.get(j) {
                        if !ret.is_nan() {
                            let diff = ret - mean;
                            let diff2 = diff * diff;
                            m2 += diff2;
                            m4 += diff2 * diff2;
                        }
                    }
                }
                
                let variance = m2 / count as f64;
                
                if variance > 0.0 {
                    // 超额峰度 = E[(X-μ)⁴] / σ⁴ - 3
                    kurtosis[i] = (m4 / count as f64) / (variance * variance) - 3.0;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("kurtosis"), kurtosis))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 最大回撤因子
    /// 
    /// 计算过去N天的最大回撤
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含max_drawdown因子列的DataFrame（负值）
    #[pyo3(signature = (df, price_col="close", period=20))]
    pub fn max_drawdown(&self, df: PyDataFrame, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut max_dd = vec![f64::NAN; len];
        
        for i in period..len {
            let mut max_price = f64::NEG_INFINITY;
            let mut max_drawdown = 0.0;
            
            for j in (i - period)..=i {
                if let Some(p) = price_values.get(j) {
                    if !p.is_nan() {
                        if p > max_price {
                            max_price = p;
                        }
                        
                        if max_price > 0.0 {
                            let drawdown = (p - max_price) / max_price;
                            if drawdown < max_drawdown {
                                max_drawdown = drawdown;
                            }
                        }
                    }
                }
            }
            
            max_dd[i] = max_drawdown;
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("max_drawdown"), max_dd))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 换手率因子
    /// 
    /// 计算相对换手率：当前换手率 / 历史平均换手率
    /// 
    /// # 参数
    /// - `df`: 包含换手率数据的DataFrame
    /// - `turnover_col`: 换手率列名（默认"turnover"）
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含turnover_factor因子列的DataFrame
    #[pyo3(signature = (df, turnover_col="turnover", period=20))]
    pub fn turnover_factor(&self, df: PyDataFrame, turnover_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let turnover = data.column(turnover_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取换手率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let turnover_values = turnover.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = turnover_values.len();
        let mut turnover_factor = vec![f64::NAN; len];
        
        for i in period..len {
            let mut sum = 0.0;
            let mut count = 0;
            
            for j in (i - period)..i {
                if let Some(to) = turnover_values.get(j) {
                    if !to.is_nan() && to > 0.0 {
                        sum += to;
                        count += 1;
                    }
                }
            }
            
            if count > 0 {
                if let Some(current_to) = turnover_values.get(i) {
                    let avg_turnover = sum / count as f64;
                    if avg_turnover > 0.0 {
                        turnover_factor[i] = current_to / avg_turnover;
                    }
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("turnover_factor"), turnover_factor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 振幅因子
    /// 
    /// 计算相对振幅：当前振幅 / 历史平均振幅
    /// 
    /// # 参数
    /// - `df`: 包含最高价和最低价的DataFrame
    /// - `high_col`: 最高价列名（默认"high"）
    /// - `low_col`: 最低价列名（默认"low"）
    /// - `close_col`: 收盘价列名（默认"close"）
    /// - `period`: 回看周期（默认20天）
    /// 
    /// # 返回
    /// 包含amplitude_factor因子列的DataFrame
    #[pyo3(signature = (df, high_col="high", low_col="low", close_col="close", period=20))]
    pub fn amplitude_factor(&self, df: PyDataFrame, high_col: &str, low_col: &str, close_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let high = data.column(high_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取最高价列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let low = data.column(low_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取最低价列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let close = data.column(close_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收盘价列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let high_values = high.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let low_values = low.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let close_values = close.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = high_values.len();
        let mut amplitude_factor = vec![f64::NAN; len];
        
        for i in period..len {
            let mut sum_amp = 0.0;
            let mut count = 0;
            
            // 计算历史平均振幅
            for j in (i - period)..i {
                if let (Some(h), Some(l), Some(c)) = (high_values.get(j), low_values.get(j), close_values.get(j)) {
                    if !h.is_nan() && !l.is_nan() && !c.is_nan() && c > 0.0 {
                        let amp = (h - l) / c;
                        sum_amp += amp;
                        count += 1;
                    }
                }
            }
            
            if count > 0 {
                if let (Some(h), Some(l), Some(c)) = (high_values.get(i), low_values.get(i), close_values.get(i)) {
                    if !h.is_nan() && !l.is_nan() && !c.is_nan() && c > 0.0 {
                        let current_amp = (h - l) / c;
                        let avg_amp = sum_amp / count as f64;
                        if avg_amp > 0.0 {
                            amplitude_factor[i] = current_amp / avg_amp;
                        }
                    }
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("amplitude_factor"), amplitude_factor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 量价背离因子
    /// 
    /// 价格上涨但成交量下降（或反之）的程度
    /// 
    /// # 参数
    /// - `df`: 包含价格和成交量数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `volume_col`: 成交量列名
    /// - `period`: 回看周期（默认5天）
    /// 
    /// # 返回
    /// 包含price_volume_divergence因子列的DataFrame
    /// 正值表示价升量跌（看跌背离），负值表示价跌量升（看涨背离）
    #[pyo3(signature = (df, price_col="close", volume_col="volume", period=5))]
    pub fn price_volume_divergence(&self, df: PyDataFrame, price_col: &str, volume_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let volume = data.column(volume_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取成交量列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let volume_values = volume.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut divergence = vec![f64::NAN; len];
        
        for i in period..len {
            if let (Some(p_now), Some(p_past), Some(v_now), Some(v_past)) = (
                price_values.get(i),
                price_values.get(i - period),
                volume_values.get(i),
                volume_values.get(i - period),
            ) {
                if p_past > 0.0 && v_past > 0.0 {
                    let price_change = (p_now - p_past) / p_past;
                    let volume_change = (v_now - v_past) / v_past;
                    
                    // 背离度 = 价格变化 - 成交量变化
                    // 正值表示价升量跌，负值表示价跌量升
                    divergence[i] = price_change - volume_change;
                }
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("price_volume_divergence"), divergence))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// RSI相对强弱因子
    /// 
    /// 基于RSI指标的因子，标准化到[-1, 1]区间
    /// 
    /// # 参数
    /// - `df`: 包含价格数据的DataFrame
    /// - `price_col`: 价格列名
    /// - `period`: RSI周期（默认14天）
    /// 
    /// # 返回
    /// 包含rsi_factor因子列的DataFrame
    #[pyo3(signature = (df, price_col="close", period=14))]
    pub fn rsi_factor(&self, df: PyDataFrame, price_col: &str, period: usize) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let price = data.column(price_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取价格列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let price_values = price.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = price_values.len();
        let mut rsi_factor = vec![f64::NAN; len];
        
        for i in period..len {
            let mut gain_sum = 0.0;
            let mut loss_sum = 0.0;
            
            for j in (i - period + 1)..=i {
                if let (Some(current), Some(prev)) = (price_values.get(j), price_values.get(j - 1)) {
                    if !current.is_nan() && !prev.is_nan() {
                        let change = current - prev;
                        if change > 0.0 {
                            gain_sum += change;
                        } else {
                            loss_sum += -change;
                        }
                    }
                }
            }
            
            let avg_gain = gain_sum / period as f64;
            let avg_loss = loss_sum / period as f64;
            
            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                let rsi = 100.0 - (100.0 / (1.0 + rs));
                // 标准化到[-1, 1]: (RSI - 50) / 50
                rsi_factor[i] = (rsi - 50.0) / 50.0;
            } else if avg_gain > 0.0 {
                rsi_factor[i] = 1.0;
            } else {
                rsi_factor[i] = 0.0;
            }
        }
        
        data.with_column(Series::new(PlSmallStr::from_str("rsi_factor"), rsi_factor))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加因子列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    // ========================================================================
    // 因子评估方法 (Factor Evaluation Methods)
    // ========================================================================
    // 
    // 设计理念：约定优于配置
    // 
    // 使用方式：
    // 1. 计算因子时自动添加到DataFrame: df = factor.momentum(df, "close", 20)
    // 2. 添加未来收益: df = factor.return(df, "close", period=1)
    // 3. 评估时使用默认列名: ic = factor.ic(df)  # 默认使用最后一个因子列
    //
    // 或者显式指定：
    //    ic = factor.ic(df, factor_col="momentum", return_col="future_return")
    // ========================================================================

    /// IC值（信息系数）
    /// 
    /// IC = 因子值与未来收益率的Pearson相关系数
    /// 衡量因子对未来收益的线性预测能力
    /// 
    /// # 使用流程
    /// ```python
    /// # 方式1: 使用默认列名（最简洁）
    /// df = factor.momentum(df, "close", period=20)      # 生成 "momentum" 列
    /// df = factor.return(df, "close", period=1)         # 生成 "future_return" 列
    /// ic = factor.ic(df, "momentum", "future_return")   # 显式指定列名
    /// 
    /// # 方式2: 批量评估多个因子
    /// for col in ["momentum", "volatility", "rsi_factor"]:
    ///     ic = factor.ic(df, col, "future_return")
    ///     print(f"{col}: IC={ic:.4f}")
    /// ```
    /// 
    /// # 参数
    /// - `df`: 包含因子和收益率数据的DataFrame
    /// - `factor_col`: 因子列名（必需，因为可能有多个因子）
    /// - `return_col`: 未来收益率列名（默认"future_return"）
    /// 
    /// # 返回
    /// IC值（相关系数，范围[-1, 1]）
    /// 
    /// # 评价标准
    /// - |IC| > 0.03: 因子有效
    /// - |IC| > 0.05: 因子较强
    /// - |IC| > 0.08: 因子很强
    #[pyo3(signature = (df, factor_col, return_col="future_return"))]
    pub fn ic(&self, df: PyDataFrame, factor_col: &str, return_col: &str) -> PyResult<f64> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let returns = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        // 计算均值
        let mut factor_sum = 0.0;
        let mut return_sum = 0.0;
        let mut count = 0;
        
        for i in 0..factor_values.len() {
            if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                if !f.is_nan() && !r.is_nan() {
                    factor_sum += f;
                    return_sum += r;
                    count += 1;
                }
            }
        }
        
        if count < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("样本量不足"));
        }
        
        let factor_mean = factor_sum / count as f64;
        let return_mean = return_sum / count as f64;
        
        // 计算协方差和标准差
        let mut cov = 0.0;
        let mut factor_var = 0.0;
        let mut return_var = 0.0;
        
        for i in 0..factor_values.len() {
            if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                if !f.is_nan() && !r.is_nan() {
                    let factor_diff = f - factor_mean;
                    let return_diff = r - return_mean;
                    cov += factor_diff * return_diff;
                    factor_var += factor_diff * factor_diff;
                    return_var += return_diff * return_diff;
                }
            }
        }
        
        let factor_std = factor_var.sqrt();
        let return_std = return_var.sqrt();
        
        if factor_std == 0.0 || return_std == 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("标准差为0，无法计算相关系数"));
        }
        
        let ic = cov / (factor_std * return_std);
        Ok(ic)
    }

    /// IR值（信息比率）
    /// 
    /// IR = IC均值 / IC标准差
    /// 衡量因子收益的稳定性和持续性，IR越高表示因子越稳定
    /// 
    /// # 参数
    /// - `df`: 包含因子和收益率数据的DataFrame，需要有时间序列
    /// - `factor_col`: 因子列名
    /// - `return_col`: 未来收益率列名（默认"future_return"）
    /// - `period`: 滚动计算IC的窗口期（默认20）
    /// 
    /// # 返回
    /// IR值（IC的均值除以IC的标准差）
    /// 
    /// # 评价标准
    /// - IR > 0.5: 因子较稳定
    /// - IR > 1.0: 因子很稳定
    /// - IR > 2.0: 因子非常优秀
    #[pyo3(signature = (df, factor_col, return_col="future_return", period=20))]
    pub fn ir(&self, df: PyDataFrame, factor_col: &str, return_col: &str, period: usize) -> PyResult<f64> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let returns = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = factor_values.len();
        if len < period + 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("数据长度不足"));
        }
        
        // 滚动计算IC
        let mut ic_series: Vec<f64> = Vec::new();
        
        for i in period..len {
            let start = i - period;
            
            // 计算这个窗口的IC
            let mut factor_sum = 0.0;
            let mut return_sum = 0.0;
            let mut count = 0;
            
            for j in start..i {
                if let (Some(f), Some(r)) = (factor_values.get(j), return_values.get(j)) {
                    if !f.is_nan() && !r.is_nan() {
                        factor_sum += f;
                        return_sum += r;
                        count += 1;
                    }
                }
            }
            
            if count < 2 {
                continue;
            }
            
            let factor_mean = factor_sum / count as f64;
            let return_mean = return_sum / count as f64;
            
            let mut cov = 0.0;
            let mut factor_var = 0.0;
            let mut return_var = 0.0;
            
            for j in start..i {
                if let (Some(f), Some(r)) = (factor_values.get(j), return_values.get(j)) {
                    if !f.is_nan() && !r.is_nan() {
                        let factor_diff = f - factor_mean;
                        let return_diff = r - return_mean;
                        cov += factor_diff * return_diff;
                        factor_var += factor_diff * factor_diff;
                        return_var += return_diff * return_diff;
                    }
                }
            }
            
            let factor_std = factor_var.sqrt();
            let return_std = return_var.sqrt();
            
            if factor_std > 0.0 && return_std > 0.0 {
                let ic = cov / (factor_std * return_std);
                ic_series.push(ic);
            }
        }
        
        if ic_series.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("有效IC数量不足"));
        }
        
        // 计算IC的均值和标准差
        let ic_mean = ic_series.iter().sum::<f64>() / ic_series.len() as f64;
        let ic_var = ic_series.iter().map(|ic| (ic - ic_mean).powi(2)).sum::<f64>() / ic_series.len() as f64;
        let ic_std = ic_var.sqrt();
        
        if ic_std == 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("IC标准差为0"));
        }
        
        let ir = ic_mean / ic_std;
        Ok(ir)
    }

    /// Rank IC值
    /// 
    /// Rank IC = 因子排名与收益率排名的Spearman秩相关系数
    /// 对异常值更稳健，适合非线性关系的因子
    /// 
    /// # 参数
    /// - `df`: 包含因子和收益率数据的DataFrame
    /// - `factor_col`: 因子列名
    /// - `return_col`: 未来收益率列名（默认"future_return"）
    /// 
    /// # 返回
    /// Rank IC值（范围[-1, 1]）
    /// 
    /// # 说明
    /// Rank IC通常比IC更稳定，因为它只关注排序关系而非绝对值
    #[pyo3(signature = (df, factor_col, return_col="future_return"))]
    pub fn rank_ic(&self, df: PyDataFrame, factor_col: &str, return_col: &str) -> PyResult<f64> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let returns = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        // 提取有效数据
        let mut valid_data: Vec<(f64, f64)> = Vec::new();
        for i in 0..factor_values.len() {
            if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                if !f.is_nan() && !r.is_nan() {
                    valid_data.push((f, r));
                }
            }
        }
        
        if valid_data.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("样本量不足"));
        }
        
        // 计算因子排名
        let mut factor_ranks = vec![0.0; valid_data.len()];
        for i in 0..valid_data.len() {
            let mut rank = 1.0;
            for j in 0..valid_data.len() {
                if valid_data[j].0 > valid_data[i].0 {
                    rank += 1.0;
                }
            }
            factor_ranks[i] = rank;
        }
        
        // 计算收益率排名
        let mut return_ranks = vec![0.0; valid_data.len()];
        for i in 0..valid_data.len() {
            let mut rank = 1.0;
            for j in 0..valid_data.len() {
                if valid_data[j].1 > valid_data[i].1 {
                    rank += 1.0;
                }
            }
            return_ranks[i] = rank;
        }
        
        // 计算排名相关系数
        let n = valid_data.len() as f64;
        let rank_mean = (n + 1.0) / 2.0;
        
        let mut cov = 0.0;
        let mut factor_var = 0.0;
        let mut return_var = 0.0;
        
        for i in 0..valid_data.len() {
            let factor_diff = factor_ranks[i] - rank_mean;
            let return_diff = return_ranks[i] - rank_mean;
            cov += factor_diff * return_diff;
            factor_var += factor_diff * factor_diff;
            return_var += return_diff * return_diff;
        }
        
        let rank_ic = cov / (factor_var.sqrt() * return_var.sqrt());
        Ok(rank_ic)
    }

    /// 分层回测（多空组合收益分析）
    /// 
    /// 将因子分成N层，计算每层的平均收益率
    /// 用于检验因子的单调性和区分度
    /// 
    /// # 参数
    /// - `df`: 包含因子和收益率数据的DataFrame
    /// - `factor_col`: 因子列名
    /// - `return_col`: 未来收益率列名（默认"future_return"）
    /// - `n_quantiles`: 分层数量（默认5）
    /// 
    /// # 返回
    /// 包含quantile和mean_return列的DataFrame
    /// 
    /// # 评价标准
    /// - 单调性：分层收益应呈现单调递增/递减
    /// - 多空收益：最高层与最低层收益差（Long-Short Return）
    /// - 区分度：各层收益差异越大越好
    #[pyo3(signature = (df, factor_col, return_col="future_return", n_quantiles=5))]
    pub fn quantile(&self, df: PyDataFrame, factor_col: &str, return_col: &str, n_quantiles: usize) -> PyResult<PyDataFrame> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let returns = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        // 提取有效数据
        let mut valid_data: Vec<(f64, f64)> = Vec::new();
        for i in 0..factor_values.len() {
            if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                if !f.is_nan() && !r.is_nan() {
                    valid_data.push((f, r));
                }
            }
        }
        
        if valid_data.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("无有效数据"));
        }
        
        // 按因子值排序
        valid_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // 分层
        let per_quantile = valid_data.len() / n_quantiles;
        let mut quantile_returns: Vec<f64> = Vec::new();
        let mut quantile_labels: Vec<i32> = Vec::new();
        
        for q in 0..n_quantiles {
            let start = q * per_quantile;
            let end = if q == n_quantiles - 1 {
                valid_data.len()
            } else {
                (q + 1) * per_quantile
            };
            
            let mut sum = 0.0;
            let mut count = 0;
            
            for i in start..end {
                sum += valid_data[i].1;
                count += 1;
            }
            
            if count > 0 {
                quantile_labels.push((q + 1) as i32);
                quantile_returns.push(sum / count as f64);
            }
        }
        
        let result = DataFrame::new(vec![
            Series::new(PlSmallStr::from_str("quantile"), quantile_labels).into(),
            Series::new(PlSmallStr::from_str("mean_return"), quantile_returns).into(),
        ]).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果DataFrame失败: {}", e)))?;
        
        Ok(PyDataFrame(result))
    }

    /// 因子覆盖度
    /// 
    /// 覆盖度 = 有效因子值数量 / 总样本数量
    /// 
    /// # 参数
    /// - `df`: 包含因子数据的DataFrame
    /// - `factor_col`: 因子列名
    /// 
    /// # 返回
    /// 覆盖度（0-1之间的比例）
    /// 
    /// # 评价标准
    /// - 覆盖度 > 0.8: 较好
    /// - 覆盖度 > 0.9: 优秀
    #[pyo3(signature = (df, factor_col))]
    pub fn coverage(&self, df: PyDataFrame, factor_col: &str) -> PyResult<f64> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let total = factor_values.len();
        let mut valid_count = 0;
        
        for i in 0..total {
            if let Some(val) = factor_values.get(i) {
                if !val.is_nan() && !val.is_infinite() {
                    valid_count += 1;
                }
            }
        }
        
        Ok(valid_count as f64 / total as f64)
    }

    /// IC胜率
    /// 
    /// IC胜率 = IC>0的次数 / 总次数
    /// 衡量因子预测方向的准确率
    /// 
    /// # 参数
    /// - `df`: 包含因子和收益率数据的DataFrame
    /// - `factor_col`: 因子列名
    /// - `return_col`: 未来收益率列名（默认"future_return"）
    /// - `period`: 滚动窗口（默认20）
    /// 
    /// # 返回
    /// IC胜率（0-1之间的比例）
    /// 
    /// # 评价标准
    /// - IC胜率 > 0.5: 因子有预测能力
    /// - IC胜率 > 0.6: 因子较强
    /// - IC胜率 > 0.7: 因子很强
    #[pyo3(signature = (df, factor_col, return_col="future_return", period=20))]
    pub fn ic_win_rate(&self, df: PyDataFrame, factor_col: &str, return_col: &str, period: usize) -> PyResult<f64> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let returns = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        let len = factor_values.len();
        if len < period + 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("数据长度不足"));
        }
        
        let mut positive_ic_count = 0;
        let mut total_ic_count = 0;
        
        // 滚动计算IC
        for i in period..len {
            let start = i - period;
            
            let mut factor_sum = 0.0;
            let mut return_sum = 0.0;
            let mut count = 0;
            
            for j in start..i {
                if let (Some(f), Some(r)) = (factor_values.get(j), return_values.get(j)) {
                    if !f.is_nan() && !r.is_nan() {
                        factor_sum += f;
                        return_sum += r;
                        count += 1;
                    }
                }
            }
            
            if count < 2 {
                continue;
            }
            
            let factor_mean = factor_sum / count as f64;
            let return_mean = return_sum / count as f64;
            
            let mut cov = 0.0;
            let mut factor_var = 0.0;
            let mut return_var = 0.0;
            
            for j in start..i {
                if let (Some(f), Some(r)) = (factor_values.get(j), return_values.get(j)) {
                    if !f.is_nan() && !r.is_nan() {
                        let factor_diff = f - factor_mean;
                        let return_diff = r - return_mean;
                        cov += factor_diff * return_diff;
                        factor_var += factor_diff * factor_diff;
                        return_var += return_diff * return_diff;
                    }
                }
            }
            
            let factor_std = factor_var.sqrt();
            let return_std = return_var.sqrt();
            
            if factor_std > 0.0 && return_std > 0.0 {
                let ic = cov / (factor_std * return_std);
                total_ic_count += 1;
                if ic > 0.0 {
                    positive_ic_count += 1;
                }
            }
        }
        
        if total_ic_count == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("无有效IC"));
        }
        
        Ok(positive_ic_count as f64 / total_ic_count as f64)
    }

    /// 多空收益（Long-Short Return）
    /// 
    /// 多空收益 = 最高分位收益 - 最低分位收益
    /// 衡量因子的盈利能力
    /// 
    /// # 参数
    /// - `df`: 包含因子和收益率数据的DataFrame
    /// - `factor_col`: 因子列名
    /// - `return_col`: 未来收益率列名（默认"future_return"）
    /// - `n_quantiles`: 分层数量（默认5）
    /// 
    /// # 返回
    /// 多空收益（最高层收益 - 最低层收益）
    #[pyo3(signature = (df, factor_col, return_col="future_return", n_quantiles=5))]
    pub fn long_short(&self, df: PyDataFrame, factor_col: &str, return_col: &str, n_quantiles: usize) -> PyResult<f64> {
        let data: DataFrame = df.into();
        
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let returns = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益率列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let return_values = returns.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        // 提取有效数据
        let mut valid_data: Vec<(f64, f64)> = Vec::new();
        for i in 0..factor_values.len() {
            if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                if !f.is_nan() && !r.is_nan() {
                    valid_data.push((f, r));
                }
            }
        }
        
        if valid_data.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("无有效数据"));
        }
        
        // 按因子值排序
        valid_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // 计算最高层和最低层的平均收益
        let per_quantile = valid_data.len() / n_quantiles;
        
        // 最低层（第1层）
        let mut low_sum = 0.0;
        let mut low_count = 0;
        for i in 0..per_quantile {
            low_sum += valid_data[i].1;
            low_count += 1;
        }
        
        // 最高层（第N层）
        let high_start = (n_quantiles - 1) * per_quantile;
        let mut high_sum = 0.0;
        let mut high_count = 0;
        for i in high_start..valid_data.len() {
            high_sum += valid_data[i].1;
            high_count += 1;
        }
        
        if low_count == 0 || high_count == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("分层数据不足"));
        }
        
        let low_return = low_sum / low_count as f64;
        let high_return = high_sum / high_count as f64;
        
        Ok(high_return - low_return)
    }

    /// 因子换手率
    /// 
    /// 换手率 = 相邻两期分层组合中股票变化的比例
    /// 衡量因子的稳定性，换手率越低表示因子越稳定
    /// 
    /// # 参数
    /// - `df`: 包含因子数据的DataFrame（需要包含时间列和分组列）
    /// - `factor_col`: 因子列名
    /// - `date_col`: 时间列名（默认"date"）
    /// - `group_col`: 分组列名（如股票代码，默认"symbol"）
    /// - `n_quantiles`: 分层数量（默认5）
    /// 
    /// # 返回
    /// 平均换手率（0-2之间，单边换手率）
    /// 
    /// # 评价标准
    /// - 换手率 < 0.3: 因子很稳定
    /// - 换手率 < 0.5: 因子较稳定
    /// - 换手率 > 0.7: 因子不稳定
    /// 
    /// # 说明
    /// 换手率的计算方法：
    /// 1. 对每个时间截面，按因子值将股票分为n_quantiles组
    /// 2. 对于相邻两期，计算每组中股票的变化比例
    /// 3. 换手率 = (新增股票数 + 减少股票数) / (2 * 组内股票总数)
    #[pyo3(signature = (df, factor_col, date_col="date", group_col="symbol", n_quantiles=5))]
    pub fn turnover(&self, df: PyDataFrame, factor_col: &str, date_col: &str, group_col: &str, n_quantiles: usize) -> PyResult<f64> {
        use std::collections::{HashMap, HashSet};
        
        let data: DataFrame = df.into();
        
        // 获取所需的列
        let factor = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let dates = data.column(date_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取日期列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let groups = data.column(group_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取分组列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let factor_values = factor.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("因子列转换失败: {}", e)))?;
        
        let group_values = groups.str()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组列转换失败: {}", e)))?;
        
        // 按日期分组数据
        let mut date_data: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        
        for i in 0..factor_values.len() {
            if let (Some(factor_val), Some(group_val)) = (factor_values.get(i), group_values.get(i)) {
                if !factor_val.is_nan() {
                    let date_str = dates.get(i).unwrap().to_string();
                    date_data.entry(date_str)
                        .or_insert_with(Vec::new)
                        .push((group_val.to_string(), factor_val));
                }
            }
        }
        
        // 对每个日期的数据按因子值排序并分组
        let mut sorted_dates: Vec<String> = date_data.keys().cloned().collect();
        sorted_dates.sort();
        
        if sorted_dates.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("至少需要2个时间截面"));
        }
        
        let mut all_turnovers = Vec::new();
        
        // 计算相邻两期的换手率
        for i in 0..(sorted_dates.len() - 1) {
            let curr_date = &sorted_dates[i];
            let next_date = &sorted_dates[i + 1];
            
            let curr_data = &date_data[curr_date];
            let next_data = &date_data[next_date];
            
            // 对当期数据按因子值排序
            let mut curr_sorted = curr_data.clone();
            curr_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut next_sorted = next_data.clone();
            next_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let curr_size = curr_sorted.len();
            let next_size = next_sorted.len();
            
            if curr_size < n_quantiles || next_size < n_quantiles {
                continue;
            }
            
            // 计算每个分位数的换手率
            for q in 0..n_quantiles {
                // 计算当前分位的索引范围
                let curr_start = (q * curr_size) / n_quantiles;
                let curr_end = ((q + 1) * curr_size) / n_quantiles;
                
                let next_start = (q * next_size) / n_quantiles;
                let next_end = ((q + 1) * next_size) / n_quantiles;
                
                // 获取当期和下期该分位的股票集合
                let curr_stocks: HashSet<String> = curr_sorted[curr_start..curr_end]
                    .iter()
                    .map(|(s, _)| s.clone())
                    .collect();
                
                let next_stocks: HashSet<String> = next_sorted[next_start..next_end]
                    .iter()
                    .map(|(s, _)| s.clone())
                    .collect();
                
                // 计算换手率：(新增 + 减少) / (2 * 平均持仓)
                let added = next_stocks.difference(&curr_stocks).count();
                let removed = curr_stocks.difference(&next_stocks).count();
                let avg_size = (curr_stocks.len() + next_stocks.len()) as f64 / 2.0;
                
                if avg_size > 0.0 {
                    let turnover_rate = (added + removed) as f64 / (2.0 * avg_size);
                    all_turnovers.push(turnover_rate);
                }
            }
        }
        
        if all_turnovers.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("无法计算换手率"));
        }
        
        // 返回平均换手率
        let avg_turnover = all_turnovers.iter().sum::<f64>() / all_turnovers.len() as f64;
        Ok(avg_turnover)
    }
}
