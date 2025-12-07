use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

// 辅助函数：计算Pearson相关系数
fn calculate_pearson_corr(x: &Series, y: &Series) -> PyResult<f64> {
    let x_f64 = x.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
    let y_f64 = y.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut count = 0;

    for i in 0..x_f64.len() {
        if let (Some(xi), Some(yi)) = (x_f64.get(i), y_f64.get(i)) {
            if !xi.is_nan() && !yi.is_nan() {
                sum_x += xi;
                sum_y += yi;
                sum_xy += xi * yi;
                sum_x2 += xi * xi;
                sum_y2 += yi * yi;
                count += 1;
            }
        }
    }

    if count < 2 {
        return Ok(f64::NAN);
    }

    let n = count as f64;
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(numerator / denominator)
    }
}

// 辅助函数：计算Spearman秩相关系数
fn calculate_spearman_corr(x: &Series, y: &Series) -> PyResult<f64> {
    let x_f64 = x.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
    let y_f64 = y.f64().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

    // 收集有效数据对
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for i in 0..x_f64.len() {
        if let (Some(xi), Some(yi)) = (x_f64.get(i), y_f64.get(i)) {
            if !xi.is_nan() && !yi.is_nan() {
                pairs.push((xi, yi));
            }
        }
    }

    if pairs.len() < 2 {
        return Ok(f64::NAN);
    }

    // 计算x的排名
    let mut x_ranks = vec![0.0; pairs.len()];
    let mut x_sorted: Vec<(usize, f64)> = pairs.iter().enumerate().map(|(i, (x, _))| (i, *x)).collect();
    x_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, (idx, _)) in x_sorted.iter().enumerate() {
        x_ranks[*idx] = (rank + 1) as f64;
    }

    // 计算y的排名
    let mut y_ranks = vec![0.0; pairs.len()];
    let mut y_sorted: Vec<(usize, f64)> = pairs.iter().enumerate().map(|(i, (_, y))| (i, *y)).collect();
    y_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, (idx, _)) in y_sorted.iter().enumerate() {
        y_ranks[*idx] = (rank + 1) as f64;
    }

    // 计算排名之间的Pearson相关系数
    let n = pairs.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..pairs.len() {
        let xi = x_ranks[i];
        let yi = y_ranks[i];
        sum_x += xi;
        sum_y += yi;
        sum_xy += xi * yi;
        sum_x2 += xi * xi;
        sum_y2 += yi * yi;
    }

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(numerator / denominator)
    }
}

// 辅助函数：因子正交化（施密特正交化）
fn orthogonalize_factors(
    data: DataFrame,
    factor_cols: Vec<String>,
    prefix: &str,
) -> PyResult<DataFrame> {
    use crate::data::linear;
    
    if factor_cols.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("因子列表不能为空"));
    }

    let mut result = data;

    // 第一个因子保持不变
    let first_col = &factor_cols[0];
    let first_orth = result.column(first_col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", first_col, e)))?
        .clone();
    
    let first_orth_name = format!("{}{}", prefix, first_col);
    let first_orth_renamed = first_orth.with_name(PlSmallStr::from(first_orth_name.as_str()));
    result = result.with_column(first_orth_renamed)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
        .clone();

    // 逐个正交化后续因子
    for i in 1..factor_cols.len() {
        let current_col = &factor_cols[i];
        
        // 收集已正交化的因子作为自变量
        let x_cols: Vec<String> = (0..i)
            .map(|j| format!("{}{}", prefix, factor_cols[j]))
            .collect();

        // 对当前因子进行回归，取残差
        let (regressed_df, _) = linear(
            PyDataFrame(result.clone()),
            x_cols,
            current_col.as_str(),
            Some("_pred"),
            Some("_resid"),
            false,
        )?;

        result = regressed_df.0;

        // 将残差重命名为正交化后的因子
        let resid_col = result.column("_resid")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取残差列失败: {}", e)))?
            .clone();

        let orth_name = format!("{}{}", prefix, current_col);
        let resid_renamed = resid_col.with_name(PlSmallStr::from(orth_name.as_str()));
        result = result.with_column(resid_renamed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
            .clone();

        // 删除临时列
        result = result.drop("_pred")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除列失败: {}", e)))?;
        result = result.drop("_resid")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除列失败: {}", e)))?;
    }

    Ok(result)
}

#[pyclass]
pub struct Factor;

#[pymethods]
impl Factor {
    #[new]
    pub fn new() -> Self {
        Factor
    }

    #[pyo3(signature = (df, numerator_col, denominator_col, factor_col=None, handle_zero="nan"))]
    pub fn ratio(&self, df: PyDataFrame, numerator_col: &str, denominator_col: &str, 
                 factor_col: Option<&str>, handle_zero: &str) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let numerator = data.column(numerator_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取分子列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let denominator = data.column(denominator_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取分母列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let num_values = numerator.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分子列类型错误: {}", e)))?;
        
        let den_values = denominator.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分母列类型错误: {}", e)))?;
        
        let len = num_values.len();
        let mut ratio_values = vec![f64::NAN; len];
        
        for i in 0..len {
            if let (Some(num), Some(den)) = (num_values.get(i), den_values.get(i)) {
                ratio_values[i] = match handle_zero {
                    "nan" => if den != 0.0 { num / den } else { f64::NAN },
                    "inf" => num / den,
                    "skip" => if den != 0.0 { num / den } else { num },
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("不支持的 handle_zero 参数: {}", handle_zero)
                    ))
                };
            }
        }
        
        let default_name = format!("{}_{}_ratio", numerator_col, denominator_col);
        let col_name = factor_col.unwrap_or(&default_name);
        let ratio_series = Series::new(col_name.into(), ratio_values);
        data.with_column(ratio_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    #[pyo3(signature = (df, col1, col2, factor_col=None, normalize=false))]
    pub fn diff(&self, df: PyDataFrame, col1: &str, col2: &str, 
                factor_col: Option<&str>, normalize: bool) -> PyResult<PyDataFrame> {
        let mut data: DataFrame = df.into();
        
        let series1 = data.column(col1)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取第一列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let series2 = data.column(col2)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取第二列失败: {}", e)))?
            .as_materialized_series()
            .clone();
        
        let values1 = series1.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("第一列类型错误: {}", e)))?;
        
        let values2 = series2.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("第二列类型错误: {}", e)))?;
        
        let len = values1.len();
        let mut diff_values = vec![f64::NAN; len];
        
        for i in 0..len {
            if let (Some(v1), Some(v2)) = (values1.get(i), values2.get(i)) {
                diff_values[i] = if normalize {
                    if v2 != 0.0 { (v1 - v2) / v2 } else { f64::NAN }
                } else {
                    v1 - v2
                };
            }
        }
        
        let default_name = format!("{}_{}_diff", col1, col2);
        let col_name = factor_col.unwrap_or(&default_name);
        let diff_series = Series::new(col_name.into(), diff_values);
        data.with_column(diff_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 加权因子计算
    /// 
    /// 计算加权平均值，常用于市值加权、成交量加权等场景
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `value_col` - 数值列名
    /// * `weight_col` - 权重列名
    /// * `factor_col` - 输出因子列名（可选）
    /// * `group_cols` - 分组列名列表（可选，用于分组加权）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含加权因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 计算市值加权PE
    /// df = factor.weighted(df, "pe_ratio", "market_cap", "weighted_pe")
    /// 
    /// # 按行业分组计算市值加权PE
    /// df = factor.weighted(df, "pe_ratio", "market_cap", "weighted_pe", ["industry"])
    /// ```
    pub fn weighted(
        &self,
        df: PyDataFrame,
        value_col: &str,
        weight_col: &str,
        factor_col: Option<&str>,
        group_cols: Option<Vec<String>>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;

        let default_name = format!("{}_{}_weighted", value_col, weight_col);
        let col_name = factor_col.unwrap_or(&default_name);

        // 获取数值列和权重列
        let value_series = data.column(value_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", value_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        let weight_series = data.column(weight_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", weight_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let values = value_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;
        let weights = weight_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取权重失败: {}", e)))?;

        let len = values.len();
        let mut weighted_values = vec![0.0; len];

        if let Some(_groups) = group_cols {
            // TODO: 实现分组加权计算（需要更复杂的逻辑）
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("分组加权计算暂未实现"));
        } else {
            // 全局加权计算
            let mut sum_weighted = 0.0;
            let mut sum_weight = 0.0;

            for i in 0..len {
                if let (Some(v), Some(w)) = (values.get(i), weights.get(i)) {
                    if !v.is_nan() && !w.is_nan() {
                        sum_weighted += v * w;
                        sum_weight += w;
                    }
                }
            }

            let weighted_value = if sum_weight > 0.0 {
                sum_weighted / sum_weight
            } else {
                f64::NAN
            };

            // 所有行使用相同的加权值
            for i in 0..len {
                weighted_values[i] = weighted_value;
            }
        }

        let weighted_series = Series::new(col_name.into(), weighted_values);
        data.with_column(weighted_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 标准化因子
    /// 
    /// 对因子进行标准化处理，支持多种标准化方法
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 需要标准化的列名
    /// * `method` - 标准化方法: "zscore"(默认), "minmax", "rank", "quantile"
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含标准化因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # Z-Score标准化
    /// df = factor.normalize(df, "pe_ratio", "zscore", "pe_normalized")
    /// 
    /// # 最小-最大标准化
    /// df = factor.normalize(df, "pb_ratio", "minmax", "pb_normalized")
    /// ```
    pub fn normalize(
        &self,
        df: PyDataFrame,
        col: &str,
        method: Option<&str>,
        factor_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;
        let method = method.unwrap_or("zscore");

        let default_name = format!("{}_{}_normalized", col, method);
        let col_name = factor_col.unwrap_or(&default_name);

        // 获取列数据
        let series = data.column(col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let values = series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let len = values.len();
        let mut normalized_values = vec![0.0; len];

        match method {
            "zscore" => {
                // 计算均值和标准差
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if !v.is_nan() {
                            sum += v;
                            count += 1;
                        }
                    }
                }
                let mean = if count > 0 { sum / count as f64 } else { 0.0 };

                let mut sum_sq = 0.0;
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if !v.is_nan() {
                            sum_sq += (v - mean).powi(2);
                        }
                    }
                }
                let std = if count > 1 { (sum_sq / (count - 1) as f64).sqrt() } else { 1.0 };

                // 标准化
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        normalized_values[i] = if !v.is_nan() && std > 0.0 {
                            (v - mean) / std
                        } else {
                            f64::NAN
                        };
                    }
                }
            },
            "minmax" => {
                // 计算最小值和最大值
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if !v.is_nan() {
                            if v < min_val { min_val = v; }
                            if v > max_val { max_val = v; }
                        }
                    }
                }

                let range = max_val - min_val;

                // 标准化
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        normalized_values[i] = if !v.is_nan() && range > 0.0 {
                            (v - min_val) / range
                        } else {
                            f64::NAN
                        };
                    }
                }
            },
            "rank" => {
                // 转换为排名（1到N）
                let mut pairs: Vec<(usize, f64)> = Vec::new();
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if !v.is_nan() {
                            pairs.push((i, v));
                        }
                    }
                }
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                for (rank, (idx, _)) in pairs.iter().enumerate() {
                    normalized_values[*idx] = (rank + 1) as f64;
                }

                // 将NaN保持为NaN
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if v.is_nan() {
                            normalized_values[i] = f64::NAN;
                        }
                    }
                }
            },
            "quantile" => {
                // 转换为分位数（0到1）
                let mut pairs: Vec<(usize, f64)> = Vec::new();
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if !v.is_nan() {
                            pairs.push((i, v));
                        }
                    }
                }
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let count = pairs.len();
                for (rank, (idx, _)) in pairs.iter().enumerate() {
                    normalized_values[*idx] = if count > 1 {
                        rank as f64 / (count - 1) as f64
                    } else {
                        0.5
                    };
                }

                // 将NaN保持为NaN
                for i in 0..len {
                    if let Some(v) = values.get(i) {
                        if v.is_nan() {
                            normalized_values[i] = f64::NAN;
                        }
                    }
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("不支持的标准化方法: {}", method)
                ));
            }
        }

        let normalized_series = Series::new(col_name.into(), normalized_values);
        data.with_column(normalized_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 排名因子
    /// 
    /// 对因子进行排名，支持分组排名和升降序
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 需要排名的列名
    /// * `factor_col` - 输出因子列名（可选）
    /// * `ascending` - 是否升序排名（True: 小值排名低, False: 大值排名低）
    /// * `pct` - 是否返回百分比排名（0-1之间）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含排名因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 按PE从小到大排名
    /// df = factor.rank(df, "pe_ratio", "pe_rank", True, False)
    /// 
    /// # 按市值从大到小排名（百分比）
    /// df = factor.rank(df, "market_cap", "cap_rank", False, True)
    /// ```
    pub fn rank(
        &self,
        df: PyDataFrame,
        col: &str,
        factor_col: Option<&str>,
        ascending: Option<bool>,
        pct: Option<bool>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;
        let ascending = ascending.unwrap_or(true);
        let pct = pct.unwrap_or(false);

        let default_name = format!("{}_rank", col);
        let col_name = factor_col.unwrap_or(&default_name);

        // 获取列数据
        let series = data.column(col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let values = series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let len = values.len();
        let mut rank_values = vec![f64::NAN; len];

        // 收集非NaN值和索引
        let mut pairs: Vec<(usize, f64)> = Vec::new();
        for i in 0..len {
            if let Some(v) = values.get(i) {
                if !v.is_nan() {
                    pairs.push((i, v));
                }
            }
        }

        // 排序
        if ascending {
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // 分配排名
        let count = pairs.len();
        for (rank, (idx, _)) in pairs.iter().enumerate() {
            rank_values[*idx] = if pct {
                if count > 1 {
                    rank as f64 / (count - 1) as f64
                } else {
                    0.5
                }
            } else {
                (rank + 1) as f64
            };
        }

        let rank_series = Series::new(col_name.into(), rank_values);
        data.with_column(rank_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 移动平均因子
    /// 
    /// 计算滑动窗口的移动平均值
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 列名
    /// * `window` - 窗口大小
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含移动平均因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 计算20日移动平均
    /// df = factor.moving_average(df, "close", 20, "ma20")
    /// ```
    /// 移动平均因子
    /// 
    /// 计算简单移动平均
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 列名
    /// * `window` - 窗口大小
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含移动平均因子的数据框
    pub fn moving_average(
        &self,
        df: PyDataFrame,
        col: &str,
        window: usize,
        factor_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        use crate::talib::calculate_sma;
        
        let mut data = df.0;

        let default_name = format!("{}_ma{}", col, window);
        let col_name = factor_col.unwrap_or(&default_name);

        let series = data.column(col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .as_materialized_series()
            .clone();

        // 使用talib的SMA计算
        let ma_series = calculate_sma(&series, window)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算MA失败: {}", e)))?
            .with_name(PlSmallStr::from(col_name));

        data = data.with_column(ma_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
            .clone();
        
        Ok(PyDataFrame(data))
    }

    /// 动量因子
    /// 
    /// 计算收益率动量（当前值相对于N期前的变化率）
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 列名
    /// * `period` - 回看期数
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含动量因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 计算20日动量
    /// df = factor.momentum(df, "close", 20, "mom20")
    /// ```
    pub fn momentum(
        &self,
        df: PyDataFrame,
        col: &str,
        period: usize,
        factor_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;

        let default_name = format!("{}_mom{}", col, period);
        let col_name = factor_col.unwrap_or(&default_name);

        let series = data.column(col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let values = series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let len = values.len();
        let mut mom_values = vec![f64::NAN; len];

        for i in period..len {
            if let (Some(current), Some(previous)) = (values.get(i), values.get(i - period)) {
                if !current.is_nan() && !previous.is_nan() && previous != 0.0 {
                    mom_values[i] = (current - previous) / previous;
                }
            }
        }

        let mom_series = Series::new(col_name.into(), mom_values);
        data.with_column(mom_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// 波动率因子
    /// 
    /// 计算滑动窗口的收益率标准差
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 列名
    /// * `window` - 窗口大小
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含波动率因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 计算20日波动率
    /// df = factor.volatility(df, "returns", 20, "vol20")
    /// ```
    /// 波动率因子
    /// 
    /// 计算滑动窗口的标准差
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 列名
    /// * `window` - 窗口大小
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含波动率因子的数据框
    pub fn volatility(
        &self,
        df: PyDataFrame,
        col: &str,
        window: usize,
        factor_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;

        let default_name = format!("{}_vol{}", col, window);
        let col_name = factor_col.unwrap_or(&default_name);
        let column_name = col.to_string();

        // 使用Polars的rolling_std计算波动率
        let vol_series = data.clone().lazy()
            .select([
                polars::prelude::col(&column_name)
                    .cast(DataType::Float64)
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: window,
                        min_periods: window,
                        ..Default::default()
                    })
                    .alias(col_name)
            ])
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算波动率失败: {}", e)))?
            .column(col_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取列失败: {}", e)))?
            .clone();

        data = data.with_column(vol_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
            .clone();
        
        Ok(PyDataFrame(data))
    }

    /// 偏度因子
    /// 
    /// 计算滑动窗口的收益率分布偏度
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 列名
    /// * `window` - 窗口大小
    /// * `factor_col` - 输出因子列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含偏度因子的数据框
    pub fn skewness(
        &self,
        df: PyDataFrame,
        col: &str,
        window: usize,
        factor_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;

        let default_name = format!("{}_skew{}", col, window);
        let col_name = factor_col.unwrap_or(&default_name);

        // 使用Polars的rolling计算偏度（手动计算，因为Polars可能没有内置rolling_skew）
        let series = data.column(col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let values = series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let len = values.len();
        let mut skew_values = vec![f64::NAN; len];

        for i in 0..len {
            if i + 1 >= window {
                let mut vals = Vec::new();
                
                // 收集窗口内的有效值
                for j in (i + 1 - window)..=i {
                    if let Some(v) = values.get(j) {
                        if !v.is_nan() {
                            vals.push(v);
                        }
                    }
                }
                
                if vals.len() > 2 {
                    let n = vals.len() as f64;
                    let mean: f64 = vals.iter().sum::<f64>() / n;
                    let m2: f64 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>();
                    let m3: f64 = vals.iter().map(|&v| (v - mean).powi(3)).sum::<f64>();
                    
                    let variance = m2 / n;
                    let std = variance.sqrt();
                    
                    if std > 0.0 {
                        skew_values[i] = (m3 / n) / std.powi(3);
                    }
                }
            }
        }

        let skew_series = Series::new(PlSmallStr::from(col_name), skew_values);
        data = data.with_column(skew_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
            .clone();
        
        Ok(PyDataFrame(data))
    }

    /// 相对强弱因子
    /// 
    /// 计算某列相对于基准列的相对强弱（比值或差值）
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `col` - 目标列名
    /// * `benchmark_col` - 基准列名
    /// * `factor_col` - 输出因子列名（可选）
    /// * `method` - 计算方法: "ratio"(比值), "diff"(差值)
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含相对强弱因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 计算股票相对于市场的相对强弱
    /// df = factor.relative_strength(df, "stock_return", "market_return", "rs", "ratio")
    /// ```
    pub fn relative_strength(
        &self,
        df: PyDataFrame,
        col: &str,
        benchmark_col: &str,
        factor_col: Option<&str>,
        method: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;
        let method = method.unwrap_or("ratio");

        let default_name = format!("{}_{}_rs", col, benchmark_col);
        let col_name = factor_col.unwrap_or(&default_name);

        let series = data.column(col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let benchmark_series = data.column(benchmark_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", benchmark_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let values = series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;
        let benchmark_values = benchmark_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let len = values.len();
        let mut rs_values = vec![f64::NAN; len];

        for i in 0..len {
            if let (Some(v), Some(b)) = (values.get(i), benchmark_values.get(i)) {
                if !v.is_nan() && !b.is_nan() {
                    rs_values[i] = match method {
                        "ratio" => if b != 0.0 { v / b } else { f64::NAN },
                        "diff" => v - b,
                        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("不支持的方法: {}", method)
                        ))
                    };
                }
            }
        }

        let rs_series = Series::new(col_name.into(), rs_values);
        data.with_column(rs_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
        
        Ok(PyDataFrame(data))
    }

    /// IC值计算（信息系数）
    /// 
    /// 计算因子值与未来收益率之间的Pearson相关系数
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `group_col` - 分组列名（可选，如按日期分组）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - IC值结果数据框
    /// 
    /// # Example
    /// ```python
    /// # 计算每日IC值
    /// ic_df = factor.ic(df, "factor_value", "next_return", "date")
    /// ```
    pub fn ic(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        group_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        if let Some(_group) = group_col {
            // 按组计算IC（简化版本，手动实现）
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("分组IC计算暂未实现，请在Python层面实现"));
        } else {
            // 全局IC
            let factor_series = data.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", factor_col, e)))?
                .as_materialized_series()
                .cast(&DataType::Float64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
            
            let return_series = data.column(return_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", return_col, e)))?
                .as_materialized_series()
                .cast(&DataType::Float64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

            // 手动计算Pearson相关系数
            let ic = calculate_pearson_corr(&factor_series, &return_series)?;
            
            let result = df!(
                "ic" => [ic]
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;
            
            Ok(PyDataFrame(result))
        }
    }

    /// IR值计算（信息比率）
    /// 
    /// 计算IC的均值除以IC的标准差
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（通常是IC序列）
    /// * `ic_col` - IC列名
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - IR值结果
    /// 
    /// # Example
    /// ```python
    /// # 先计算IC，再计算IR
    /// ic_df = factor.ic(df, "factor_value", "next_return", "date")
    /// ir_df = factor.ir(ic_df, "ic")
    /// ```
    pub fn ir(
        &self,
        df: PyDataFrame,
        ic_col: &str,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        let ic_series = data.column(ic_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", ic_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let ic_values = ic_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..ic_values.len() {
            if let Some(v) = ic_values.get(i) {
                if !v.is_nan() {
                    sum += v;
                    count += 1;
                }
            }
        }
        let mean = if count > 0 { sum / count as f64 } else { f64::NAN };

        let mut sum_sq = 0.0;
        for i in 0..ic_values.len() {
            if let Some(v) = ic_values.get(i) {
                if !v.is_nan() {
                    sum_sq += (v - mean).powi(2);
                }
            }
        }
        let std = if count > 1 { (sum_sq / (count - 1) as f64).sqrt() } else { f64::NAN };

        let ir = if std > 0.0 { mean / std } else { f64::NAN };

        let result = df!(
            "ic_mean" => [mean],
            "ic_std" => [std],
            "ir" => [ir]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// Rank IC计算（秩相关系数）
    /// 
    /// 计算因子值与未来收益率之间的Spearman秩相关系数
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `group_col` - 分组列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - Rank IC值结果
    /// 
    /// # Example
    /// ```python
    /// # 计算每日Rank IC
    /// rank_ic_df = factor.rank_ic(df, "factor_value", "next_return", "date")
    /// ```
    pub fn rank_ic(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        group_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        if let Some(_group) = group_col {
            // 按组计算Rank IC（简化版本）
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("分组Rank IC计算暂未实现，请在Python层面实现"));
        } else {
            // 全局Rank IC
            let factor_series = data.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", factor_col, e)))?
                .as_materialized_series()
                .cast(&DataType::Float64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
            
            let return_series = data.column(return_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", return_col, e)))?
                .as_materialized_series()
                .cast(&DataType::Float64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

            let rank_ic = calculate_spearman_corr(&factor_series, &return_series)?;
            
            let result = df!(
                "rank_ic" => [rank_ic]
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;
            
            Ok(PyDataFrame(result))
        }
    }

    /// 分层分析
    /// 
    /// 将因子值分成N层，计算各层的平均收益率
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `n_quantiles` - 分层数量（默认5）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 各层统计结果
    /// 
    /// # Example
    /// ```python
    /// # 5分层分析
    /// quantile_df = factor.quantile(df, "factor_value", "next_return", 5)
    /// ```
    pub fn quantile(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        n_quantiles: Option<i32>,
    ) -> PyResult<PyDataFrame> {
        let mut data = df.0;
        let n = n_quantiles.unwrap_or(5);

        // 计算分位数并添加分组列
        let factor_series = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", factor_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let factor_values = factor_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let len = factor_values.len();
        let mut quantile_groups = vec![0i32; len];

        // 排序并分组
        let mut pairs: Vec<(usize, f64)> = Vec::new();
        for i in 0..len {
            if let Some(v) = factor_values.get(i) {
                if !v.is_nan() {
                    pairs.push((i, v));
                }
            }
        }
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let group_size = (pairs.len() as f64 / n as f64).ceil() as usize;
        for (rank, (idx, _)) in pairs.iter().enumerate() {
            let group = (rank / group_size).min((n - 1) as usize) as i32 + 1;
            quantile_groups[*idx] = group;
        }

        let group_series = Series::new("quantile_group".into(), quantile_groups);
        data.with_column(group_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加分组列失败: {}", e)))?;

        // 按分组计算统计量
        let result = data.lazy()
            .group_by([col("quantile_group")])
            .agg([
                col(return_col).mean().alias("mean_return"),
                col(return_col).count().alias("count"),
            ])
            .sort(["quantile_group"], Default::default())
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分层统计失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 因子覆盖度
    /// 
    /// 计算因子的非空值占比
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `factor_col` - 因子列名
    /// * `group_col` - 分组列名（可选）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 覆盖度统计
    /// 
    /// # Example
    /// ```python
    /// # 计算每日因子覆盖度
    /// coverage_df = factor.coverage(df, "factor_value", "date")
    /// ```
    pub fn coverage(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        group_col: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        if let Some(group) = group_col {
            // 按组计算覆盖度
            let result = data.lazy()
                .group_by([col(group)])
                .agg([
                    col(factor_col).count().alias("valid_count"),
                    len().alias("total_count"),
                ])
                .with_column(
                    (col("valid_count").cast(DataType::Float64) / 
                     col("total_count").cast(DataType::Float64)).alias("coverage")
                )
                .collect()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算覆盖度失败: {}", e)))?;
            
            Ok(PyDataFrame(result))
        } else {
            // 全局覆盖度
            let factor_series = data.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", factor_col, e)))?;

            let valid_count = factor_series.len() - factor_series.null_count();
            let total_count = data.height();
            let coverage = valid_count as f64 / total_count as f64;

            let result = df!(
                "valid_count" => [valid_count as i64],
                "total_count" => [total_count as i64],
                "coverage" => [coverage]
            ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

            Ok(PyDataFrame(result))
        }
    }

    /// IC胜率
    /// 
    /// 计算IC > 0的比例
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（IC序列）
    /// * `ic_col` - IC列名
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - IC胜率统计
    /// 
    /// # Example
    /// ```python
    /// ic_df = factor.ic(df, "factor_value", "next_return", "date")
    /// win_rate_df = factor.ic_win_rate(ic_df, "ic")
    /// ```
    pub fn ic_win_rate(
        &self,
        df: PyDataFrame,
        ic_col: &str,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        let ic_series = data.column(ic_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", ic_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let ic_values = ic_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        let mut win_count = 0;
        let mut total_count = 0;
        for i in 0..ic_values.len() {
            if let Some(v) = ic_values.get(i) {
                if !v.is_nan() {
                    total_count += 1;
                    if v > 0.0 {
                        win_count += 1;
                    }
                }
            }
        }

        let win_rate = if total_count > 0 {
            win_count as f64 / total_count as f64
        } else {
            f64::NAN
        };

        let result = df!(
            "win_count" => [win_count as i64],
            "total_count" => [total_count as i64],
            "win_rate" => [win_rate]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 多空收益率
    /// 
    /// 计算多头组合（高因子值）与空头组合（低因子值）的收益差
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `top_pct` - 多头比例（默认0.2，即前20%）
    /// * `bottom_pct` - 空头比例（默认0.2，即后20%）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 多空收益统计
    /// 
    /// # Example
    /// ```python
    /// # 计算多空收益（前20% vs 后20%）
    /// ls_df = factor.long_short(df, "factor_value", "next_return", 0.2, 0.2)
    /// ```
    pub fn long_short(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        top_pct: Option<f64>,
        bottom_pct: Option<f64>,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;
        let top = top_pct.unwrap_or(0.2);
        let bottom = bottom_pct.unwrap_or(0.2);

        let factor_series = data.column(factor_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", factor_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let return_series = data.column(return_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", return_col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

        let factor_values = factor_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;
        let return_values = return_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

        // 收集有效数据对
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for i in 0..factor_values.len() {
            if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                if !f.is_nan() && !r.is_nan() {
                    pairs.push((f, r));
                }
            }
        }

        // 按因子值排序
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n = pairs.len();
        let top_n = (n as f64 * top).ceil() as usize;
        let bottom_n = (n as f64 * bottom).ceil() as usize;

        // 计算多头和空头的平均收益
        let long_return = if top_n > 0 {
            let sum: f64 = pairs.iter().rev().take(top_n).map(|(_, r)| r).sum();
            sum / top_n as f64
        } else {
            f64::NAN
        };

        let short_return = if bottom_n > 0 {
            let sum: f64 = pairs.iter().take(bottom_n).map(|(_, r)| r).sum();
            sum / bottom_n as f64
        } else {
            f64::NAN
        };

        let ls_return = long_return - short_return;

        let result = df!(
            "long_return" => [long_return],
            "short_return" => [short_return],
            "long_short_return" => [ls_return]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 因子换手率
    /// 
    /// 计算相邻两期因子排名的变化程度
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列和标识列）
    /// * `factor_col` - 因子列名
    /// * `time_col` - 时间列名
    /// * `id_col` - 标识列名（如股票代码）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 换手率统计
    /// 
    /// # Example
    /// ```python
    /// # 计算因子换手率
    /// turnover_df = factor.turnover(df, "factor_value", "date", "stock_code")
    /// ```
    pub fn turnover(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        time_col: &str,
        id_col: &str,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        // 按时间和因子值排序，计算排名
        let ranked = data.lazy()
            .with_column(
                col(factor_col).rank(
                    RankOptions {
                        method: RankMethod::Average,
                        descending: false,
                    },
                    None
                ).over([col(time_col)]).alias("factor_rank")
            )
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("排名失败: {}", e)))?;

        // 计算相邻期的排名变化
        let with_lag = ranked.lazy()
            .sort([time_col], Default::default())
            .with_column(
                col("factor_rank").shift(lit(1)).over([col(id_col)]).alias("prev_rank")
            )
            .with_column(
                (col("factor_rank") - col("prev_rank")).abs().alias("rank_change")
            )
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算排名变化失败: {}", e)))?;

        // 按时间汇总换手率
        let result = with_lag.lazy()
            .group_by([col(time_col)])
            .agg([
                col("rank_change").mean().alias("avg_rank_change"),
                col("rank_change").count().alias("count"),
            ])
            .sort([time_col], Default::default())
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("汇总换手率失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 因子清洗与正交化
    /// 
    /// 对多个因子进行清洗、中性化和正交化处理
    /// 
    /// # Arguments
    /// * `df` - 输入数据框
    /// * `factor_cols` - 要处理的因子列名列表
    /// * `winsorize` - 是否进行缩尾处理，默认false
    /// * `winsorize_n` - 缩尾的标准差倍数，默认3.0
    /// * `neutralize` - 是否进行中性化，默认false
    /// * `industry_col` - 行业列名，默认"industry"
    /// * `cap_col` - 市值列名，默认"market_cap"
    /// * `standardize` - 是否标准化，默认false
    /// * `orthogonalize` - 是否正交化，默认false
    /// * `suffix` - 输出列名后缀，默认"_clean"
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 包含清洗后因子的数据框
    /// 
    /// # Example
    /// ```python
    /// # 对多个因子进行清洗和正交化
    /// df_clean = factor.clean(
    ///     df,
    ///     ["factor1", "factor2", "factor3"],
    ///     winsorize=True,
    ///     neutralize=True,
    ///     orthogonalize=True
    /// )
    /// ```
    #[pyo3(signature = (df, factor_cols, winsorize=false, winsorize_n=3.0, neutralize=false, industry_col="industry", cap_col="market_cap", standardize=false, orthogonalize=false, suffix="_clean"))]
    pub fn clean(
        &self,
        df: PyDataFrame,
        factor_cols: Vec<String>,
        winsorize: bool,
        winsorize_n: f64,
        neutralize: bool,
        industry_col: &str,
        cap_col: &str,
        standardize: bool,
        orthogonalize: bool,
        suffix: &str,
    ) -> PyResult<PyDataFrame> {
        use crate::data::clean as data_clean;
        
        if factor_cols.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("因子列表不能为空"));
        }

        let mut data = df.0;

        // 第一步：对每个因子使用data::clean进行基础清洗
        for factor_col in &factor_cols {
            let winsorize_method = if winsorize { "std" } else { "none" };
            let neutralize_method = if neutralize { "both" } else { "none" };
            let standardize_method = if standardize { "zscore" } else { "none" };

            let cleaned = data_clean(
                PyDataFrame(data.clone()),
                factor_col,
                None,
                winsorize_method,
                Some(winsorize_n),
                standardize_method,
                neutralize_method,
                Some(industry_col),
                Some(cap_col),
            )?;

            data = cleaned.0;
        }

        // 第二步：如果需要正交化且有多个因子
        if orthogonalize && factor_cols.len() > 1 {
            let cleaned_cols: Vec<String> = factor_cols.iter()
                .map(|col| format!("{}_clean", col))
                .collect();
            
            data = orthogonalize_factors(data, cleaned_cols, "orth_")?;

            // 重命名正交化后的列
            for factor_col in &factor_cols {
                let orth_col_name = format!("orth_{}_clean", factor_col);
                let final_col_name = format!("{}{}", factor_col, suffix);

                if let Ok(col) = data.column(&orth_col_name) {
                    let renamed = col.clone().with_name(PlSmallStr::from(final_col_name.as_str()));
                    data = data.drop(&orth_col_name)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除列失败: {}", e)))?;
                    data = data.with_column(renamed)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
                        .clone();
                }
                
                // 删除中间的_clean列
                let clean_col_name = format!("{}_clean", factor_col);
                if data.column(&clean_col_name).is_ok() {
                    data = data.drop(&clean_col_name)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除列失败: {}", e)))?;
                }
            }
        } else {
            // 不正交化，直接重命名
            for factor_col in &factor_cols {
                let clean_col_name = format!("{}_clean", factor_col);
                let final_col_name = format!("{}{}", factor_col, suffix);

                if let Ok(col) = data.column(&clean_col_name) {
                    let renamed = col.clone().with_name(PlSmallStr::from(final_col_name.as_str()));
                    data = data.drop(&clean_col_name)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除列失败: {}", e)))?;
                    data = data.with_column(renamed)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
                        .clone();
                }
            }
        }

        Ok(PyDataFrame(data))
    }

    // ====================================================================
    // 单因子检验 (Single Factor Tests)
    // ====================================================================

    /// IC检验
    /// 
    /// 计算因子与收益率的信息系数，返回IC均值、标准差、t统计量和p值
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `method` - 相关系数方法："pearson"或"spearman"，默认"pearson"
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - IC统计结果（ic_mean, ic_std, t_stat, p_value）
    #[pyo3(signature = (df, factor_col, return_col, time_col, method="pearson"))]
    pub fn test(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        method: &str,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        // 按时间分组计算IC
        let time_groups = data.partition_by([time_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let mut ic_values = Vec::new();

        for group_df in time_groups {
            let factor_series = group_df.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
                .as_materialized_series()
                .clone();
            let return_series = group_df.column(return_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益列失败: {}", e)))?
                .as_materialized_series()
                .clone();

            let ic = match method {
                "pearson" => calculate_pearson_corr(&factor_series, &return_series)?,
                "spearman" => calculate_spearman_corr(&factor_series, &return_series)?,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("不支持的方法: {}", method)
                )),
            };

            if !ic.is_nan() {
                ic_values.push(ic);
            }
        }

        if ic_values.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("没有有效的IC值"));
        }

        // 计算IC统计量
        let n = ic_values.len() as f64;
        let ic_mean = ic_values.iter().sum::<f64>() / n;
        let ic_var = ic_values.iter().map(|x| (x - ic_mean).powi(2)).sum::<f64>() / (n - 1.0);
        let ic_std = ic_var.sqrt();
        let t_stat = ic_mean / (ic_std / n.sqrt());
        
        // 简化的p值计算（双尾检验）
        let p_value = if t_stat.abs() > 2.576 {
            0.01
        } else if t_stat.abs() > 1.96 {
            0.05
        } else if t_stat.abs() > 1.645 {
            0.10
        } else {
            1.0 - (t_stat.abs() / 3.0).min(0.999)
        };

        let result = df!(
            "ic_mean" => [ic_mean],
            "ic_std" => [ic_std],
            "t_stat" => [t_stat],
            "p_value" => [p_value],
            "n_periods" => [n]
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 分组回测
    /// 
    /// 将因子分N组，计算各组收益率，并进行单调性检验
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `n_groups` - 分组数量，默认5
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 各组平均收益率
    #[pyo3(signature = (df, factor_col, return_col, time_col, n_groups=5))]
    pub fn sorts(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        n_groups: usize,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        if n_groups < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("分组数量必须>=2"));
        }

        // 按时间和因子值分组 - 使用cut进行等频分组
        let with_rank = data.lazy()
            .with_column(
                col(factor_col)
                    .rank(
                        RankOptions {
                            method: RankMethod::Average,
                            descending: false,
                        },
                        None
                    )
                    .over([col(time_col)])
                    .alias("_rank")
            )
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("排名失败: {}", e)))?;

        // 将排名转换为分组
        let with_group = with_rank.lazy()
            .with_column(
                ((col("_rank") - lit(1.0)) * lit(n_groups as f64) / col("_rank").max().over([col(time_col)]))
                    .floor()
                    .cast(DataType::Int32)
                    .alias("factor_group")
            )
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        // 删除临时_rank列
        let with_group = with_group.drop("_rank")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除临时列失败: {}", e)))?;

        // 计算各组平均收益
        let result = with_group.lazy()
            .group_by([col("factor_group")])
            .agg([
                col(return_col).mean().alias("mean_return"),
                col(return_col).std(1).alias("std_return"),
                col(return_col).count().alias("count"),
            ])
            .sort(["factor_group"], Default::default())
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("计算失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 因子收益率
    /// 
    /// 计算因子的时间序列收益率（用于因子择时）
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 因子收益率时间序列
    #[pyo3(signature = (df, factor_col, return_col, time_col))]
    pub fn mimick(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
    ) -> PyResult<PyDataFrame> {
        use crate::data::linear;
        
        let data = df.0;

        // 按时间分组，进行横截面回归
        let time_groups = data.partition_by([time_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let mut times = Vec::new();
        let mut factor_returns = Vec::new();

        for group_df in time_groups {
            let time_value = group_df.column(time_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取时间失败: {}", e)))?
                .get(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取时间值失败: {}", e)))?
                .to_string();

            // 横截面回归
            let (_, stats) = linear(
                PyDataFrame(group_df.clone()),
                vec![factor_col.to_string()],
                return_col,
                None,
                None,
                true,
            )?;

            if let Some((coeffs, _)) = stats {
                if !coeffs.is_empty() {
                    times.push(time_value);
                    factor_returns.push(coeffs[0]);
                }
            }
        }

        let result = df!(
            time_col => times,
            "factor_return" => factor_returns
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// IC衰减分析
    /// 
    /// 计算因子对未来N期收益的预测能力衰减
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列和标识列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `id_col` - 标识列名
    /// * `max_periods` - 最大预测期数，默认10
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - IC衰减序列
    #[pyo3(signature = (df, factor_col, return_col, time_col, id_col, max_periods=10))]
    pub fn decay(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        id_col: &str,
        max_periods: i64,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        let mut periods = Vec::new();
        let mut ic_values = Vec::new();

        for lag in 1..=max_periods {
            // 创建lag期的收益率
            let with_lag = data.clone().lazy()
                .sort([time_col], Default::default())
                .with_column(
                    col(return_col).shift(lit(-lag)).over([col(id_col)]).alias("future_return")
                )
                .collect()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建lag失败: {}", e)))?;

            // 计算IC
            let factor_series = with_lag.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子列失败: {}", e)))?
                .as_materialized_series()
                .clone();
            let return_series = with_lag.column("future_return")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益列失败: {}", e)))?
                .as_materialized_series()
                .clone();

            let ic = calculate_pearson_corr(&factor_series, &return_series)?;

            if !ic.is_nan() {
                periods.push(lag);
                ic_values.push(ic);
            }
        }

        let result = df!(
            "period" => periods,
            "ic" => ic_values
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    // ====================================================================
    // 多因子回归分析 (Multi-Factor Regression)
    // ====================================================================

    /// Fama-MacBeth回归
    /// 
    /// 横截面回归+时间序列统计，估计因子风险溢价
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_cols` - 因子列名列表
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 因子风险溢价及t统计量
    #[pyo3(signature = (df, factor_cols, return_col, time_col))]
    pub fn fama(
        &self,
        df: PyDataFrame,
        factor_cols: Vec<String>,
        return_col: &str,
        time_col: &str,
    ) -> PyResult<PyDataFrame> {
        use crate::data::linear;
        
        let data = df.0;

        // 按时间分组进行横截面回归
        let time_groups = data.partition_by([time_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let n_factors = factor_cols.len();
        let mut all_coeffs = vec![Vec::new(); n_factors];

        for group_df in time_groups {
            let (_, stats) = linear(
                PyDataFrame(group_df),
                factor_cols.clone(),
                return_col,
                None,
                None,
                true,
            )?;

            if let Some((coeffs, _)) = stats {
                for (i, coeff) in coeffs.iter().enumerate() {
                    if i < n_factors {
                        all_coeffs[i].push(*coeff);
                    }
                }
            }
        }

        // 计算时间序列统计量
        let mut factor_names = Vec::new();
        let mut mean_coeffs = Vec::new();
        let mut t_stats = Vec::new();

        for (i, factor_name) in factor_cols.iter().enumerate() {
            let coeffs = &all_coeffs[i];
            if !coeffs.is_empty() {
                let n = coeffs.len() as f64;
                let mean = coeffs.iter().sum::<f64>() / n;
                let var = coeffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                let std = var.sqrt();
                let t_stat = mean / (std / n.sqrt());

                factor_names.push(factor_name.clone());
                mean_coeffs.push(mean);
                t_stats.push(t_stat);
            }
        }

        let result = df!(
            "factor" => factor_names,
            "risk_premium" => mean_coeffs,
            "t_stat" => t_stats
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 时间序列回归
    /// 
    /// 分析个股收益对因子的暴露度（factor loading）
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列和标识列）
    /// * `factor_cols` - 因子列名列表
    /// * `return_col` - 收益率列名
    /// * `id_col` - 标识列名
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 各股票的因子暴露度
    #[pyo3(signature = (df, factor_cols, return_col, id_col))]
    pub fn regress(
        &self,
        df: PyDataFrame,
        factor_cols: Vec<String>,
        return_col: &str,
        id_col: &str,
    ) -> PyResult<PyDataFrame> {
        use crate::data::linear;
        
        let data = df.0;

        // 按标识分组进行时间序列回归
        let id_groups = data.partition_by([id_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let mut ids = Vec::new();
        let mut all_loadings = vec![Vec::new(); factor_cols.len()];
        let mut r_squareds = Vec::new();

        for group_df in id_groups {
            let id_value = group_df.column(id_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取ID失败: {}", e)))?
                .get(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取ID值失败: {}", e)))?
                .to_string();

            let (_, stats) = linear(
                PyDataFrame(group_df.clone()),
                factor_cols.clone(),
                return_col,
                None,
                None,
                true,
            )?;

            if let Some((loadings, r_sq)) = stats {
                ids.push(id_value);
                r_squareds.push(r_sq);
                
                for (i, loading) in loadings.iter().enumerate() {
                    if i < factor_cols.len() {
                        all_loadings[i].push(*loading);
                    }
                }
            }
        }

        // 构建结果DataFrame
        let mut result = df!(
            id_col => ids.clone(),
            "r_squared" => r_squareds
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        for (i, factor_name) in factor_cols.iter().enumerate() {
            let loading_col = Series::new(PlSmallStr::from(factor_name.as_str()), all_loadings[i].clone());
            result = result.with_column(loading_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?
                .clone();
        }

        Ok(PyDataFrame(result))
    }

    /// 因子模拟组合
    /// 
    /// 构建因子模拟组合（Factor Mimicking Portfolio）
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `long_pct` - 多头比例，默认0.3
    /// * `short_pct` - 空头比例，默认0.3
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 因子模拟组合收益序列
    #[pyo3(signature = (df, factor_col, return_col, time_col, long_pct=0.3, short_pct=0.3))]
    pub fn portfolio(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        long_pct: f64,
        short_pct: f64,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        // 按时间分组
        let time_groups = data.partition_by([time_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let mut times = Vec::new();
        let mut portfolio_returns = Vec::new();

        for group_df in time_groups {
            let time_col_data = group_df.column(time_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取时间失败: {}", e)))?;
            let time_value = time_col_data
                .get(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取时间值失败: {}", e)))?
                .to_string();

            let factor_series = group_df.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子失败: {}", e)))?
                .cast(&DataType::Float64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

            let return_series = group_df.column(return_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益失败: {}", e)))?
                .cast(&DataType::Float64)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;

            let factor_values = factor_series.f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;
            let return_values = return_series.f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;

            // 收集有效数据对
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for i in 0..factor_values.len() {
                if let (Some(f), Some(r)) = (factor_values.get(i), return_values.get(i)) {
                    if !f.is_nan() && !r.is_nan() {
                        pairs.push((f, r));
                    }
                }
            }

            if pairs.is_empty() {
                continue;
            }

            // 按因子值排序
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let n = pairs.len();
            let long_n = (n as f64 * long_pct).ceil() as usize;
            let short_n = (n as f64 * short_pct).ceil() as usize;

            // 计算多空组合收益
            let long_return = if long_n > 0 {
                let sum: f64 = pairs.iter().rev().take(long_n).map(|(_, r)| r).sum();
                sum / long_n as f64
            } else {
                0.0
            };

            let short_return = if short_n > 0 {
                let sum: f64 = pairs.iter().take(short_n).map(|(_, r)| r).sum();
                sum / short_n as f64
            } else {
                0.0
            };

            times.push(time_value);
            portfolio_returns.push(long_return - short_return);
        }

        let result = df!(
            time_col => times,
            "portfolio_return" => portfolio_returns
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    // ====================================================================
    // 稳健性检验 (Robustness Tests)
    // ====================================================================

    /// 子期检验
    /// 
    /// 将样本分为多个子期，检验因子在各子期的稳定性
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `n_subsamples` - 子期数量，默认3
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 各子期的IC统计
    #[pyo3(signature = (df, factor_col, return_col, time_col, n_subsamples=3))]
    pub fn subsample(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        n_subsamples: usize,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        if n_subsamples < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("子期数量必须>=2"));
        }

        let n_total = data.height();
        let subsample_size = n_total / n_subsamples;

        let mut subsample_ids: Vec<i32> = Vec::new();
        let mut ic_means = Vec::new();
        let mut ic_stds = Vec::new();

        for i in 0..n_subsamples {
            let start_idx = i * subsample_size;
            let end_idx = if i == n_subsamples - 1 {
                n_total
            } else {
                (i + 1) * subsample_size
            };

            let subsample = data.slice(start_idx as i64, end_idx - start_idx);

            // 对子样本调用test
            let ic_result = self.test(
                PyDataFrame(subsample),
                factor_col,
                return_col,
                time_col,
                "pearson",
            )?;

            let ic_mean = ic_result.0.column("ic_mean")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取IC均值失败: {}", e)))?
                .f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?
                .get(0)
                .unwrap_or(f64::NAN);

            let ic_std = ic_result.0.column("ic_std")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取IC标准差失败: {}", e)))?
                .f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?
                .get(0)
                .unwrap_or(f64::NAN);

            subsample_ids.push((i + 1) as i32);
            ic_means.push(ic_mean);
            ic_stds.push(ic_std);
        }

        let result = df!(
            "subsample" => subsample_ids,
            "ic_mean" => ic_means,
            "ic_std" => ic_stds
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 分组检验
    /// 
    /// 按行业或市值分组，检验因子在各组的表现
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列和分组列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `group_col` - 分组列名（如行业、市值分组）
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 各组的IC统计
    #[pyo3(signature = (df, factor_col, return_col, time_col, group_col))]
    pub fn subgroup(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        group_col: &str,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        // 按分组列分组
        let groups = data.partition_by([group_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let mut group_names = Vec::new();
        let mut ic_means = Vec::new();
        let mut ic_stds = Vec::new();

        for group_df in groups {
            let group_value = group_df.column(group_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取分组值失败: {}", e)))?
                .get(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取分组名失败: {}", e)))?
                .to_string();

            // 对每组调用test
            let ic_result = self.test(
                PyDataFrame(group_df.clone()),
                factor_col,
                return_col,
                time_col,
                "pearson",
            )?;

            let ic_mean = ic_result.0.column("ic_mean")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取IC均值失败: {}", e)))?
                .f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?
                .get(0)
                .unwrap_or(f64::NAN);

            let ic_std = ic_result.0.column("ic_std")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取IC标准差失败: {}", e)))?
                .f64()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?
                .get(0)
                .unwrap_or(f64::NAN);

            group_names.push(group_value);
            ic_means.push(ic_mean);
            ic_stds.push(ic_std);
        }

        let result = df!(
            group_col => group_names,
            "ic_mean" => ic_means,
            "ic_std" => ic_stds
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }

    /// 滚动IC分析
    /// 
    /// 使用滚动窗口计算IC，观察因子效果的时间稳定性
    /// 
    /// # Arguments
    /// * `df` - 输入数据框（需包含时间列）
    /// * `factor_col` - 因子列名
    /// * `return_col` - 收益率列名
    /// * `time_col` - 时间列名
    /// * `window` - 滚动窗口大小（期数），默认20
    /// 
    /// # Returns
    /// * `PyResult<PyDataFrame>` - 滚动IC序列
    #[pyo3(signature = (df, factor_col, return_col, time_col, window=20))]
    pub fn rolling(
        &self,
        df: PyDataFrame,
        factor_col: &str,
        return_col: &str,
        time_col: &str,
        window: usize,
    ) -> PyResult<PyDataFrame> {
        let data = df.0;

        if window < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("窗口大小必须>=2"));
        }

        // 按时间排序
        let sorted = data.lazy()
            .sort([time_col], Default::default())
            .collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("排序失败: {}", e)))?;

        // 按时间分组
        let time_groups = sorted.partition_by([time_col], true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("分组失败: {}", e)))?;

        let n_periods = time_groups.len();
        
        if n_periods < window {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("期数({})小于窗口大小({})", n_periods, window)
            ));
        }

        let mut times = Vec::new();
        let mut rolling_ics = Vec::new();

        // 对每个时间窗口计算IC
        for i in (window - 1)..n_periods {
            // 合并窗口内的数据
            let mut window_data = time_groups[i - window + 1].clone();
            for j in (i - window + 2)..=i {
                window_data = window_data.vstack(&time_groups[j])
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("合并失败: {}", e)))?;
            }

            // 计算窗口IC
            let factor_series = window_data.column(factor_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取因子失败: {}", e)))?
                .as_materialized_series()
                .clone();
            let return_series = window_data.column(return_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取收益失败: {}", e)))?
                .as_materialized_series()
                .clone();

            let ic = calculate_pearson_corr(&factor_series, &return_series)?;

            // 获取窗口结束时间
            let end_time = time_groups[i].column(time_col)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取时间失败: {}", e)))?
                .get(0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取时间值失败: {}", e)))?
                .to_string();

            times.push(end_time);
            rolling_ics.push(ic);
        }

        let result = df!(
            time_col => times,
            "rolling_ic" => rolling_ics
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("创建结果失败: {}", e)))?;

        Ok(PyDataFrame(result))
    }
}
