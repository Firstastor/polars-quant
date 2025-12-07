use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3::prelude::*;
use std::path::Path;

/// 计算收益率
/// 
/// # 参数
/// - df: 输入DataFrame
/// - price_col: 价格列名，默认 "close"
/// - period: 计算周期，默认 1
/// - method: 计算方法
///   - "simple": 简单收益率 (price[t] - price[t-period]) / price[t-period]
///   - "log": 对数收益率 ln(price[t] / price[t-period])
/// - return_col: 返回列名，默认 "return"
#[pyfunction]
#[pyo3(signature = (df, price_col="close", period=1, method="simple", return_col="return"))]
pub fn returns(
    df: PyDataFrame,
    price_col: &str,
    period: i64,
    method: &str,
    return_col: &str,
) -> PyResult<PyDataFrame> {
    let mut df: DataFrame = df.into();
    
    // 获取价格列
    let price_col_data = df.column(price_col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("列 '{}' 不存在: {}", price_col, e)
        ))?;
    
    // 提取浮点数数据
    let price_values: Vec<f64> = price_col_data.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("列 '{}' 不是数值类型: {}", price_col, e)
        ))?
        .into_iter()
        .map(|x| x.unwrap_or(f64::NAN))
        .collect();
    
    let len = price_values.len();
    let mut returns_values = vec![f64::NAN; len];
    
    // 计算收益率
    match method {
        "simple" => {
            // 简单收益率: (price[t] - price[t-period]) / price[t-period]
            for i in (period as usize)..len {
                let current = price_values[i];
                let past = price_values[i - period as usize];
                if past != 0.0 && !past.is_nan() && !current.is_nan() {
                    returns_values[i] = (current - past) / past;
                }
            }
        },
        "log" => {
            // 对数收益率: ln(price[t] / price[t-period])
            for i in (period as usize)..len {
                let current = price_values[i];
                let past = price_values[i - period as usize];
                if past > 0.0 && current > 0.0 {
                    returns_values[i] = (current / past).ln();
                }
            }
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("不支持的方法: {}，支持的方法: 'simple', 'log'", method)
            ));
        }
    };
    
    // 创建收益率列
    let return_series = Series::new(PlSmallStr::from(return_col), returns_values);
    
    df.with_column(return_series)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("添加收益率列失败: {}", e)
        ))?;
    
    Ok(PyDataFrame(df))
}


/// 从文件夹批量加载股票数据
/// 
/// # 参数
/// - folder: 数据文件夹路径
/// - file_type: 文件类型，支持 "parquet", "csv", "xlsx", "xls", "json", "feather", "ipc"
///              可以是单个字符串或字符串列表，None 表示支持所有格式
/// - prefix: 文件名前缀过滤（可选）
/// - suffix: 文件名后缀过滤（可选）
/// - has_header: CSV/Excel 文件是否包含表头，默认 true
/// 
/// # 返回
/// 合并后的 DataFrame，包含所有股票数据
/// 列格式: date, {symbol}_open, {symbol}_high, {symbol}_low, {symbol}_close, {symbol}_volume
#[pyfunction]
#[pyo3(signature = (folder, file_type=None, prefix=None, suffix=None, has_header=true))]
pub fn load(
    folder: &str,
    file_type: Option<Vec<String>>,
    prefix: Option<&str>,
    suffix: Option<&str>,
    has_header: bool,
) -> PyResult<PyDataFrame> {
    let folder_path = Path::new(folder);
    
    if !folder_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("文件夹不存在: {}", folder)
        ));
    }
    
    // 支持的文件类型
    let supported_types = if let Some(types) = file_type {
        types
    } else {
        vec![
            "parquet".to_string(),
            "csv".to_string(),
            "xlsx".to_string(),
            "xls".to_string(),
            "json".to_string(),
            "feather".to_string(),
            "ipc".to_string(),
        ]
    };
    
    // 读取目录中的所有文件
    let entries = std::fs::read_dir(folder_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("读取文件夹失败: {}", e)
        ))?;
    
    let mut dataframes = Vec::new();
    
    for entry in entries {
        let entry = entry.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("读取文件条目失败: {}", e)
        ))?;
        
        let path = entry.path();
        
        // 跳过目录
        if path.is_dir() {
            continue;
        }
        
        let file_name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        let file_stem = path.file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        
        // 检查前缀
        if let Some(pre) = prefix {
            if !file_stem.starts_with(pre) {
                continue;
            }
        }
        
        // 检查后缀
        if let Some(suf) = suffix {
            if !file_stem.ends_with(suf) {
                continue;
            }
        }
        
        // 检查文件类型
        if !supported_types.iter().any(|t| t == extension) {
            continue;
        }
        
        // 读取文件
        let mut df = match extension {
            "parquet" => {
                use std::fs::File;
                ParquetReader::new(File::open(&path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("打开 parquet 文件 {} 失败: {}", file_name, e)
                    )
                })?)
                .finish()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("读取 parquet 文件 {} 失败: {}", file_name, e)
                ))?
            },
            "csv" => {
                CsvReadOptions::default()
                    .with_has_header(has_header)
                    .try_into_reader_with_file_path(Some(path.clone()))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("读取 CSV 文件 {} 失败: {}", file_name, e)
                    ))?
                    .finish()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("处理 CSV 文件 {} 失败: {}", file_name, e)
                    ))?
            },
            "json" => {
                JsonReader::new(std::fs::File::open(&path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("打开 JSON 文件 {} 失败: {}", file_name, e)
                    )
                })?)
                .finish()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("读取 JSON 文件 {} 失败: {}", file_name, e)
                ))?
            },
            "feather" | "ipc" => {
                IpcReader::new(std::fs::File::open(&path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("打开 IPC 文件 {} 失败: {}", file_name, e)
                    )
                })?)
                .finish()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    format!("读取 IPC 文件 {} 失败: {}", file_name, e)
                ))?
            },
            "xlsx" | "xls" => {
                use calamine::{Reader, open_workbook_auto};
                
                let mut workbook = open_workbook_auto(&path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(
                        format!("打开 Excel 文件失败 {}: {}", file_name, e)
                    )
                })?;
                
                let sheet_names = workbook.sheet_names().to_owned();
                if sheet_names.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Excel 文件 {} 没有工作表", file_name)
                    ));
                }
                
                let range = workbook.worksheet_range(&sheet_names[0]).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("读取 Excel 工作表失败 {}: {:?}", file_name, e)
                    )
                })?;
                
                let mut columns: Vec<Column> = Vec::new();
                let (nrows, ncols) = range.get_size();
                
                if nrows == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Excel 文件 {} 为空", file_name)
                    ));
                }
                
                let start_row = if has_header { 1 } else { 0 };
                
                for col_idx in 0..ncols {
                    let col_name = if has_header {
                        range.get_value((0, col_idx as u32))
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| format!("column_{}", col_idx))
                    } else {
                        format!("column_{}", col_idx)
                    };
                    
                    let mut values: Vec<Option<f64>> = Vec::new();
                    for row_idx in start_row..nrows {
                        let value = range.get_value((row_idx as u32, col_idx as u32))
                            .and_then(|v| {
                                use calamine::DataType;
                                v.as_f64()
                            });
                        values.push(value);
                    }
                    
                    let series = Series::new(PlSmallStr::from(col_name.as_str()), values);
                    columns.push(Column::new(series.name().clone(), series));
                }
                
                DataFrame::new(columns).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("创建 DataFrame 失败 {}: {}", file_name, e)
                    )
                })?
            },
            _ => continue,
        };
        
        // 使用文件名（去除扩展名）作为股票代码
        let symbol = file_stem;
        
        // 重命名列，除了日期列
        let date_col = df.get_column_names()[0]; // 假设第一列是日期
        
        let mut new_columns = Vec::new();
        for col_name in df.get_column_names() {
            let new_name = if col_name == date_col {
                "date".to_string()
            } else {
                format!("{}_{}", symbol, col_name)
            };
            new_columns.push(new_name);
        }
        
        df.set_column_names(&new_columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("重命名列失败: {}", e)
            ))?;
        
        dataframes.push(df);
    }
    
    if dataframes.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "未找到符合条件的文件"
        ));
    }
    
    // 合并所有 DataFrame
    let mut result = dataframes[0].clone();
    
    for df in dataframes.iter().skip(1) {
        result = result.join(
            df,
            ["date"],
            ["date"],
            JoinType::Full.into(),
            None,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("合并 DataFrame 失败: {}", e)
        ))?;
    }
    
    // 按日期排序
    result = result.sort(["date"], SortMultipleOptions::default())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("排序失败: {}", e)
        ))?;
    
    Ok(PyDataFrame(result))
}

// ============================================================================
// 多元线性回归
// ============================================================================

/// 多元线性回归辅助函数
/// 
/// 计算多元线性回归: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
/// 
/// # 参数
/// - X: m×n 矩阵，m个样本，n个自变量
/// - y: m维向量，因变量
/// 
/// # 返回
/// - coefficients: (n+1)维向量，[b0, b1, b2, ..., bn]（包含截距）
/// - r_squared: R²拟合优度
#[allow(non_snake_case)]
pub(crate) fn calculate_multiple_linear_regression(X: &[Vec<f64>], y: &[f64]) -> PyResult<(Vec<f64>, f64)> {
    let m = y.len(); // 样本数
    if m == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("没有有效数据"));
    }
    
    let n = X[0].len(); // 自变量个数
    if m < n + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("样本数({})必须大于自变量数({})", m, n)
        ));
    }
    
    // 检查所有样本的自变量个数是否一致
    for row in X.iter() {
        if row.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("自变量维度不一致"));
        }
    }
    
    // 构建设计矩阵 X_mat (m × (n+1))，第一列全为1（截距项）
    let mut X_mat = vec![vec![0.0; n + 1]; m];
    for i in 0..m {
        X_mat[i][0] = 1.0; // 截距项
        for j in 0..n {
            X_mat[i][j + 1] = X[i][j];
        }
    }
    
    // 计算 X^T * X (使用正规方程)
    let mut XtX = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..(n + 1) {
        for j in 0..(n + 1) {
            let mut sum = 0.0;
            for k in 0..m {
                sum += X_mat[k][i] * X_mat[k][j];
            }
            XtX[i][j] = sum;
        }
    }
    
    // 计算 X^T * y
    let mut Xty = vec![0.0; n + 1];
    for i in 0..(n + 1) {
        let mut sum = 0.0;
        for k in 0..m {
            sum += X_mat[k][i] * y[k];
        }
        Xty[i] = sum;
    }
    
    // 高斯消元法求解 (X^T * X) * β = X^T * y
    let mut A = XtX.clone();
    let mut b = Xty.clone();
    
    for i in 0..(n + 1) {
        // 找到主元
        let mut max_row = i;
        for k in (i + 1)..(n + 1) {
            if A[k][i].abs() > A[max_row][i].abs() {
                max_row = k;
            }
        }
        
        // 交换行
        A.swap(i, max_row);
        b.swap(i, max_row);
        
        // 检查主元是否为0
        if A[i][i].abs() < 1e-10 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("矩阵奇异，无法求解"));
        }
        
        // 消元
        for k in (i + 1)..(n + 1) {
            let factor = A[k][i] / A[i][i];
            for j in i..(n + 1) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // 回代求解
    let mut coefficients = vec![0.0; n + 1];
    for i in (0..(n + 1)).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..(n + 1) {
            sum += A[i][j] * coefficients[j];
        }
        coefficients[i] = (b[i] - sum) / A[i][i];
    }
    
    // 计算R²
    let mean_y: f64 = y.iter().sum::<f64>() / m as f64;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    
    for i in 0..m {
        // 预测值
        let mut y_pred = coefficients[0];
        for j in 0..n {
            y_pred += coefficients[j + 1] * X[i][j];
        }
        
        ss_tot += (y[i] - mean_y).powi(2);
        ss_res += (y[i] - y_pred).powi(2);
    }
    
    let r_squared = if ss_tot.abs() < 1e-10 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    };
    
    Ok((coefficients, r_squared))
}

/// 多元线性回归
/// 
/// 对多个自变量和一个因变量进行线性回归分析
/// y = b0 + b1*x1 + b2*x2 + ... + bn*xn
/// 
/// # 参数
/// - df: 输入DataFrame
/// - x_cols: 自变量列名列表
/// - y_col: 因变量列名
/// - pred_col: 预测值列名（可选）
/// - resid_col: 残差列名（默认为 "{y_col}_resid"）
/// - return_stats: 是否返回统计信息
/// 
/// # 返回
/// (DataFrame, Option<(Vec<f64>, f64)>)
/// - DataFrame: 添加了预测值和/或残差列的数据框
/// - 统计信息: (系数[b0, b1, ..., bn], R²)
#[pyfunction]
#[pyo3(signature = (df, x_cols, y_col, pred_col=None, resid_col=None, return_stats=false))]
pub fn linear(
    df: PyDataFrame,
    x_cols: Vec<String>,
    y_col: &str,
    pred_col: Option<&str>,
    resid_col: Option<&str>,
    return_stats: bool,
) -> PyResult<(PyDataFrame, Option<(Vec<f64>, f64)>)> {
    let mut data: DataFrame = df.into();
    
    if x_cols.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("至少需要一个自变量"));
    }
    
    let len = data.height();
    
    // 获取因变量y
    let y_series = data.column(y_col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", y_col, e)))?
        .cast(&DataType::Float64)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
    
    let y_values = y_series.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取y数值失败: {}", e)))?;
    
    // 获取自变量X
    let mut x_series_list = Vec::new();
    for col in x_cols.iter() {
        let series = data.column(col.as_str())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
        
        x_series_list.push(series);
    }
    
    // 从Series中提取数值
    let mut x_values_list = Vec::new();
    for series in x_series_list.iter() {
        let values = series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取x数值失败: {}", e)))?;
        x_values_list.push(values);
    }
    
    // 收集有效数据（无NaN的行）
    let mut x_valid: Vec<Vec<f64>> = Vec::new();
    let mut y_valid: Vec<f64> = Vec::new();
    let mut valid_indices: Vec<usize> = Vec::new();
    
    for i in 0..len {
        let mut is_valid = true;
        
        // 检查y是否有效
        if let Some(y_val) = y_values.get(i) {
            if y_val.is_nan() {
                is_valid = false;
            }
        } else {
            is_valid = false;
        }
        
        // 检查所有x是否有效
        let mut x_row = Vec::new();
        if is_valid {
            for x_vals in x_values_list.iter() {
                if let Some(x_val) = x_vals.get(i) {
                    if x_val.is_nan() {
                        is_valid = false;
                        break;
                    }
                    x_row.push(x_val);
                } else {
                    is_valid = false;
                    break;
                }
            }
        }
        
        if is_valid {
            x_valid.push(x_row);
            y_valid.push(y_values.get(i).unwrap());
            valid_indices.push(i);
        }
    }
    
    // 调用多元线性回归
    let (coefficients, r_squared) = calculate_multiple_linear_regression(&x_valid, &y_valid)?;
    
    // 生成预测值和残差
    let mut pred_values = vec![f64::NAN; len];
    let mut resid_values = vec![f64::NAN; len];
    
    for (idx, &i) in valid_indices.iter().enumerate() {
        let mut y_pred = coefficients[0]; // 截距
        for (j, x_vals) in x_values_list.iter().enumerate() {
            y_pred += coefficients[j + 1] * x_vals.get(i).unwrap();
        }
        pred_values[i] = y_pred;
        resid_values[i] = y_valid[idx] - y_pred;
    }
    
    // 添加预测值列（如果指定）
    if let Some(pred_name) = pred_col {
        let pred_series = Series::new(PlSmallStr::from(pred_name), pred_values);
        data.with_column(pred_series)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加预测值列失败: {}", e)))?;
    }
    
    // 添加残差列
    let default_resid_name = format!("{}_resid", y_col);
    let resid_name = resid_col.unwrap_or(&default_resid_name);
    let resid_series = Series::new(PlSmallStr::from(resid_name), resid_values);
    data.with_column(resid_series)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加残差列失败: {}", e)))?;
    
    let stats = if return_stats {
        Some((coefficients, r_squared))
    } else {
        None
    };
    
    Ok((PyDataFrame(data), stats))
}

/// 数据清洗
/// 
/// 对数据进行去极值、中性化、标准化处理
/// 
/// # 参数
/// - df: 输入DataFrame
/// - col: 数据列名
/// - factor_col: 输出列名（默认为 "{col}_clean"）
/// - winsorize: 去极值方法 ("none", "mad", "sigma", "percentile")
/// - winsorize_n: 去极值倍数
/// - standardize: 标准化方法 ("none", "zscore", "minmax", "rank")
/// - neutralize: 中性化方法 ("none", "industry", "market_cap")
/// - industry_col: 行业列名（行业中性化需要）
/// - cap_col: 市值列名（市值中性化需要）
/// 
/// # 返回
/// 添加了清洗后数据列的DataFrame
#[pyfunction]
#[pyo3(signature = (df, col, factor_col=None, winsorize="none", winsorize_n=None, standardize="none", neutralize="none", industry_col=None, cap_col=None))]
pub fn clean(
    df: PyDataFrame,
    col: &str,
    factor_col: Option<&str>,
    winsorize: &str,
    winsorize_n: Option<f64>,
    standardize: &str,
    neutralize: &str,
    industry_col: Option<&str>,
    cap_col: Option<&str>,
) -> PyResult<PyDataFrame> {
    use std::collections::HashMap;
    
    let mut data: DataFrame = df.into();
    
    let default_name = format!("{}_clean", col);
    let col_name = factor_col.unwrap_or(&default_name);
    
    // 获取列数据
    let series = data.column(col)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("列 '{}' 不存在: {}", col, e)))?
        .cast(&DataType::Float64)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("转换失败: {}", e)))?;
    
    let values = series.f64()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取数值失败: {}", e)))?;
    
    let len = values.len();
    let mut cleaned_values = vec![f64::NAN; len];
    
    // 收集有效值
    let mut valid_values: Vec<f64> = Vec::new();
    for i in 0..len {
        if let Some(v) = values.get(i) {
            if !v.is_nan() {
                valid_values.push(v);
                cleaned_values[i] = v;
            }
        }
    }
    
    if valid_values.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("没有有效数据"));
    }
    
    // 步骤1: 去极值
    let (lower_bound, upper_bound) = match winsorize {
        "none" => (f64::NEG_INFINITY, f64::INFINITY),
        "mad" => {
            let n = winsorize_n.unwrap_or(3.0);
            let mut sorted = valid_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };
            
            let mut deviations: Vec<f64> = valid_values.iter().map(|&x| (x - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = if deviations.len() % 2 == 0 {
                (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
            } else {
                deviations[deviations.len() / 2]
            };
            
            (median - n * mad, median + n * mad)
        },
        "sigma" => {
            let n = winsorize_n.unwrap_or(3.0);
            let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            let variance: f64 = valid_values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (valid_values.len() - 1) as f64;
            let std = variance.sqrt();
            
            (mean - n * std, mean + n * std)
        },
        "percentile" => {
            let mut sorted = valid_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lower_idx = ((sorted.len() as f64 - 1.0) * 0.01).floor() as usize;
            let upper_idx = ((sorted.len() as f64 - 1.0) * 0.99).ceil() as usize;
            (sorted[lower_idx], sorted[upper_idx])
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("不支持的去极值方法: {}", winsorize)
            ));
        }
    };
    
    for i in 0..len {
        if !cleaned_values[i].is_nan() {
            if cleaned_values[i] < lower_bound {
                cleaned_values[i] = lower_bound;
            } else if cleaned_values[i] > upper_bound {
                cleaned_values[i] = upper_bound;
            }
        }
    }
    
    // 步骤2: 中性化
    if neutralize != "none" {
        match neutralize {
            "industry" => {
                if let Some(ind_col) = industry_col {
                    let industry_series = data.column(ind_col)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("行业列 '{}' 不存在: {}", ind_col, e)))?;
                    
                    let mut industry_sums: HashMap<String, (f64, usize)> = HashMap::new();
                    for i in 0..len {
                        if !cleaned_values[i].is_nan() {
                            if let Ok(ind) = industry_series.get(i) {
                                let ind_str = ind.to_string();
                                let entry = industry_sums.entry(ind_str).or_insert((0.0, 0));
                                entry.0 += cleaned_values[i];
                                entry.1 += 1;
                            }
                        }
                    }
                    
                    let mut industry_means: HashMap<String, f64> = HashMap::new();
                    for (ind, (sum, count)) in industry_sums.iter() {
                        industry_means.insert(ind.clone(), sum / *count as f64);
                    }
                    
                    for i in 0..len {
                        if !cleaned_values[i].is_nan() {
                            if let Ok(ind) = industry_series.get(i) {
                                let ind_str = ind.to_string();
                                if let Some(&mean) = industry_means.get(&ind_str) {
                                    cleaned_values[i] -= mean;
                                }
                            }
                        }
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("行业中性化需要提供industry_col"));
                }
            },
            "market_cap" => {
                if let Some(cap_col_name) = cap_col {
                    // 使用线性回归进行市值中性化
                    let temp_df = PyDataFrame(data.clone());
                    let (result_df, _) = linear(
                        temp_df,
                        vec![cap_col_name.to_string()],
                        col,
                        None,
                        Some("__temp_resid__"),
                        false,
                    )?;
                    
                    data = result_df.0;
                    let resid_series = data.column("__temp_resid__")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取残差失败: {}", e)))?;
                    
                    let resid_values = resid_series.f64()
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("获取残差数值失败: {}", e)))?;
                    
                    for i in 0..len {
                        if let Some(resid) = resid_values.get(i) {
                            if !resid.is_nan() {
                                cleaned_values[i] = resid;
                            }
                        }
                    }
                    
                    // 删除临时列
                    data = data.drop("__temp_resid__")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("删除临时列失败: {}", e)))?;
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("市值中性化需要提供cap_col"));
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("不支持的中性化方法: {}", neutralize)
                ));
            }
        }
    }
    
    // 步骤3: 标准化
    match standardize {
        "none" => {},
        "zscore" => {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..len {
                if !cleaned_values[i].is_nan() {
                    sum += cleaned_values[i];
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            
            let mut sum_sq = 0.0;
            for i in 0..len {
                if !cleaned_values[i].is_nan() {
                    sum_sq += (cleaned_values[i] - mean).powi(2);
                }
            }
            let std = if count > 1 { (sum_sq / (count - 1) as f64).sqrt() } else { 1.0 };
            
            if std > 0.0 {
                for i in 0..len {
                    if !cleaned_values[i].is_nan() {
                        cleaned_values[i] = (cleaned_values[i] - mean) / std;
                    }
                }
            }
        },
        "minmax" => {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..len {
                if !cleaned_values[i].is_nan() {
                    if cleaned_values[i] < min_val {
                        min_val = cleaned_values[i];
                    }
                    if cleaned_values[i] > max_val {
                        max_val = cleaned_values[i];
                    }
                }
            }
            
            let range = max_val - min_val;
            if range > 0.0 {
                for i in 0..len {
                    if !cleaned_values[i].is_nan() {
                        cleaned_values[i] = (cleaned_values[i] - min_val) / range;
                    }
                }
            }
        },
        "rank" => {
            let mut pairs: Vec<(usize, f64)> = Vec::new();
            for i in 0..len {
                if !cleaned_values[i].is_nan() {
                    pairs.push((i, cleaned_values[i]));
                }
            }
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let count = pairs.len() as f64;
            for (rank, (idx, _)) in pairs.iter().enumerate() {
                cleaned_values[*idx] = rank as f64 / (count - 1.0);
            }
        },
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("不支持的标准化方法: {}", standardize)
            ));
        }
    }
    
    let cleaned_series = Series::new(PlSmallStr::from(col_name), cleaned_values);
    data.with_column(cleaned_series)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("添加列失败: {}", e)))?;
    
    Ok(PyDataFrame(data))
}
