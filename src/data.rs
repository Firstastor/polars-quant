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
