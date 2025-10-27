use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyList};
use pyo3_polars::PyDataFrame;
use std::fs::{self, File};
use std::path::Path;

#[pyclass]
pub struct StockSelector {
    ohlcv_data: DataFrame,
    selected_symbols: Option<Vec<String>>,
}

#[pymethods]
impl StockSelector {
    #[new]
    pub fn new(ohlcv_data: PyDataFrame) -> PyResult<Self> {
        let df: DataFrame = ohlcv_data.into();
        
        if df.width() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "OHLCV 数据至少需要 2 列"
            ));
        }
        
        Ok(StockSelector {
            ohlcv_data: df,
            selected_symbols: None,
        })
    }
    
    #[staticmethod]
    #[pyo3(signature = (folder, file_type=None, prefix=None, suffix=None, has_header=true))]
    pub fn from_folder(
        _py: Python,
        folder: &str,
        file_type: Option<Bound<'_, PyAny>>,
        prefix: Option<&str>,
        suffix: Option<&str>,
        has_header: bool,
    ) -> PyResult<Self> {
        let dir = Path::new(folder);
        if !dir.is_dir() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("不是有效的目录: {}", folder)
            ));
        }

        let allowed_types: Vec<String> = if let Some(ft) = file_type {
            if let Ok(s) = ft.downcast::<PyString>() {
                vec![s.to_string()]
            } else if let Ok(list) = ft.downcast::<PyList>() {
                let mut types = Vec::new();
                for item in list.iter() {
                    if let Ok(s) = item.downcast::<PyString>() {
                        types.push(s.to_string());
                    }
                }
                types
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "file_type 必须是字符串或字符串列表"
                ));
            }
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

        let mut result_df: Option<DataFrame> = None;
        let mut file_count = 0;

        for entry in fs::read_dir(dir).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("读取目录失败: {}", e)
        ))? {
            let entry = entry.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("读取目录条目失败: {}", e)
            ))?;

            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let ext = match path.extension() {
                Some(e) => e.to_string_lossy().to_string(),
                None => continue,
            };

            if !allowed_types.contains(&ext) {
                continue;
            }

            let symbol = match path.file_stem() {
                Some(name) => name.to_string_lossy().to_string(),
                None => continue,
            };

            if let Some(pref) = prefix {
                if !symbol.starts_with(pref) {
                    continue;
                }
            }

            if let Some(suf) = suffix {
                if !symbol.ends_with(suf) {
                    continue;
                }
            }

            let mut df = match ext.as_str() {
                "parquet" => {
                    let file = File::open(&path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("打开文件失败 {}: {}", path.display(), e)
                        )
                    })?;

                    ParquetReader::new(file)
                        .finish()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("读取 parquet 文件失败 {}: {}", path.display(), e)
                            )
                        })?
                }
                "csv" => {
                    CsvReadOptions::default()
                        .with_has_header(has_header)
                        .try_into_reader_with_file_path(Some(path.clone()))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("创建 CSV 读取器失败 {}: {}", path.display(), e)
                            )
                        })?
                        .finish()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("读取 CSV 文件失败 {}: {}", path.display(), e)
                            )
                        })?
                }
                "xlsx" | "xls" => {
                    use calamine::{Reader, open_workbook_auto};
                    
                    let mut workbook = open_workbook_auto(&path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("打开 Excel 文件失败 {}: {}", path.display(), e)
                        )
                    })?;
                    
                    let sheet_names = workbook.sheet_names().to_owned();
                    if sheet_names.is_empty() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Excel 文件 {} 没有工作表", path.display())
                        ));
                    }
                    
                    let range = workbook.worksheet_range(&sheet_names[0]).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("读取 Excel 工作表失败 {}: {:?}", path.display(), e)
                        )
                    })?;
                    
                    let mut columns: Vec<Column> = Vec::new();
                    let (nrows, ncols) = range.get_size();
                    
                    if nrows == 0 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Excel 文件 {} 为空", path.display())
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
                            use calamine::DataType;
                            let value = range.get_value((row_idx as u32, col_idx as u32))
                                .and_then(|v| v.as_f64());
                            values.push(value);
                        }
                        
                        let series = Series::new(col_name.clone().into(), values);
                        columns.push(Column::new(series.name().clone(), series));
                    }
                    
                    DataFrame::new(columns).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("创建 DataFrame 失败 {}: {}", path.display(), e)
                        )
                    })?
                }
                "json" => {
                    JsonReader::new(File::open(&path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("打开 JSON 文件失败 {}: {}", path.display(), e)
                        )
                    })?)
                        .finish()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("读取 JSON 文件失败 {}: {}", path.display(), e)
                            )
                        })?
                }
                "feather" | "ipc" => {
                    IpcReader::new(File::open(&path).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("打开 IPC/Feather 文件失败 {}: {}", path.display(), e)
                        )
                    })?)
                        .finish()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("读取 IPC/Feather 文件失败 {}: {}", path.display(), e)
                            )
                        })?
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("不支持的文件类型: {}", ext)
                    ));
                }
            };

            let required_cols = vec!["date", "open", "high", "low", "close", "volume"];
            for col in &required_cols {
                if df.column(col).is_err() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("文件 {} 缺少必需的列: {}", path.display(), col)
                    ));
                }
            }

            let columns_to_rename = vec![
                ("open", format!("{}_open", symbol)),
                ("high", format!("{}_high", symbol)),
                ("low", format!("{}_low", symbol)),
                ("close", format!("{}_close", symbol)),
                ("volume", format!("{}_volume", symbol)),
            ];

            for (old_name, new_name) in columns_to_rename {
                df.rename(old_name, PlSmallStr::from(new_name.as_str()))
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("重命名列失败 {} -> {}: {}", old_name, new_name, e)
                        )
                    })?;
            }

            result_df = if let Some(existing) = result_df {
                let joined = existing
                    .join(
                        &df,
                        vec!["date"],
                        vec!["date"],
                        JoinArgs::new(JoinType::Full),
                        None,
                    )
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("合并 DataFrame 失败: {}", e)
                        )
                    })?;
                Some(joined)
            } else {
                Some(df)
            };

            file_count += 1;
        }

        if file_count == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("未找到匹配的文件 (支持的格式: {})", allowed_types.join(", "))
            ));
        }

        let df = result_df.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("加载文件失败")
        })?;
        
        Ok(StockSelector {
            ohlcv_data: df,
            selected_symbols: None,
        })
    }
    
    #[pyo3(signature = (
        price_min=None, price_max=None,
        volume_min=None, volume_avg_days=None,
        return_min=None, return_max=None, return_period=None,
        volatility_min=None, volatility_max=None, volatility_period=None,
        ma_above=None, ma_below=None,
        rsi_min=None, rsi_max=None, rsi_period=None,
        macd=None, macd_fast=12, macd_slow=26, macd_signal=9,
        kdj=None, kdj_period=9,
        limit_type=None, limit_threshold=9.9,
        volume_change=None, volume_multiplier=2.0, volume_change_days=5,
        consecutive=None, consecutive_days=3,
        breakout=None, breakout_period=20
    ))]
    pub fn filter<'a>(
        mut slf: PyRefMut<'a, Self>,
        price_min: Option<f64>,
        price_max: Option<f64>,
        volume_min: Option<f64>,
        volume_avg_days: Option<usize>,
        return_min: Option<f64>,
        return_max: Option<f64>,
        return_period: Option<usize>,
        volatility_min: Option<f64>,
        volatility_max: Option<f64>,
        volatility_period: Option<usize>,
        ma_above: Option<usize>,
        ma_below: Option<usize>,
        rsi_min: Option<f64>,
        rsi_max: Option<f64>,
        rsi_period: Option<usize>,
        macd: Option<&str>,
        macd_fast: usize,
        macd_slow: usize,
        macd_signal: usize,
        kdj: Option<&str>,
        kdj_period: usize,
        limit_type: Option<&str>,
        limit_threshold: f64,
        volume_change: Option<&str>,
        volume_multiplier: f64,
        volume_change_days: usize,
        consecutive: Option<&str>,
        consecutive_days: usize,
        breakout: Option<&str>,
        breakout_period: usize,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let mut candidates = if let Some(ref selected) = slf.selected_symbols {
            selected.clone()
        } else {
            slf.extract_stock_symbols()?
        };
        
        if price_min.is_some() || price_max.is_some() {
            candidates = slf.filter_by_price(&candidates, price_min, price_max)?;
        }
        
        if let Some(min_vol) = volume_min {
            candidates = slf.filter_by_volume(&candidates, min_vol, volume_avg_days)?;
        }
        
        if return_min.is_some() || return_max.is_some() {
            let period = return_period.unwrap_or(1);
            candidates = slf.filter_by_return(&candidates, return_min, return_max, period)?;
        }
        
        if volatility_min.is_some() || volatility_max.is_some() {
            let period = volatility_period.unwrap_or(20);
            candidates = slf.filter_by_volatility(&candidates, volatility_min, volatility_max, period)?;
        }
        
        if let Some(ma) = ma_above {
            candidates = slf.filter_by_ma(&candidates, ma, true)?;
        }
        if let Some(ma) = ma_below {
            candidates = slf.filter_by_ma(&candidates, ma, false)?;
        }
        
        if rsi_min.is_some() || rsi_max.is_some() {
            let period = rsi_period.unwrap_or(14);
            candidates = slf.filter_by_rsi(&candidates, period, rsi_min, rsi_max)?;
        }
        
        if let Some(condition) = macd {
            candidates = slf.filter_by_macd(&candidates, condition, macd_fast, macd_slow, macd_signal)?;
        }
        
        if let Some(condition) = kdj {
            candidates = slf.filter_by_kdj(&candidates, condition, kdj_period)?;
        }
        
        if let Some(condition) = limit_type {
            candidates = slf.filter_by_limit(&candidates, condition, limit_threshold)?;
        }
        
        if let Some(condition) = volume_change {
            candidates = slf.filter_by_volume_change(&candidates, condition, volume_multiplier, volume_change_days)?;
        }
        
        if let Some(condition) = consecutive {
            candidates = slf.filter_by_consecutive(&candidates, condition, consecutive_days)?;
        }
        
        if let Some(condition) = breakout {
            candidates = slf.filter_by_breakout(&candidates, condition, breakout_period)?;
        }
        
        slf.selected_symbols = Some(candidates);
        Ok(slf)
    }
    
    pub fn result(&self) -> PyResult<Vec<String>> {
        if let Some(ref symbols) = self.selected_symbols {
            Ok(symbols.clone())
        } else {
            self.extract_stock_symbols()
        }
    }
    
    pub fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.selected_symbols = None;
        slf
    }
    
    #[pyo3(signature = (by, ascending=false, top_n=None))]
    pub fn sort<'a>(
        mut slf: PyRefMut<'a, Self>,
        by: &str,
        ascending: bool,
        top_n: Option<usize>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let symbols = if let Some(ref selected) = slf.selected_symbols {
            selected.clone()
        } else {
            slf.extract_stock_symbols()?
        };
        
        let sorted = slf.rank_stocks_internal(&symbols, by, ascending, top_n)?;
        slf.selected_symbols = Some(sorted);
        Ok(slf)
    }
    
    pub fn info(&self) -> PyResult<PyDataFrame> {
        let symbols = if let Some(ref selected) = self.selected_symbols {
            selected.clone()
        } else {
            self.extract_stock_symbols()?
        };
        
        self.get_stock_info_internal(&symbols)
    }
}

impl StockSelector {
    fn filter_by_price(
        &self,
        symbols: &[String],
        min_price: Option<f64>,
        max_price: Option<f64>,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let price_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&price_col) {
                if let Ok(prices) = col.f64() {
                    if let Some(latest_price) = prices.last() {
                        let mut pass = true;
                        
                        if let Some(min) = min_price {
                            if latest_price < min {
                                pass = false;
                            }
                        }
                        
                        if let Some(max) = max_price {
                            if latest_price > max {
                                pass = false;
                            }
                        }
                        
                        if pass {
                            selected.push(symbol.clone());
                        }
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_volume(
        &self,
        symbols: &[String],
        min_volume: f64,
        avg_days: Option<usize>,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let volume_col = format!("{}_volume", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&volume_col) {
                if let Ok(volumes) = col.f64() {
                    let volume_value = if let Some(days) = avg_days {
                        let len = volumes.len();
                        if len == 0 {
                            continue;
                        }
                        let start = if len > days { len - days } else { 0 };
                        let slice = volumes.slice(start as i64, len - start);
                        slice.mean().unwrap_or(0.0)
                    } else {
                        volumes.last().unwrap_or(0.0)
                    };
                    
                    if volume_value >= min_volume {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_return(
        &self,
        symbols: &[String],
        min_return: Option<f64>,
        max_return: Option<f64>,
        period: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < period + 1 {
                        continue;
                    }
                    
                    let current_price = closes.get(len - 1).unwrap_or(0.0);
                    let prev_price = closes.get(len - 1 - period).unwrap_or(0.0);
                    
                    if prev_price <= 0.0 {
                        continue;
                    }
                    
                    let return_pct = (current_price - prev_price) / prev_price * 100.0;
                    
                    let mut pass = true;
                    if let Some(min) = min_return {
                        if return_pct < min {
                            pass = false;
                        }
                    }
                    
                    if let Some(max) = max_return {
                        if return_pct > max {
                            pass = false;
                        }
                    }
                    
                    if pass {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_volatility(
        &self,
        symbols: &[String],
        min_volatility: Option<f64>,
        max_volatility: Option<f64>,
        period: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < period + 1 {
                        continue;
                    }
                    
                    let mut returns = Vec::with_capacity(period);
                    for i in (len - period)..len {
                        let curr = closes.get(i).unwrap_or(0.0);
                        let prev = closes.get(i - 1).unwrap_or(0.0);
                        if prev > 0.0 {
                            returns.push((curr - prev) / prev);
                        }
                    }
                    
                    if returns.is_empty() {
                        continue;
                    }
                    
                    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                    let variance = returns.iter()
                        .map(|r| (r - mean).powi(2))
                        .sum::<f64>() / returns.len() as f64;
                    let volatility = variance.sqrt() * 100.0 * (252.0_f64).sqrt();
                    
                    let mut pass = true;
                    if let Some(min) = min_volatility {
                        if volatility < min {
                            pass = false;
                        }
                    }
                    
                    if let Some(max) = max_volatility {
                        if volatility > max {
                            pass = false;
                        }
                    }
                    
                    if pass {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_ma(
        &self,
        symbols: &[String],
        ma_period: usize,
        above: bool,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < ma_period {
                        continue;
                    }
                    
                    let ma_sum: f64 = (0..ma_period)
                        .filter_map(|i| closes.get(len - 1 - i))
                        .sum();
                    let ma = ma_sum / ma_period as f64;
                    
                    let current_price = closes.get(len - 1).unwrap_or(0.0);
                    
                    let condition_met = if above {
                        current_price > ma
                    } else {
                        current_price < ma
                    };
                    
                    if condition_met {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_rsi(
        &self,
        symbols: &[String],
        period: usize,
        min_rsi: Option<f64>,
        max_rsi: Option<f64>,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < period + 1 {
                        continue;
                    }
                    
                    let mut gains = Vec::new();
                    let mut losses = Vec::new();
                    
                    for i in (len - period)..len {
                        let curr = closes.get(i).unwrap_or(0.0);
                        let prev = closes.get(i - 1).unwrap_or(0.0);
                        let change = curr - prev;
                        
                        if change > 0.0 {
                            gains.push(change);
                            losses.push(0.0);
                        } else {
                            gains.push(0.0);
                            losses.push(change.abs());
                        }
                    }
                    
                    let avg_gain = gains.iter().sum::<f64>() / period as f64;
                    let avg_loss = losses.iter().sum::<f64>() / period as f64;
                    
                    let rsi = if avg_loss == 0.0 {
                        100.0
                    } else {
                        let rs = avg_gain / avg_loss;
                        100.0 - (100.0 / (1.0 + rs))
                    };
                    
                    let mut pass = true;
                    if let Some(min) = min_rsi {
                        if rsi < min {
                            pass = false;
                        }
                    }
                    
                    if let Some(max) = max_rsi {
                        if rsi > max {
                            pass = false;
                        }
                    }
                    
                    if pass {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_macd(
        &self,
        symbols: &[String],
        condition: &str,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < slow_period + signal_period {
                        continue;
                    }
                    
                    let mut ema_fast = Vec::with_capacity(len);
                    let mut ema_slow = Vec::with_capacity(len);
                    
                    let mut sum_fast = 0.0;
                    let mut sum_slow = 0.0;
                    for i in 0..fast_period.max(slow_period) {
                        let price = closes.get(i).unwrap_or(0.0);
                        if i < fast_period {
                            sum_fast += price;
                        }
                        if i < slow_period {
                            sum_slow += price;
                        }
                    }
                    ema_fast.push(sum_fast / fast_period as f64);
                    ema_slow.push(sum_slow / slow_period as f64);
                    
                    let alpha_fast = 2.0 / (fast_period as f64 + 1.0);
                    let alpha_slow = 2.0 / (slow_period as f64 + 1.0);
                    
                    for i in 1..len {
                        let price = closes.get(i).unwrap_or(0.0);
                        let prev_fast = ema_fast.last().unwrap();
                        let prev_slow = ema_slow.last().unwrap();
                        ema_fast.push(price * alpha_fast + prev_fast * (1.0 - alpha_fast));
                        ema_slow.push(price * alpha_slow + prev_slow * (1.0 - alpha_slow));
                    }
                    
                    let mut dif = Vec::with_capacity(len);
                    for i in 0..len {
                        dif.push(ema_fast[i] - ema_slow[i]);
                    }
                    
                    let mut dea = Vec::with_capacity(len);
                    let mut sum_dea = 0.0;
                    for i in 0..signal_period.min(dif.len()) {
                        sum_dea += dif[i];
                    }
                    dea.push(sum_dea / signal_period as f64);
                    
                    let alpha_signal = 2.0 / (signal_period as f64 + 1.0);
                    for i in 1..dif.len() {
                        let prev_dea = dea.last().unwrap();
                        dea.push(dif[i] * alpha_signal + prev_dea * (1.0 - alpha_signal));
                    }
                    
                    let macd = dif[dif.len()-1] - dea[dea.len()-1];
                    let prev_macd = if dif.len() > 1 {
                        dif[dif.len()-2] - dea[dea.len()-2]
                    } else {
                        0.0
                    };
                    
                    let condition_met = match condition {
                        "golden_cross" => prev_macd <= 0.0 && macd > 0.0,
                        "death_cross" => prev_macd >= 0.0 && macd < 0.0,
                        "above_zero" => macd > 0.0,
                        "below_zero" => macd < 0.0,
                        _ => false,
                    };
                    
                    if condition_met {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_kdj(
        &self,
        symbols: &[String],
        condition: &str,
        period: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            let high_col = format!("{}_high", symbol);
            let low_col = format!("{}_low", symbol);
            
            if let (Ok(close), Ok(high), Ok(low)) = (
                self.ohlcv_data.column(&close_col),
                self.ohlcv_data.column(&high_col),
                self.ohlcv_data.column(&low_col),
            ) {
                if let (Ok(closes), Ok(highs), Ok(lows)) = (close.f64(), high.f64(), low.f64()) {
                    let len = closes.len();
                    if len < period + 1 {
                        continue;
                    }
                    
                    let mut highest = f64::MIN;
                    let mut lowest = f64::MAX;
                    for i in (len-period)..len {
                        let h = highs.get(i).unwrap_or(0.0);
                        let l = lows.get(i).unwrap_or(0.0);
                        if h > highest { highest = h; }
                        if l < lowest { lowest = l; }
                    }
                    
                    let current_close = closes.get(len - 1).unwrap_or(0.0);
                    let prev_close = closes.get(len - 2).unwrap_or(0.0);
                    
                    let rsv = if highest - lowest > 0.0 {
                        (current_close - lowest) / (highest - lowest) * 100.0
                    } else {
                        50.0
                    };
                    
                    let prev_rsv = if highest - lowest > 0.0 {
                        (prev_close - lowest) / (highest - lowest) * 100.0
                    } else {
                        50.0
                    };
                    
                    let k = rsv;
                    let d = (rsv + prev_rsv) / 2.0;
                    let prev_k = prev_rsv;
                    
                    let condition_met = match condition {
                        "golden_cross" => prev_k <= d && k > d,
                        "death_cross" => prev_k >= d && k < d,
                        "oversold" => k < 20.0 && d < 20.0,
                        "overbought" => k > 80.0 && d > 80.0,
                        _ => false,
                    };
                    
                    if condition_met {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_limit(
        &self,
        symbols: &[String],
        condition: &str,
        threshold: f64,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < 2 {
                        continue;
                    }
                    
                    let current = closes.get(len - 1).unwrap_or(0.0);
                    let prev = closes.get(len - 2).unwrap_or(0.0);
                    
                    if prev <= 0.0 {
                        continue;
                    }
                    
                    let change_pct = (current - prev) / prev * 100.0;
                    
                    let condition_met = match condition {
                        "limit_up" => change_pct >= threshold,
                        "limit_down" => change_pct <= -threshold,
                        "near_limit_up" => change_pct >= threshold * 0.8,
                        "near_limit_down" => change_pct <= -threshold * 0.8,
                        _ => false,
                    };
                    
                    if condition_met {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_volume_change(
        &self,
        symbols: &[String],
        condition: &str,
        multiplier: f64,
        avg_days: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let volume_col = format!("{}_volume", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&volume_col) {
                if let Ok(volumes) = col.f64() {
                    let len = volumes.len();
                    if len < avg_days + 1 {
                        continue;
                    }
                    
                    let current_volume = volumes.get(len - 1).unwrap_or(0.0);
                    
                    let mut sum = 0.0;
                    for i in (len - 1 - avg_days)..(len - 1) {
                        sum += volumes.get(i).unwrap_or(0.0);
                    }
                    let avg_volume = sum / avg_days as f64;
                    
                    if avg_volume <= 0.0 {
                        continue;
                    }
                    
                    let condition_met = match condition {
                        "volume_surge" => current_volume >= avg_volume * multiplier,
                        "volume_shrink" => current_volume <= avg_volume / multiplier,
                        _ => false,
                    };
                    
                    if condition_met {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_consecutive(
        &self,
        symbols: &[String],
        condition: &str,
        days: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            
            if let Ok(col) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = col.f64() {
                    let len = closes.len();
                    if len < days + 1 {
                        continue;
                    }
                    
                    let mut consecutive_count = 0;
                    let check_up = condition == "consecutive_up";
                    
                    for i in (len - days)..len {
                        let current = closes.get(i).unwrap_or(0.0);
                        let prev = closes.get(i - 1).unwrap_or(0.0);
                        
                        if prev <= 0.0 {
                            break;
                        }
                        
                        let is_up = current > prev;
                        if is_up == check_up {
                            consecutive_count += 1;
                        } else {
                            break;
                        }
                    }
                    
                    if consecutive_count >= days {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn filter_by_breakout(
        &self,
        symbols: &[String],
        condition: &str,
        period: usize,
    ) -> PyResult<Vec<String>> {
        let mut selected = Vec::new();
        
        for symbol in symbols {
            let high_col = format!("{}_high", symbol);
            let low_col = format!("{}_low", symbol);
            let close_col = format!("{}_close", symbol);
            
            if let (Ok(high), Ok(low), Ok(close)) = (
                self.ohlcv_data.column(&high_col),
                self.ohlcv_data.column(&low_col),
                self.ohlcv_data.column(&close_col),
            ) {
                if let (Ok(highs), Ok(lows), Ok(closes)) = (high.f64(), low.f64(), close.f64()) {
                    let len = closes.len();
                    if len < period + 1 {
                        continue;
                    }
                    
                    let current_close = closes.get(len - 1).unwrap_or(0.0);
                    
                    let mut highest = f64::MIN;
                    let mut lowest = f64::MAX;
                    for i in (len - 1 - period)..(len - 1) {
                        let h = highs.get(i).unwrap_or(0.0);
                        let l = lows.get(i).unwrap_or(f64::MAX);
                        if h > highest { highest = h; }
                        if l < lowest { lowest = l; }
                    }
                    
                    let condition_met = match condition {
                        "breakout_high" => current_close > highest,
                        "breakdown_low" => current_close < lowest,
                        _ => false,
                    };
                    
                    if condition_met {
                        selected.push(symbol.clone());
                    }
                }
            }
        }
        
        Ok(selected)
    }
    
    fn rank_stocks_internal(
        &self,
        symbols: &[String],
        sort_by: &str,
        ascending: bool,
        top_n: Option<usize>,
    ) -> PyResult<Vec<String>> {
        let mut stock_values: Vec<(String, f64)> = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            let volume_col = format!("{}_volume", symbol);
            
            if let Ok(close) = self.ohlcv_data.column(&close_col) {
                if let Ok(closes) = close.f64() {
                    let len = closes.len();
                    
                    let value = match sort_by {
                        "price" => closes.get(len - 1).unwrap_or(0.0),
                        "return_1d" => {
                            if len < 2 { continue; }
                            let curr = closes.get(len - 1).unwrap_or(0.0);
                            let prev = closes.get(len - 2).unwrap_or(0.0);
                            if prev > 0.0 { (curr - prev) / prev * 100.0 } else { 0.0 }
                        },
                        "return_5d" => {
                            if len < 6 { continue; }
                            let curr = closes.get(len - 1).unwrap_or(0.0);
                            let prev = closes.get(len - 6).unwrap_or(0.0);
                            if prev > 0.0 { (curr - prev) / prev * 100.0 } else { 0.0 }
                        },
                        "return_20d" => {
                            if len < 21 { continue; }
                            let curr = closes.get(len - 1).unwrap_or(0.0);
                            let prev = closes.get(len - 21).unwrap_or(0.0);
                            if prev > 0.0 { (curr - prev) / prev * 100.0 } else { 0.0 }
                        },
                        "volume" => {
                            if let Ok(vol) = self.ohlcv_data.column(&volume_col) {
                                if let Ok(volumes) = vol.f64() {
                                    volumes.get(len - 1).unwrap_or(0.0)
                                } else { continue; }
                            } else { continue; }
                        },
                        "volatility" => {
                            if len < 21 { continue; }
                            let mut returns = Vec::new();
                            for i in (len - 20)..len {
                                let curr = closes.get(i).unwrap_or(0.0);
                                let prev = closes.get(i - 1).unwrap_or(0.0);
                                if prev > 0.0 {
                                    returns.push((curr - prev) / prev);
                                }
                            }
                            if returns.is_empty() { continue; }
                            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                            let variance = returns.iter()
                                .map(|r| (r - mean).powi(2))
                                .sum::<f64>() / returns.len() as f64;
                            variance.sqrt() * 100.0 * (252.0_f64).sqrt()
                        },
                        _ => continue,
                    };
                    
                    stock_values.push((symbol.clone(), value));
                }
            }
        }
        
        if ascending {
            stock_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            stock_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        let result: Vec<String> = if let Some(n) = top_n {
            stock_values.iter().take(n).map(|(s, _)| s.clone()).collect()
        } else {
            stock_values.iter().map(|(s, _)| s.clone()).collect()
        };
        
        Ok(result)
    }
    
    fn get_stock_info_internal(&self, symbols: &[String]) -> PyResult<PyDataFrame> {
        let mut symbol_list = Vec::new();
        let mut prices = Vec::new();
        let mut open_prices = Vec::new();
        let mut high_prices = Vec::new();
        let mut low_prices = Vec::new();
        let mut volumes = Vec::new();
        let mut returns_1d = Vec::new();
        let mut returns_5d = Vec::new();
        let mut returns_20d = Vec::new();
        let mut volatility_20d = Vec::new();
        let mut ma_5 = Vec::new();
        let mut ma_10 = Vec::new();
        let mut ma_20 = Vec::new();
        let mut volume_ratio = Vec::new();
        let mut amplitude = Vec::new();
        
        for symbol in symbols {
            let close_col = format!("{}_close", symbol);
            let open_col = format!("{}_open", symbol);
            let high_col = format!("{}_high", symbol);
            let low_col = format!("{}_low", symbol);
            let volume_col = format!("{}_volume", symbol);
            
            if let (Ok(close), Ok(open), Ok(high), Ok(low), Ok(volume)) = (
                self.ohlcv_data.column(&close_col),
                self.ohlcv_data.column(&open_col),
                self.ohlcv_data.column(&high_col),
                self.ohlcv_data.column(&low_col),
                self.ohlcv_data.column(&volume_col)
            ) {
                if let (Ok(closes), Ok(opens), Ok(highs), Ok(lows), Ok(vols)) = 
                    (close.f64(), open.f64(), high.f64(), low.f64(), volume.f64()) 
                {
                    let len = closes.len();
                    if len < 21 {
                        continue;
                    }
                    
                    let current_price = closes.get(len - 1).unwrap_or(0.0);
                    let current_open = opens.get(len - 1).unwrap_or(0.0);
                    let current_high = highs.get(len - 1).unwrap_or(0.0);
                    let current_low = lows.get(len - 1).unwrap_or(0.0);
                    let current_volume = vols.get(len - 1).unwrap_or(0.0);
                    
                    // 收益率计算
                    let prev_1d = closes.get(len - 2).unwrap_or(0.0);
                    let ret_1d = if prev_1d > 0.0 {
                        (current_price - prev_1d) / prev_1d * 100.0
                    } else {
                        0.0
                    };
                    
                    let prev_5d = closes.get(len - 6).unwrap_or(0.0);
                    let ret_5d = if prev_5d > 0.0 {
                        (current_price - prev_5d) / prev_5d * 100.0
                    } else {
                        0.0
                    };
                    
                    let prev_20d = closes.get(len - 21).unwrap_or(0.0);
                    let ret_20d = if prev_20d > 0.0 {
                        (current_price - prev_20d) / prev_20d * 100.0
                    } else {
                        0.0
                    };
                    
                    // 波动率计算（20日年化波动率）
                    let mut returns = Vec::new();
                    for i in (len - 20)..len {
                        let curr = closes.get(i).unwrap_or(0.0);
                        let prev = closes.get(i - 1).unwrap_or(0.0);
                        if prev > 0.0 {
                            returns.push((curr - prev) / prev);
                        }
                    }
                    let volatility = if !returns.is_empty() {
                        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                        let variance = returns.iter()
                            .map(|r| (r - mean).powi(2))
                            .sum::<f64>() / returns.len() as f64;
                        variance.sqrt() * 100.0 * (252.0_f64).sqrt()
                    } else {
                        0.0
                    };
                    
                    // 均线计算
                    let calc_ma = |period: usize| -> f64 {
                        if len < period {
                            return 0.0;
                        }
                        let sum: f64 = (0..period)
                            .filter_map(|i| closes.get(len - 1 - i))
                            .sum();
                        sum / period as f64
                    };
                    
                    let ma5 = calc_ma(5);
                    let ma10 = calc_ma(10);
                    let ma20 = calc_ma(20);
                    
                    // 量比（当前成交量 / 5日平均成交量）
                    let avg_volume_5d = if len >= 6 {
                        let sum: f64 = (1..6)
                            .filter_map(|i| vols.get(len - 1 - i))
                            .sum();
                        sum / 5.0
                    } else {
                        current_volume
                    };
                    let vol_ratio = if avg_volume_5d > 0.0 {
                        current_volume / avg_volume_5d
                    } else {
                        1.0
                    };
                    
                    // 振幅（当日最高最低价差 / 收盘价）
                    let amp = if current_price > 0.0 {
                        (current_high - current_low) / current_price * 100.0
                    } else {
                        0.0
                    };
                    
                    symbol_list.push(symbol.clone());
                    prices.push(current_price);
                    open_prices.push(current_open);
                    high_prices.push(current_high);
                    low_prices.push(current_low);
                    volumes.push(current_volume);
                    returns_1d.push(ret_1d);
                    returns_5d.push(ret_5d);
                    returns_20d.push(ret_20d);
                    volatility_20d.push(volatility);
                    ma_5.push(ma5);
                    ma_10.push(ma10);
                    ma_20.push(ma20);
                    volume_ratio.push(vol_ratio);
                    amplitude.push(amp);
                }
            }
        }
        
        let df = DataFrame::new(vec![
            Column::new("symbol".into(), symbol_list),
            Column::new("price".into(), prices),
            Column::new("open".into(), open_prices),
            Column::new("high".into(), high_prices),
            Column::new("low".into(), low_prices),
            Column::new("volume".into(), volumes),
            Column::new("return_1d".into(), returns_1d),
            Column::new("return_5d".into(), returns_5d),
            Column::new("return_20d".into(), returns_20d),
            Column::new("volatility".into(), volatility_20d),
            Column::new("ma_5".into(), ma_5),
            Column::new("ma_10".into(), ma_10),
            Column::new("ma_20".into(), ma_20),
            Column::new("volume_ratio".into(), volume_ratio),
            Column::new("amplitude".into(), amplitude),
        ]).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("创建 DataFrame 失败: {}", e)
        ))?;
        
        Ok(PyDataFrame(df))
    }
    
    fn extract_stock_symbols(&self) -> PyResult<Vec<String>> {
        let columns = self.ohlcv_data.get_column_names();
        let mut symbols = std::collections::HashSet::new();
        
        for col_name in columns {
            if col_name == "date" {
                continue;
            }
            
            if let Some(pos) = col_name.rfind('_') {
                let symbol = &col_name[..pos];
                symbols.insert(symbol.to_string());
            }
        }
        
        Ok(symbols.into_iter().collect())
    }
}
