use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use chrono::NaiveDate;

#[derive(Debug, Clone)]
struct Position {
    symbol: String,
    entry_date: String,
    entry_price: f64,
    quantity: f64,
    exit_date: Option<String>,
    exit_price: Option<f64>,
    pnl: Option<f64>,
    pnl_pct: Option<f64>,
    holding_days: Option<i32>,
}

/// 单只股票的回测结果
#[derive(Debug, Clone)]
struct StockBacktestResult {
    symbol: String,
    daily_dates: Vec<String>,
    daily_cash: Vec<f64>,
    daily_stock_value: Vec<f64>,
    daily_total_value: Vec<f64>,
    positions: Vec<Position>,
}

#[pyclass]
pub struct Backtest {
    prices: DataFrame,
    buy_signals: DataFrame,
    sell_signals: DataFrame,
    initial_capital: f64,
    commission_rate: f64,
    min_commission: f64,
    slippage: f64,
    daily_records: Option<DataFrame>,
    position_records: Option<DataFrame>,
}

#[pymethods]
impl Backtest {
    #[new]
    #[pyo3(signature = (prices, buy_signals, sell_signals, initial_capital=100000.0, commission_rate=0.0003, min_commission=5.0, slippage=0.0))]
    pub fn new(
        prices: PyDataFrame,
        buy_signals: PyDataFrame,
        sell_signals: PyDataFrame,
        initial_capital: f64,
        commission_rate: f64,
        min_commission: f64,
        slippage: f64,
    ) -> PyResult<Self> {
        let prices_df: DataFrame = prices.into();
        let buy_df: DataFrame = buy_signals.into();
        let sell_df: DataFrame = sell_signals.into();
        
        if prices_df.width() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "prices必须至少有2列"
            ));
        }
        
        if buy_df.width() != prices_df.width() || sell_df.width() != prices_df.width() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "三个DataFrame列数必须一致"
            ));
        }
        
        Ok(Backtest {
            prices: prices_df,
            buy_signals: buy_df,
            sell_signals: sell_df,
            initial_capital,
            commission_rate,
            min_commission,
            slippage,
            daily_records: None,
            position_records: None,
        })
    }
    
    pub fn get_daily_records(&self) -> PyResult<PyDataFrame> {
        match &self.daily_records {
            Some(df) => Ok(PyDataFrame(df.clone())),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "请先运行run()方法"
            )),
        }
    }
    
    pub fn get_position_records(&self) -> PyResult<PyDataFrame> {
        match &self.position_records {
            Some(df) => Ok(PyDataFrame(df.clone())),
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "没有交易记录"
            )),
        }
    }
    
    /// 获取单只股票的每日资金记录
    pub fn get_stock_daily(&self, symbol: &str) -> PyResult<PyDataFrame> {
        match &self.daily_records {
            Some(df) => {
                let filtered = df.filter(
                    &df.column("symbol")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("获取symbol列失败: {}", e)
                        ))?
                        .str()
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("转换为字符串类型失败: {}", e)
                        ))?
                        .equal(symbol)
                ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("筛选DataFrame失败: {}", e)
                ))?;
                
                Ok(PyDataFrame(filtered))
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "请先运行run()方法"
            )),
        }
    }
    
    /// 获取单只股票的交易记录
    pub fn get_stock_positions(&self, symbol: &str) -> PyResult<PyDataFrame> {
        match &self.position_records {
            Some(df) => {
                let filtered = df.filter(
                    &df.column("symbol")
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("获取symbol列失败: {}", e)
                        ))?
                        .str()
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("转换为字符串类型失败: {}", e)
                        ))?
                        .equal(symbol)
                ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("筛选DataFrame失败: {}", e)
                ))?;
                
                Ok(PyDataFrame(filtered))
            },
            None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "没有交易记录"
            )),
        }
    }
    
    /// 获取单只股票的统计摘要
    pub fn get_stock_summary(&self, symbol: &str) -> PyResult<String> {
        if self.position_records.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "请先运行run()方法"
            ));
        }
        
        let pos_df = self.position_records.as_ref().unwrap();
        
        // 筛选该股票的交易
        let stock_positions = pos_df.filter(
            &pos_df.column("symbol")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("获取symbol列失败: {}", e)
                ))?
                .str()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("转换为字符串类型失败: {}", e)
                ))?
                .equal(symbol)
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("筛选DataFrame失败: {}", e)
        ))?;
        
        if stock_positions.height() == 0 {
            return Ok(format!("股票 {} 没有交易记录", symbol));
        }
        
        // 统计数据
        let total_trades = stock_positions.height();
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut total_pnl = 0.0;
        let mut total_win = 0.0;
        let mut total_loss = 0.0;
        let mut max_win = 0.0;
        let mut max_loss = 0.0;
        
        if let Ok(pnl_series) = stock_positions.column("pnl") {
            if let Ok(pnl_values) = pnl_series.f64() {
                for i in 0..pnl_values.len() {
                    if let Some(pnl) = pnl_values.get(i) {
                        total_pnl += pnl;
                        if pnl > 0.01 {
                            winning_trades += 1;
                            total_win += pnl;
                            if pnl > max_win {
                                max_win = pnl;
                            }
                        } else if pnl < -0.01 {
                            losing_trades += 1;
                            total_loss += pnl.abs();
                            if pnl < max_loss {
                                max_loss = pnl;
                            }
                        }
                    }
                }
            }
        }
        
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };
        
        let avg_win = if winning_trades > 0 {
            total_win / winning_trades as f64
        } else {
            0.0
        };
        
        let avg_loss = if losing_trades > 0 {
            total_loss / losing_trades as f64
        } else {
            0.0
        };
        
        Ok(format!(
            r#"
========================================
  股票 {} 回测摘要
========================================

交易次数: {}
盈利交易: {} ({:.2}%)
亏损交易: {} ({:.2}%)

总盈亏: {:.2}
胜率: {:.2}%

盈利交易:
  总盈利: {:.2}
  平均盈利: {:.2}
  最大盈利: {:.2}

亏损交易:
  总亏损: {:.2}
  平均亏损: {:.2}
  最大亏损: {:.2}

========================================
"#,
            symbol,
            total_trades,
            winning_trades, win_rate,
            losing_trades, if total_trades > 0 { losing_trades as f64 / total_trades as f64 * 100.0 } else { 0.0 },
            total_pnl,
            win_rate,
            total_win,
            avg_win,
            max_win,
            total_loss,
            avg_loss,
            max_loss,
        ))
    }
    
    /// 运行回测
    pub fn run(&mut self) -> PyResult<()> {
        let dates = self.prices.column("date")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("获取date列失败: {}", e)
            ))?;
        
        // 尝试获取日期列，支持字符串或日期类型
        let date_strings: Vec<String> = if let Ok(str_dates) = dates.str() {
            // 字符串类型
            str_dates.iter()
                .map(|opt| opt.unwrap_or("").to_string())
                .collect()
        } else {
            // 尝试转换为字符串
            dates.cast(&DataType::String)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("无法将date列转换为字符串: {}", e)
                ))?
                .str()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("转换后获取字符串失败: {}", e)
                ))?
                .iter()
                .map(|opt| opt.unwrap_or("").to_string())
                .collect()
        };
        
        let stock_names: Vec<String> = self.prices.get_column_names()
            .iter()
            .skip(1)
            .map(|s| s.to_string())
            .collect();
        
        let num_stocks = stock_names.len();
        
        // ========== 智能并行策略 ==========
        // 根据任务数和CPU核心数智能决定是否并行
        let use_parallel = num_stocks >= 4; // 至少4只股票才并行
        
        // 如果需要并行，动态调整线程池大小
        let stock_results: Vec<StockBacktestResult> = if use_parallel {
            // 计算最优线程数：min(股票数, CPU核心数)
            let num_cpus = rayon::current_num_threads();
            let optimal_threads = num_stocks.min(num_cpus);
            
            // 使用自定义线程池
            rayon::ThreadPoolBuilder::new()
                .num_threads(optimal_threads)
                .build()
                .unwrap()
                .install(|| {
                    stock_names.par_iter()
                        .filter_map(|stock_name| {
                            // 提取该股票的完整时间序列
                            let price_col = self.prices.column(stock_name).ok()?;
                            let prices = price_col.f64().ok()?;
                            let price_vec: Vec<f64> = prices.into_iter()
                                .map(|opt| opt.unwrap_or(f64::NAN))
                                .collect();
                            
                            let buy_col = self.buy_signals.column(stock_name).ok()?;
                            let buy_signals = buy_col.bool().ok()?;
                            let buy_vec: Vec<bool> = buy_signals.into_iter()
                                .map(|opt| opt.unwrap_or(false))
                                .collect();
                            
                            let sell_col = self.sell_signals.column(stock_name).ok()?;
                            let sell_signals = sell_col.bool().ok()?;
                            let sell_vec: Vec<bool> = sell_signals.into_iter()
                                .map(|opt| opt.unwrap_or(false))
                                .collect();
                            
                            // 早期退出：如果没有任何买入信号，跳过回测
                            if !buy_vec.iter().any(|&b| b) {
                                return None;
                            }
                            
                            // 对该股票进行独立回测
                            Some(Self::backtest_single_stock(
                                stock_name.clone(),
                                &date_strings,
                                &price_vec,
                                &buy_vec,
                                &sell_vec,
                                self.initial_capital,
                                self.commission_rate,
                                self.min_commission,
                                self.slippage,
                            ))
                        })
                        .collect()
                })
        } else {
            // 串行处理（小数据集）
            stock_names.iter()
                .filter_map(|stock_name| {
                    let price_col = self.prices.column(stock_name).ok()?;
                    let prices = price_col.f64().ok()?;
                    let price_vec: Vec<f64> = prices.into_iter()
                        .map(|opt| opt.unwrap_or(f64::NAN))
                        .collect();
                    
                    let buy_col = self.buy_signals.column(stock_name).ok()?;
                    let buy_signals = buy_col.bool().ok()?;
                    let buy_vec: Vec<bool> = buy_signals.into_iter()
                        .map(|opt| opt.unwrap_or(false))
                        .collect();
                    
                    let sell_col = self.sell_signals.column(stock_name).ok()?;
                    let sell_signals = sell_col.bool().ok()?;
                    let sell_vec: Vec<bool> = sell_signals.into_iter()
                        .map(|opt| opt.unwrap_or(false))
                        .collect();
                    
                    // 早期退出：如果没有任何买入信号，跳过回测
                    if !buy_vec.iter().any(|&b| b) {
                        return None;
                    }
                    
                    Some(Self::backtest_single_stock(
                        stock_name.clone(),
                        &date_strings,
                        &price_vec,
                        &buy_vec,
                        &sell_vec,
                        self.initial_capital,
                        self.commission_rate,
                        self.min_commission,
                        self.slippage,
                    ))
                })
                .collect()
        };
        
        // ========== 优化内存分配：预先计算总容量 ==========
        let total_capacity: usize = stock_results.iter()
            .map(|r| r.daily_dates.len())
            .sum();
        
        let mut all_symbols: Vec<String> = Vec::with_capacity(total_capacity);
        let mut all_dates: Vec<String> = Vec::with_capacity(total_capacity);
        let mut all_cash: Vec<f64> = Vec::with_capacity(total_capacity);
        let mut all_stock_value: Vec<f64> = Vec::with_capacity(total_capacity);
        let mut all_total_value: Vec<f64> = Vec::with_capacity(total_capacity);
        
        let total_positions: usize = stock_results.iter()
            .map(|r| r.positions.len())
            .sum();
        let mut all_positions: Vec<Position> = Vec::with_capacity(total_positions);
        
        // ========== 优化数据聚合：减少clone ==========
        for result in stock_results {
            let symbol = result.symbol;
            let n_records = result.daily_dates.len();
            
            // 扩展symbol（使用repeat避免逐个clone）
            all_symbols.extend(std::iter::repeat(symbol).take(n_records));
            
            // 移动而非克隆
            all_dates.extend(result.daily_dates);
            all_cash.extend(result.daily_cash);
            all_stock_value.extend(result.daily_stock_value);
            all_total_value.extend(result.daily_total_value);
            all_positions.extend(result.positions);
        }
        
        // 构建每日资金记录DataFrame
        self.daily_records = Some(DataFrame::new(vec![
            Column::new("symbol".into(), all_symbols),
            Column::new("date".into(), all_dates),
            Column::new("cash".into(), all_cash),
            Column::new("stock_value".into(), all_stock_value),
            Column::new("total_value".into(), all_total_value),
        ]).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("创建daily_records失败: {}", e)
        ))?);
        
        // 构建持仓记录DataFrame
        if !all_positions.is_empty() {
            // 优化：单次遍历收集所有字段
            let n_pos = all_positions.len();
            let mut symbols = Vec::with_capacity(n_pos);
            let mut entry_dates = Vec::with_capacity(n_pos);
            let mut entry_prices = Vec::with_capacity(n_pos);
            let mut quantities = Vec::with_capacity(n_pos);
            let mut exit_dates = Vec::with_capacity(n_pos);
            let mut exit_prices = Vec::with_capacity(n_pos);
            let mut pnls = Vec::with_capacity(n_pos);
            let mut pnl_pcts = Vec::with_capacity(n_pos);
            let mut holding_days = Vec::with_capacity(n_pos);
            
            for pos in all_positions {
                symbols.push(pos.symbol);
                entry_dates.push(pos.entry_date);
                entry_prices.push(pos.entry_price);
                quantities.push(pos.quantity);
                exit_dates.push(pos.exit_date);
                exit_prices.push(pos.exit_price);
                pnls.push(pos.pnl);
                pnl_pcts.push(pos.pnl_pct);
                holding_days.push(pos.holding_days);
            }
            
            self.position_records = Some(DataFrame::new(vec![
                Column::new("symbol".into(), symbols),
                Column::new("entry_date".into(), entry_dates),
                Column::new("entry_price".into(), entry_prices),
                Column::new("quantity".into(), quantities),
                Column::new("exit_date".into(), exit_dates),
                Column::new("exit_price".into(), exit_prices),
                Column::new("pnl".into(), pnls),
                Column::new("pnl_pct".into(), pnl_pcts),
                Column::new("holding_days".into(), holding_days),
            ]).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("创建position_records失败: {}", e)
            ))?);
        }
        
        Ok(())
    }
    
    /// 输出回测统计摘要
    pub fn summary(&self) -> PyResult<()> {
        if self.daily_records.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "请先运行run()方法"
            ));
        }
        
        let daily_df = self.daily_records.as_ref().unwrap();
        let total_value_series = daily_df.column("total_value")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("获取total_value列失败: {}", e)
            ))?;
        
        let total_values = total_value_series.f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("total_value必须是f64: {}", e)
            ))?;
        
        let initial = self.initial_capital;
        let final_value = total_values.get(total_values.len() - 1).unwrap_or(initial);
        let total_return = (final_value - initial) / initial;
        
        let n_days = total_values.len();
        let years = n_days as f64 / 252.0;
        let annual_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };
        
        // 计算夏普比率
        let mut daily_returns: Vec<f64> = Vec::with_capacity(n_days - 1);
        for i in 1..n_days {
            let prev = total_values.get(i - 1).unwrap_or(initial);
            let curr = total_values.get(i).unwrap_or(initial);
            if prev > 0.0 {
                daily_returns.push((curr - prev) / prev);
            }
        }
        
        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / daily_returns.len() as f64;
        let std_return = variance.sqrt();
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * (252.0_f64).sqrt()
        } else {
            0.0
        };
        
        // 计算最大回撤
        let mut max_value = initial;
        let mut max_drawdown = 0.0;
        let mut current_dd_days = 0;
        let mut max_dd_duration = 0;
        let mut in_drawdown = false;
        
        for i in 0..n_days {
            let value = total_values.get(i).unwrap_or(initial);
            if value > max_value {
                max_value = value;
                if in_drawdown && current_dd_days > max_dd_duration {
                    max_dd_duration = current_dd_days;
                }
                in_drawdown = false;
                current_dd_days = 0;
            } else {
                in_drawdown = true;
                current_dd_days += 1;
            }
            
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        if in_drawdown && current_dd_days > max_dd_duration {
            max_dd_duration = current_dd_days;
        }
        
        // 交易统计
        let mut total_trades = 0;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut break_even_trades = 0;
        let mut total_win = 0.0;
        let mut total_loss = 0.0;
        let mut largest_win = 0.0;
        let mut largest_loss = 0.0;
        let mut largest_win_pct = 0.0;
        let mut largest_loss_pct = 0.0;
        let mut total_holding_days = 0;
        let mut total_commission = 0.0;
        let mut consecutive_wins = 0;
        let mut consecutive_losses = 0;
        let mut max_consecutive_wins = 0;
        let mut max_consecutive_losses = 0;
        let mut win_holding_days = 0;
        let mut loss_holding_days = 0;
        let mut total_buy_value = 0.0;
        let mut total_sell_value = 0.0;
        
        // 按股票统计
        let mut stock_trades: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut stock_wins: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut stock_pnl: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        
        if let Some(pos_df) = &self.position_records {
            total_trades = pos_df.height();
            
            // 获取各列数据
            let pnl_series = pos_df.column("pnl").ok();
            let pnl_pct_series = pos_df.column("pnl_pct").ok();
            let days_series = pos_df.column("holding_days").ok();
            let symbol_series = pos_df.column("symbol").ok();
            let quantity_series = pos_df.column("quantity").ok();
            let entry_price_series = pos_df.column("entry_price").ok();
            let exit_price_series = pos_df.column("exit_price").ok();
            
            if let (Some(pnl_col), Some(pnl_pct_col)) = (pnl_series, pnl_pct_series) {
                if let (Ok(pnl_values), Ok(pnl_pct_values)) = (pnl_col.f64(), pnl_pct_col.f64()) {
                    for i in 0..pnl_values.len() {
                        if let (Some(pnl), Some(pnl_pct)) = (pnl_values.get(i), pnl_pct_values.get(i)) {
                            // 统计盈亏
                            if pnl > 0.01 {
                                winning_trades += 1;
                                total_win += pnl;
                                consecutive_wins += 1;
                                consecutive_losses = 0;
                                if consecutive_wins > max_consecutive_wins {
                                    max_consecutive_wins = consecutive_wins;
                                }
                                if pnl > largest_win {
                                    largest_win = pnl;
                                }
                                if pnl_pct > largest_win_pct {
                                    largest_win_pct = pnl_pct;
                                }
                                
                                // 盈利持仓天数
                                if let Some(days_col) = &days_series {
                                    if let Ok(days_values) = days_col.i32() {
                                        if let Some(days) = days_values.get(i) {
                                            win_holding_days += days as usize;
                                        }
                                    }
                                }
                                
                                // 按股票统计盈利次数
                                if let Some(symbol_col) = &symbol_series {
                                    if let Ok(symbol_values) = symbol_col.str() {
                                        if let Some(symbol) = symbol_values.get(i) {
                                            *stock_wins.entry(symbol.to_string()).or_insert(0) += 1;
                                        }
                                    }
                                }
                            } else if pnl < -0.01 {
                                losing_trades += 1;
                                total_loss += pnl.abs();
                                consecutive_losses += 1;
                                consecutive_wins = 0;
                                if consecutive_losses > max_consecutive_losses {
                                    max_consecutive_losses = consecutive_losses;
                                }
                                if pnl < largest_loss {
                                    largest_loss = pnl;
                                }
                                if pnl_pct < largest_loss_pct {
                                    largest_loss_pct = pnl_pct;
                                }
                                
                                // 亏损持仓天数
                                if let Some(days_col) = &days_series {
                                    if let Ok(days_values) = days_col.i32() {
                                        if let Some(days) = days_values.get(i) {
                                            loss_holding_days += days as usize;
                                        }
                                    }
                                }
                            } else {
                                break_even_trades += 1;
                            }
                            
                            // 按股票统计交易次数和盈亏
                            if let Some(symbol_col) = &symbol_series {
                                if let Ok(symbol_values) = symbol_col.str() {
                                    if let Some(symbol) = symbol_values.get(i) {
                                        *stock_trades.entry(symbol.to_string()).or_insert(0) += 1;
                                        *stock_pnl.entry(symbol.to_string()).or_insert(0.0) += pnl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // 统计持仓天数
            if let Some(days_col) = days_series {
                if let Ok(days_values) = days_col.i32() {
                    for i in 0..days_values.len() {
                        if let Some(days) = days_values.get(i) {
                            total_holding_days += days as usize;
                        }
                    }
                }
            }
            
            // 统计交易额
            if let (Some(qty_col), Some(entry_col), Some(exit_col)) = 
                (quantity_series, entry_price_series, exit_price_series) {
                if let (Ok(qty_values), Ok(entry_values), Ok(exit_values)) = 
                    (qty_col.f64(), entry_col.f64(), exit_col.f64()) {
                    for i in 0..qty_values.len() {
                        if let (Some(qty), Some(entry), Some(exit)) = 
                            (qty_values.get(i), entry_values.get(i), exit_values.get(i)) {
                            total_buy_value += qty * entry;
                            total_sell_value += qty * exit;
                        }
                    }
                }
            }
            
            // 计算总手续费（更精确的估算）
            total_commission = (total_buy_value + total_sell_value) * self.commission_rate;
            if total_trades > 0 {
                total_commission = total_commission.max(self.min_commission * (total_trades * 2) as f64);
            }
        }
        
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        
        let avg_win = if winning_trades > 0 {
            total_win / winning_trades as f64
        } else {
            0.0
        };
        
        let avg_loss = if losing_trades > 0 {
            total_loss / losing_trades as f64
        } else {
            0.0
        };
        
        let profit_factor = if total_loss > 0.0 {
            total_win / total_loss
        } else if total_win > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        
        let avg_holding_period = if total_trades > 0 {
            total_holding_days as f64 / total_trades as f64
        } else {
            0.0
        };
        
        let avg_win_holding = if winning_trades > 0 {
            win_holding_days as f64 / winning_trades as f64
        } else {
            0.0
        };
        
        let avg_loss_holding = if losing_trades > 0 {
            loss_holding_days as f64 / losing_trades as f64
        } else {
            0.0
        };
        
        // 计算索提诺比率（Sortino Ratio）- 只考虑下行波动
        let downside_returns: Vec<f64> = daily_returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_std = if !downside_returns.is_empty() {
            let downside_variance = downside_returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            downside_variance.sqrt()
        } else {
            0.0
        };
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (252.0_f64).sqrt()
        } else {
            0.0
        };
        
        // 计算卡尔马比率（Calmar Ratio）- 年化收益 / 最大回撤
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };
        
        // 计算每日平均收益和波动率
        let daily_avg_return = mean_return * 100.0;
        let daily_volatility = std_return * 100.0;
        let annual_volatility = std_return * (252.0_f64).sqrt() * 100.0;
        
        // 计算正收益天数和负收益天数
        let positive_days = daily_returns.iter().filter(|&&r| r > 0.0).count();
        let negative_days = daily_returns.iter().filter(|&&r| r < 0.0).count();
        let daily_win_rate = if !daily_returns.is_empty() {
            positive_days as f64 / daily_returns.len() as f64 * 100.0
        } else {
            0.0
        };
        
        // 计算平均交易额
        let avg_trade_value = if total_trades > 0 {
            total_buy_value / total_trades as f64
        } else {
            0.0
        };
        
        // 计算资金使用率
        let capital_utilization = if self.initial_capital > 0.0 {
            avg_trade_value / self.initial_capital * 100.0
        } else {
            0.0
        };
        
        // 按股票统计，找出表现最好和最差的股票
        let mut stock_performance: Vec<(String, f64, usize, usize)> = stock_pnl.iter()
            .map(|(symbol, pnl)| {
                let trades = *stock_trades.get(symbol).unwrap_or(&0);
                let wins = *stock_wins.get(symbol).unwrap_or(&0);
                (symbol.clone(), *pnl, trades, wins)
            })
            .collect();
        stock_performance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let best_stock = stock_performance.first()
            .map(|(s, p, t, w)| format!("{} (盈亏: {:.2}, 交易{}次, 盈利{}次)", s, p, t, w))
            .unwrap_or_else(|| "无".to_string());
        let worst_stock = stock_performance.last()
            .map(|(s, p, t, w)| format!("{} (盈亏: {:.2}, 交易{}次, 盈利{}次)", s, p, t, w))
            .unwrap_or_else(|| "无".to_string());
        
        let num_stocks_traded = stock_trades.len();
        
        let summary_text = format!(
            r#"
================================================================================
                              回测统计摘要
================================================================================

【基本信息】
  回测期间: {} 天 ({:.2} 年)
  初始资金: {:.2}
  最终资金: {:.2}
  总盈亏: {:.2}

【收益指标】
  总收益率: {:.2}%
  年化收益率: {:.2}%
  日均收益率: {:.4}%
  
【风险指标】
  最大回撤: {:.2}%
  最大回撤持续: {} 天
  日波动率: {:.4}%
  年化波动率: {:.2}%
  
【风险调整收益】
  夏普比率: {:.4}
  索提诺比率: {:.4}
  卡尔马比率: {:.4}

【交易统计】
  总交易次数: {}
  盈利交易: {} ({:.2}%)
  亏损交易: {} ({:.2}%)
  盈亏平衡: {}
  
  胜率: {:.2}%
  盈亏比: {:.4}
  
【盈利分析】
  总盈利: {:.2}
  平均盈利: {:.2}
  最大单笔盈利: {:.2} ({:.2}%)
  平均盈利持仓: {:.1} 天
  
【亏损分析】
  总亏损: {:.2}
  平均亏损: {:.2}
  最大单笔亏损: {:.2} ({:.2}%)
  平均亏损持仓: {:.1} 天
  
【持仓分析】
  平均持仓周期: {:.1} 天
  总持仓天数: {} 天
  最大连续盈利: {} 次
  最大连续亏损: {} 次

【交易成本】
  总交易额(买入): {:.2}
  总交易额(卖出): {:.2}
  总手续费: {:.2}
  手续费占比: {:.4}%
  
【资金使用】
  平均单笔交易额: {:.2}
  资金使用率: {:.2}%

【日收益分析】
  日收益数据量: {} 天
  正收益天数: {} ({:.2}%)
  负收益天数: {} ({:.2}%)
  持平天数: {}

【股票维度】
  交易股票数量: {} 只
  表现最好: {}
  表现最差: {}

================================================================================
"#,
            n_days, years,
            initial,
            final_value,
            final_value - initial,
            
            total_return * 100.0,
            annual_return * 100.0,
            daily_avg_return,
            
            max_drawdown * 100.0,
            max_dd_duration,
            daily_volatility,
            annual_volatility,
            
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            
            total_trades,
            winning_trades, win_rate,
            losing_trades, if total_trades > 0 { losing_trades as f64 / total_trades as f64 * 100.0 } else { 0.0 },
            break_even_trades,
            
            win_rate * 100.0,
            profit_factor,
            
            total_win,
            avg_win,
            largest_win, largest_win_pct,
            avg_win_holding,
            
            total_loss,
            avg_loss,
            largest_loss, largest_loss_pct,
            avg_loss_holding,
            
            avg_holding_period,
            total_holding_days,
            max_consecutive_wins,
            max_consecutive_losses,
            
            total_buy_value,
            total_sell_value,
            total_commission,
            if final_value > initial { total_commission / (final_value - initial) * 100.0 } else { 0.0 },
            
            avg_trade_value,
            capital_utilization,
            
            daily_returns.len(),
            positive_days, daily_win_rate,
            negative_days, if !daily_returns.is_empty() { negative_days as f64 / daily_returns.len() as f64 * 100.0 } else { 0.0 },
            daily_returns.len() - positive_days - negative_days,
            
            num_stocks_traded,
            best_stock,
            worst_stock,
        );
        
        println!("{}", summary_text);
        Ok(())
    }
}

impl Backtest {
    /// 对单只股票进行回测
    fn backtest_single_stock(
        symbol: String,
        dates: &[String],
        prices: &[f64],
        buy_signals: &[bool],
        sell_signals: &[bool],
        initial_capital: f64,
        commission_rate: f64,
        min_commission: f64,
        slippage: f64,
    ) -> StockBacktestResult {
        let n_days = dates.len();
        let mut cash = initial_capital;
        let mut position: Option<Position> = None;
        let mut closed_positions: Vec<Position> = Vec::new();
        
        let mut daily_dates: Vec<String> = Vec::with_capacity(n_days);
        let mut daily_cash: Vec<f64> = Vec::with_capacity(n_days);
        let mut daily_stock_value: Vec<f64> = Vec::with_capacity(n_days);
        let mut daily_total_value: Vec<f64> = Vec::with_capacity(n_days);
        
        for day_idx in 0..n_days {
            let current_date = &dates[day_idx];
            let price = prices[day_idx];
            let buy_signal = buy_signals[day_idx];
            let sell_signal = sell_signals[day_idx];
            
            // 跳过无效价格
            if price.is_nan() || price <= 0.0 {
                daily_dates.push(current_date.clone());
                daily_cash.push(cash);
                daily_stock_value.push(0.0);
                daily_total_value.push(cash);
                continue;
            }
            
            // 处理卖出信号
            if sell_signal && position.is_some() {
                let mut pos = position.take().unwrap();
                
                // 应用滑点：卖出价格下降
                let sell_price = price * (1.0 - slippage);
                let sell_value = pos.quantity * sell_price;
                let sell_commission = (sell_value * commission_rate).max(min_commission);
                
                cash += sell_value - sell_commission;
                
                // 预先计算买入成本，避免重复计算
                let entry_value = pos.quantity * pos.entry_price;
                let buy_commission = (entry_value * commission_rate).max(min_commission);
                let pnl = sell_value - entry_value - buy_commission - sell_commission;
                
                pos.exit_date = Some(current_date.clone());
                pos.exit_price = Some(sell_price);
                pos.pnl = Some(pnl);
                pos.pnl_pct = Some(pnl / entry_value * 100.0); // 复用 entry_value
                pos.holding_days = Self::calculate_holding_days(&pos.entry_date, current_date);
                
                closed_positions.push(pos);
            }
            
            // 处理买入信号
            if buy_signal && position.is_none() {
                // 应用滑点：买入价格上浮
                let buy_price = price * (1.0 + slippage);
                
                // 计算可以买入的整百股数量（使用全部可用资金）
                let available_cash = cash;
                
                // 先估算可买数量（考虑佣金）
                let estimated_quantity = available_cash / (buy_price * (1.0 + commission_rate));
                let lots = (estimated_quantity / 100.0).floor() as i32; // 整百手数
                
                if lots > 0 {
                    let quantity = (lots * 100) as f64;
                    let buy_value = quantity * buy_price;
                    let commission = (buy_value * commission_rate).max(min_commission); // 应用最低佣金
                    let total_cost = buy_value + commission;
                    
                    // 确保资金充足
                    if total_cost <= cash {
                        cash -= total_cost;
                        
                        position = Some(Position {
                            symbol: symbol.clone(),
                            entry_date: current_date.clone(),
                            entry_price: buy_price, // 使用滑点后的价格
                            quantity,
                            exit_date: None,
                            exit_price: None,
                            pnl: None,
                            pnl_pct: None,
                            holding_days: None,
                        });
                    }
                }
            }
            
            // 计算当日持仓市值（优化：使用 map_or 避免分支）
            let stock_value = position.as_ref().map_or(0.0, |pos| pos.quantity * price);
            let total_value = cash + stock_value;
            
            daily_dates.push(current_date.clone());
            daily_cash.push(cash);
            daily_stock_value.push(stock_value);
            daily_total_value.push(total_value);
        }
        
        // 强制平仓最后持仓
        if let Some(mut pos) = position.take() {
            let last_day_idx = n_days - 1;
            let last_date = &dates[last_day_idx];
            let price = prices[last_day_idx];
            
            if !price.is_nan() && price > 0.0 {
                // 应用滑点：卖出价格下降
                let sell_price = price * (1.0 - slippage);
                let sell_value = pos.quantity * sell_price;
                let sell_commission = (sell_value * commission_rate).max(min_commission);
                
                cash += sell_value - sell_commission;
                
                // 预先计算买入成本，避免重复计算
                let entry_value = pos.quantity * pos.entry_price;
                let buy_commission = (entry_value * commission_rate).max(min_commission);
                let pnl = sell_value - entry_value - buy_commission - sell_commission;
                
                pos.exit_date = Some(last_date.clone());
                pos.exit_price = Some(sell_price);
                pos.pnl = Some(pnl);
                pos.pnl_pct = Some(pnl / entry_value * 100.0); // 复用 entry_value
                pos.holding_days = Self::calculate_holding_days(&pos.entry_date, last_date);
                
                closed_positions.push(pos);
                
                // 更新最后一天的资金（优化：合并更新）
                if let (Some(last_cash), Some(last_total)) = 
                    (daily_cash.last_mut(), daily_total_value.last_mut()) {
                    *last_cash = cash;
                    *last_total = cash;
                }
            }
        }
        
        StockBacktestResult {
            symbol,
            daily_dates,
            daily_cash,
            daily_stock_value,
            daily_total_value,
            positions: closed_positions,
        }
    }
    
    /// 计算两个日期字符串之间的天数差（优化版：早期退出）
    fn calculate_holding_days(entry_date: &str, exit_date: &str) -> Option<i32> {
        // 优化：尝试最常用的格式，成功后立即返回
        let formats = [
            "%Y-%m-%d",           // 2023-01-01 (最常用)
            "%Y/%m/%d",           // 2023/01/01
            "%Y%m%d",             // 20230101
            "%Y-%m-%d %H:%M:%S", // 2023-01-01 00:00:00
            "%d/%m/%Y",           // 01/01/2023
            "%m/%d/%Y",           // 01/01/2023
        ];
        
        // 优化：尝试解析入场日期，找到格式后直接解析出场日期
        for format in &formats {
            if let Ok(entry_d) = NaiveDate::parse_from_str(entry_date, format) {
                // 使用同一格式解析出场日期
                if let Ok(exit_d) = NaiveDate::parse_from_str(exit_date, format) {
                    return Some((exit_d - entry_d).num_days() as i32);
                }
            }
        }
        
        // 如果同格式解析失败，分别尝试所有格式
        let entry = formats.iter()
            .find_map(|fmt| NaiveDate::parse_from_str(entry_date, fmt).ok());
        
        let exit = formats.iter()
            .find_map(|fmt| NaiveDate::parse_from_str(exit_date, fmt).ok());
        
        match (entry, exit) {
            (Some(entry_d), Some(exit_d)) => Some((exit_d - entry_d).num_days() as i32),
            _ => None,
        }
    }
}
