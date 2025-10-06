#![allow(dead_code)]

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;
use std::fmt;
use wide::f64x4;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

// 高性能HashMap类型别名
type FastHashMap<K, V> = FxHashMap<K, V>;

// 常用常量
const FLOAT_EPSILON: f64 = 1e-10;


// ============================================================================
// 向量化计算模块
// ============================================================================

/// 批量乘法运算
#[inline(always)]
pub fn batch_multiply(prices: &[f64], quantities: &[f64], output: &mut [f64]) {
    assert_eq!(prices.len(), quantities.len());
    assert_eq!(prices.len(), output.len());
    
    let chunks = prices.len() / 4;
    let remainder = prices.len() % 4;
    
    // 使用向量化处理4个元素的批次
    for i in 0..chunks {
        let base = i * 4;
        
        let price_vec = f64x4::new([
            prices[base], prices[base + 1], 
            prices[base + 2], prices[base + 3]
        ]);
        
        let qty_vec = f64x4::new([
            quantities[base], quantities[base + 1],
            quantities[base + 2], quantities[base + 3] 
        ]);
        
        let result = price_vec * qty_vec;
        let result_array = result.to_array();
        
        output[base] = result_array[0];
        output[base + 1] = result_array[1];
        output[base + 2] = result_array[2];
        output[base + 3] = result_array[3];
    }
    
    // 处理剩余元素
    let base = chunks * 4;
    for i in 0..remainder {
        output[base + i] = prices[base + i] * quantities[base + i];
    }
}

/// 向量化求和
#[inline(always)]  
pub fn vector_sum(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let chunks = values.len() / 4;
    let remainder = values.len() % 4;
    
    let mut sum_vec = f64x4::splat(0.0);
    
    // 批量求和
    for i in 0..chunks {
        let base = i * 4;
        let vec = f64x4::new([
            values[base], values[base + 1],
            values[base + 2], values[base + 3]
        ]);
        sum_vec += vec;
    }
    
    // 水平求和
    let sum_array = sum_vec.to_array();
    let mut total = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // 处理剩余元素
    let base = chunks * 4;
    for i in 0..remainder {
        total += values[base + i];
    }
    
    total
}

// ============================================================================
// 辅助函数：更健壮的数据提取
// ============================================================================

#[inline]
fn series_to_bool_vec(s: &Column, fallback_len: usize) -> PolarsResult<Vec<bool>> {
    match s.dtype() {
        DataType::Boolean => {
            Ok(
                s.bool()?
                    .into_iter()
                    .map(|v| v.unwrap_or(false))
                    .collect(),
            )
        }
        _ => {
            // 尝试转换为布尔；失败则回退为全 false
            match s.cast(&DataType::Boolean) {
                Ok(casted) => Ok(
                    casted
                        .bool()?
                        .into_iter()
                        .map(|v| v.unwrap_or(false))
                        .collect(),
                ),
                Err(_) => Ok(vec![false; fallback_len]),
            }
        }
    }
}

#[inline]
fn backfill_f64<I: IntoIterator<Item = Option<f64>>>(iter: I) -> Vec<f64> {
    let vals: Vec<Option<f64>> = iter.into_iter().collect();
    let mut out = vec![0.0; vals.len()];
    let mut next_valid: Option<f64> = None;
    for i in (0..vals.len()).rev() {
        if let Some(v) = vals[i] {
            out[i] = v;
            next_valid = Some(v);
        } else if let Some(v) = next_valid {
            out[i] = v;
        } else {
            // 序列尾部全是缺失时，用0.0兜底
            out[i] = 0.0;
        }
    }
    out
}

// ============================================================================
// 并行计算模块
// ============================================================================

/// 并行处理股票数据的分块结果
struct ParallelChunkResult {
    symbol_orders: Vec<Order>,
    symbol_trades: Vec<Trade>,
    chunk_performance: f64,
}

/// 处理单个股票的回测逻辑
fn process_symbol_parallel(
    symbol: &str,
    price_data: &[f64],
    entry_data: &[bool], 
    exit_data: &[bool],
    initial_allocation: f64,
    fee_rate: f64,
) -> ParallelChunkResult {
    let start_time = std::time::Instant::now();
    
    let mut orders = Vec::new();
    let mut trades = Vec::new();
    let mut position: f64 = 0.0; // 当前持仓
    let mut avg_price: f64 = 0.0;
    
    for i in 0..price_data.len() {
        let current_price = price_data[i];
        let timestamp = i as i64;
        
        // 处理出场信号
        if i < exit_data.len() && exit_data[i] && position.abs() > FLOAT_EPSILON {
            let side = if position > 0.0 { Side::Sell } else { Side::Buy };
            let quantity = position.abs();
            
            let order = Order::new_market_order(
                format!("{}_{}", symbol, timestamp),
                symbol.to_string(),
                side,
                quantity,
                timestamp,
            );
            
            let trade = Trade {
                symbol: symbol.to_string(),
                side,
                quantity,
                price: current_price,
                timestamp,
                fees: current_price * quantity * fee_rate,
                pnl: if position > 0.0 {
                    (current_price - avg_price) * quantity
                } else {
                    (avg_price - current_price) * quantity
                },
            };
            
            orders.push(order);
            trades.push(trade);
            position = 0.0;
            avg_price = 0.0;
        }
        
        // 处理入场信号
        if i < entry_data.len() && entry_data[i] && position.abs() < FLOAT_EPSILON {
            let quantity = (initial_allocation / current_price).floor();
            if quantity >= 1.0 {
                let order = Order::new_market_order(
                    format!("{}_{}", symbol, timestamp),
                    symbol.to_string(),
                    Side::Buy,
                    quantity,
                    timestamp,
                );
                
                let trade = Trade {
                    symbol: symbol.to_string(),
                    side: Side::Buy,
                    quantity,
                    price: current_price,
                    timestamp,
                    fees: current_price * quantity * fee_rate,
                    pnl: 0.0, // 开仓时盈亏为0
                };
                
                orders.push(order);
                trades.push(trade);
                position = quantity;
                avg_price = current_price;
            }
        }
    }
    
    let duration = start_time.elapsed().as_nanos() as f64 / 1_000_000.0; // ms
    
    ParallelChunkResult {
        symbol_orders: orders,
        symbol_trades: trades,
        chunk_performance: duration,
    }
}

/// 并行回测处理器
pub struct ParallelBacktestProcessor {
    chunk_size: usize,
    thread_pool_size: usize,
}

impl ParallelBacktestProcessor {
    pub fn new(chunk_size: usize, thread_pool_size: Option<usize>) -> Self {
        // 注意：不设置全局线程池，避免多次调用时panic
        // Rayon会自动使用合理的默认线程池
        // 如果需要自定义线程数，应在程序启动时一次性设置
        
        Self {
            chunk_size,
            thread_pool_size: thread_pool_size.unwrap_or_else(num_cpus::get),
        }
    }
    
    /// 并行处理多个股票的回测
    pub fn parallel_backtest(
        &self,
        symbols: &[String],
        price_data: &FastHashMap<String, Vec<f64>>,
        entry_data: &FastHashMap<String, Vec<bool>>,
        exit_data: &FastHashMap<String, Vec<bool>>,
        initial_cash: f64,
        fee_rate: f64,
    ) -> (Vec<Order>, Vec<Trade>, f64) {
        let per_symbol_allocation = initial_cash / symbols.len() as f64;
        
        // 并行处理每个股票
        let results: Vec<ParallelChunkResult> = symbols.par_iter()
            .map(|symbol| {
                let prices = price_data.get(symbol).map(|v| v.as_slice()).unwrap_or(&[]);
                let entries = entry_data.get(symbol).map(|v| v.as_slice()).unwrap_or(&[]);
                let exits = exit_data.get(symbol).map(|v| v.as_slice()).unwrap_or(&[]);
                
                process_symbol_parallel(
                    symbol,
                    prices,
                    entries, 
                    exits,
                    per_symbol_allocation,
                    fee_rate,
                )
            })
            .collect();
        
        // 合并结果
        let mut all_orders = Vec::new();
        let mut all_trades = Vec::new();
        let mut total_duration = 0.0;
        
        for result in results {
            all_orders.extend(result.symbol_orders);
            all_trades.extend(result.symbol_trades);
            total_duration += result.chunk_performance;
        }
        
        (all_orders, all_trades, total_duration)
    }
}

// ============================================================================
// 核心数据结构定义
// ============================================================================

/// 订单方向
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// 订单状态
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
}

/// 订单类型
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum OrderType {
    Market,
    Limit,
}

/// 订单结构体
#[derive(Debug, Clone)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>, // 限价单使用，市价单为None
    pub status: OrderStatus,
    pub timestamp: i64,
    pub fill_price: Option<f64>,
    pub fill_quantity: f64,
}

impl Order {
    pub fn new_market_order(
        id: String,
        symbol: String,
        side: Side,
        quantity: f64,
        timestamp: i64,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            status: OrderStatus::Pending,
            timestamp,
            fill_price: None,
            fill_quantity: 0.0,
        }
    }

    pub fn fill(&mut self, price: f64, quantity: f64) {
        self.fill_price = Some(price);
        self.fill_quantity += quantity;
        if self.fill_quantity >= self.quantity {
            self.status = OrderStatus::Filled;
        }
    }
}

/// 持仓结构体
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    // 频繁访问的字段
    pub quantity: f64,     // 正数为多头，负数为空头
    pub avg_price: f64,    // 平均成本价
    pub total_cost: f64,   // 总成本
    pub unrealized_pnl: f64, // 浮动盈亏
    pub realized_pnl: f64,   // 已实现盈亏
}

impl Position {
    #[inline]
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            quantity: 0.0,
            avg_price: 0.0,
            total_cost: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
        }
    }

    /// 更新持仓
    #[inline]
    pub fn update(&mut self, side: Side, quantity: f64, price: f64, fees: f64) {
        let signed_quantity = match side {
            Side::Buy => quantity,
            Side::Sell => -quantity,
        };

        let is_zero_position = self.quantity.abs() < f64::EPSILON;
        let same_direction = (self.quantity > 0.0) == (signed_quantity > 0.0);

        if is_zero_position {
            // 开新仓
            self.quantity = signed_quantity;
            self.avg_price = price;
            self.total_cost = quantity * price + fees;
        } else if same_direction && !is_zero_position {
            // 加仓
            let old_value = self.quantity * self.avg_price;
            let new_value = signed_quantity * price;
            let new_quantity = self.quantity + signed_quantity;
            
            self.avg_price = (old_value + new_value) / new_quantity;
            self.quantity = new_quantity;
            self.total_cost += quantity * price + fees;
        } else {
            // 减仓或平仓
            let close_quantity = signed_quantity.abs().min(self.quantity.abs());
            let remaining_quantity = self.quantity + signed_quantity;
            
            let realized = if self.quantity > 0.0 {
                close_quantity * (price - self.avg_price) - fees
            } else {
                close_quantity * (self.avg_price - price) - fees
            };
            
            self.realized_pnl += realized;
            self.quantity = remaining_quantity;
            
            if remaining_quantity.abs() < f64::EPSILON {
                self.quantity = 0.0;
                self.avg_price = 0.0;
                self.total_cost = 0.0;
            }
        }
    }

    #[inline]
    /// 计算持仓市值（需要当前价格）
    pub fn position_value(&self, current_price: f64) -> f64 {
        if self.quantity != 0.0 {
            self.quantity.abs() * current_price
        } else {
            0.0
        }
    }

    pub fn update_unrealized_pnl(&mut self, current_price: f64) {
        if self.quantity != 0.0 {
            if self.quantity > 0.0 {
                self.unrealized_pnl = self.quantity * (current_price - self.avg_price);
            } else {
                self.unrealized_pnl = self.quantity.abs() * (self.avg_price - current_price);
            }
        } else {
            self.unrealized_pnl = 0.0;
        }
    }

    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl
    }
}

/// 交易记录
#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: i64,
    pub fees: f64,
    pub pnl: f64, // 此次交易的盈亏（仅平仓时有意义）
}

/// 每日资金记录
#[derive(Debug, Clone)]
pub struct DailyRecord {
    pub date: String,
    pub cash: f64,
    pub equity: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub daily_pnl: f64, // 当日盈亏
}

/// 投资组合账户
#[derive(Debug)]
pub struct Portfolio {
    pub cash: f64,
    pub initial_cash: f64,
    pub positions: HashMap<String, Position>,
    pub trades: Vec<Trade>,
    pub orders: Vec<Order>,
    pub fee_rate: f64,
}

impl Portfolio {
    pub fn new(initial_cash: f64, fee_rate: f64) -> Self {
        Self {
            cash: initial_cash,
            initial_cash,
            positions: HashMap::new(),
            trades: Vec::new(),
            orders: Vec::new(),
            fee_rate,
        }
    }

    /// 执行订单
    pub fn execute_order(&mut self, mut order: Order, market_price: f64) -> Result<(), String> {
        let execution_price = match order.order_type {
            OrderType::Market => market_price,
            OrderType::Limit => {
                match order.price {
                    Some(limit_price) => {
                        // 简化的限价单执行逻辑
                        match order.side {
                            Side::Buy => {
                                if market_price <= limit_price {
                                    market_price
                                } else {
                                    return Ok(()); // 限价单未成交
                                }
                            }
                            Side::Sell => {
                                if market_price >= limit_price {
                                    market_price
                                } else {
                                    return Ok(()); // 限价单未成交
                                }
                            }
                        }
                    }
                    None => return Err("Limit order must have price".to_string()),
                }
            }
        };

        let fees = order.quantity * execution_price * self.fee_rate;
        let total_cost = order.quantity * execution_price + fees;

        // 检查资金是否足够
        match order.side {
            Side::Buy => {
                if self.cash < total_cost {
                    return Err("Insufficient funds".to_string());
                }
                self.cash -= total_cost;
            }
            Side::Sell => {
                // 对于卖出，需要检查是否有足够的持仓
                let position = self.positions.get(&order.symbol);
                if let Some(pos) = position {
                    if pos.quantity < order.quantity {
                        return Err("Insufficient position".to_string());
                    }
                } else {
                    return Err("No position to sell".to_string());
                }
                self.cash += order.quantity * execution_price - fees;
            }
        }

        // 更新持仓
        let position = self.positions
            .entry(order.symbol.clone())
            .or_insert_with(|| Position::new(order.symbol.clone()));

        let old_quantity = position.quantity;
        let old_avg_price = position.avg_price; // 保存旧的平均价格用于PNL计算
        
        position.update(order.side, order.quantity, execution_price, fees);

        // 记录交易 - 修复PNL计算
        let pnl = if old_quantity != 0.0 {
            match order.side {
                Side::Buy if old_quantity < 0.0 => {
                    // 买入平空仓
                    let close_qty = order.quantity.min(old_quantity.abs());
                    close_qty * (old_avg_price - execution_price) // 使用旧的平均价格
                }
                Side::Sell if old_quantity > 0.0 => {
                    // 卖出平多仓  
                    let close_qty = order.quantity.min(old_quantity);
                    close_qty * (execution_price - old_avg_price) // 使用旧的平均价格
                }
                _ => 0.0,
            }
        } else {
            0.0
        };

        let trade = Trade {
            symbol: order.symbol.clone(),
            side: order.side,
            quantity: order.quantity,
            price: execution_price,
            timestamp: order.timestamp,
            fees,
            pnl,
        };

        self.trades.push(trade);
        order.fill(execution_price, order.quantity);
        self.orders.push(order);

        Ok(())
    }

    /// 更新所有持仓的浮动盈亏
    pub fn update_positions(&mut self, prices: &HashMap<String, f64>) {
        // 小规模数据使用简单更新
        if self.positions.len() < 8 {
            for (symbol, position) in &mut self.positions {
                if let Some(&current_price) = prices.get(symbol) {
                    position.update_unrealized_pnl(current_price);
                }
            }
            return;
        }
        
        // 大规模数据使用批量更新
        let symbols: Vec<String> = self.positions.keys().cloned().collect();
        let mut current_prices = Vec::with_capacity(symbols.len());
        let mut quantities = Vec::with_capacity(symbols.len());
        let mut avg_prices = Vec::with_capacity(symbols.len());
        
        // 收集数据进行批量计算
        for symbol in &symbols {
            let price = prices.get(symbol).copied().unwrap_or(0.0);
            current_prices.push(price);
            
            if let Some(position) = self.positions.get(symbol) {
                quantities.push(position.quantity);
                avg_prices.push(position.avg_price);
            } else {
                quantities.push(0.0);
                avg_prices.push(0.0);
            }
        }
        
        // 批量计算价差和未实现盈亏
        let mut price_diffs = vec![0.0; current_prices.len()];
        for i in 0..current_prices.len() {
            price_diffs[i] = current_prices[i] - avg_prices[i];
        }
        
        let mut unrealized_values = vec![0.0; price_diffs.len()];
        if !price_diffs.is_empty() {
            batch_multiply(&price_diffs, &quantities, &mut unrealized_values);
        }
        
        // 更新持仓数据
        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(position) = self.positions.get_mut(symbol) {
                if current_prices[i] > FLOAT_EPSILON {
                    position.unrealized_pnl = unrealized_values[i];
                }
            }
        }
    }

    /// 计算总权益（使用当前价格）
    pub fn total_equity_with_prices(&self, current_prices: &HashMap<String, f64>) -> f64 {
        let mut total_position_value = 0.0;
        for (symbol, position) in &self.positions {
            if let Some(&current_price) = current_prices.get(symbol) {
                total_position_value += position.quantity.abs() * current_price;
            }
        }
        self.cash + total_position_value
    }

    /// 计算总权益
    pub fn total_equity(&self) -> f64 {
        if self.positions.is_empty() {
            return self.cash;
        }
        
        // 小规模数据使用传统方法
        if self.positions.len() < 8 {
            let unrealized_pnl: f64 = self.positions.values()
                .map(|p| p.unrealized_pnl)
                .sum();
            return self.cash + unrealized_pnl;
        }
        
        // 大规模数据使用向量化求和
        let unrealized_values: Vec<f64> = self.positions.values()
            .map(|p| p.unrealized_pnl)
            .collect();
            
        let total_unrealized = vector_sum(&unrealized_values);
        self.cash + total_unrealized
    }

    /// 计算已实现盈亏
    pub fn realized_pnl(&self) -> f64 {
        self.positions.values()
            .map(|p| p.realized_pnl)
            .sum()
    }
}

// ============================================================================
// 回测引擎
// ============================================================================

/// 回测引擎
pub struct BacktestEngine {
    pub portfolio: Portfolio,
    pub data: DataFrame,
    pub entry_signals: DataFrame,
    pub exit_signals: DataFrame,
    pub results: Vec<BacktestRecord>,
    pub daily_records: Vec<DailyRecord>,
    pub start_time: Option<std::time::Instant>,
    pub end_time: Option<std::time::Instant>,
    pub backtest_duration_ms: u64,
}

/// 回测记录
#[derive(Debug, Clone)]
pub struct BacktestRecord {
    pub timestamp: i64,
    pub equity: f64,
    pub cash: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

impl BacktestEngine {
    pub fn new(
        data: DataFrame,
        entry_signals: DataFrame,
        exit_signals: DataFrame,
        initial_cash: f64,
        fee_rate: f64,
    ) -> PolarsResult<Self> {
        let portfolio = Portfolio::new(initial_cash, fee_rate);
        
        Ok(Self {
            portfolio,
            data,
            entry_signals,
            exit_signals,
            results: Vec::new(),
            daily_records: Vec::new(),
            start_time: None,
            end_time: None,
            backtest_duration_ms: 0,
        })
    }

    /// 回测运行
    pub fn run(&mut self) -> PolarsResult<()> {
        self.run_with_parallelism(None)
    }
    
    /// 并行回测运行
    pub fn run_with_parallelism(&mut self, enable_parallel: Option<bool>) -> PolarsResult<()> {
        self.start_time = Some(std::time::Instant::now());
        
        let data_height = self.data.height();
        let price_columns: Vec<String> = self
            .data
            .get_column_names()
            .iter()
            .filter(|&col| col.as_str() != "Date" && col.as_str() != "date")
            .map(|col| col.to_string())
            .collect();

        // 使用更快的HashMap
        let mut price_data: FastHashMap<String, Vec<f64>> = FastHashMap::default();
        price_data.reserve(price_columns.len());
        
        let mut entry_data: FastHashMap<String, Vec<bool>> = FastHashMap::default(); 
        entry_data.reserve(price_columns.len());
        
        let mut exit_data: FastHashMap<String, Vec<bool>> = FastHashMap::default();
        exit_data.reserve(price_columns.len());
        
        // 批量提取数据（价格：向后插入法 backfill；尾部仍缺失用0.0兜底）
        for symbol in &price_columns {
            let col = self.data.column(symbol)?.f64()?;
            let prices = backfill_f64(col.into_iter());
            price_data.insert(symbol.clone(), prices);
        }
        
        // 批量提取信号数据（null -> false；可尝试类型转换）
        for symbol in &price_columns {
            if let Ok(entry_col) = self.entry_signals.column(symbol) {
                let entries = series_to_bool_vec(entry_col, data_height)?;
                entry_data.insert(symbol.clone(), entries);
            } else {
                entry_data.insert(symbol.clone(), vec![false; data_height]);
            }
        }
        
        for symbol in &price_columns {
            if let Ok(exit_col) = self.exit_signals.column(symbol) {
                let exits = series_to_bool_vec(exit_col, data_height)?;
                exit_data.insert(symbol.clone(), exits);
            } else {
                exit_data.insert(symbol.clone(), vec![false; data_height]);
            }
        }

        // 获取日期数据（兼容 Date/date；统一转 String 再读取）
        let date_series = if self
            .data
            .get_column_names()
            .iter()
            .any(|c| c.as_str() == "Date")
        {
            self.data.column("Date")?
        } else if self
            .data
            .get_column_names()
            .iter()
            .any(|c| c.as_str() == "date")
        {
            self.data.column("date")?
        } else {
            return Err(PolarsError::ComputeError("Date/date column not found".into()));
        };
        let date_utf8 = if date_series.dtype() != &DataType::String {
            date_series.cast(&DataType::String)?
        } else {
            date_series.clone()
        };
        let date_strings = date_utf8
            .str()?
            .into_no_null_iter()
            .collect::<Vec<_>>();

        // 判断是否使用并行处理
        let use_parallel = enable_parallel.unwrap_or(price_columns.len() >= 10);
        
        if use_parallel && price_columns.len() > 1 {
            // 并行处理版本
            let processor = ParallelBacktestProcessor::new(4, Some(num_cpus::get().min(8)));
            let (parallel_orders, parallel_trades, _parallel_duration) = processor.parallel_backtest(
                &price_columns,
                &price_data,
                &entry_data,
                &exit_data,
                self.portfolio.initial_cash,
                self.portfolio.fee_rate,
            );
            
            // 将并行结果合并到组合中
            self.portfolio.orders.extend(parallel_orders);
            self.portfolio.trades.extend(parallel_trades);
            
            // 重新构建portfolio状态：基于trades重建正确的cash和positions
            self.portfolio.cash = self.portfolio.initial_cash;
            self.portfolio.positions.clear();
            
            // 按时间排序trades并重新执行以重建portfolio状态
            self.portfolio.trades.sort_by_key(|t| t.timestamp);
            
            for trade in &self.portfolio.trades {
                // 重新执行trade以更新portfolio状态
                match trade.side {
                    Side::Buy => {
                        self.portfolio.cash -= trade.quantity * trade.price + trade.fees;
                    }
                    Side::Sell => {
                        self.portfolio.cash += trade.quantity * trade.price - trade.fees;
                    }
                }
                
                // 更新position
                let position = self.portfolio.positions
                    .entry(trade.symbol.clone())
                    .or_insert_with(|| Position::new(trade.symbol.clone()));
                position.update(trade.side, trade.quantity, trade.price, trade.fees);
            }
            
            // 重新计算每日记录（正确版本）
            let mut prev_equity = self.portfolio.initial_cash;
            for (i, date) in date_strings.iter().enumerate() {
                // 构建当前价格映射用于更新持仓
                let mut current_prices: HashMap<String, f64> = HashMap::new();
                for symbol in &price_columns {
                    if let Some(prices) = price_data.get(symbol) {
                        if i < prices.len() {
                            current_prices.insert(symbol.clone(), prices[i]);
                        }
                    }
                }
                
                // 更新持仓市值
                self.portfolio.update_positions(&current_prices);
                let current_equity = self.portfolio.total_equity_with_prices(&current_prices);
                let daily_pnl = current_equity - prev_equity;
                let realized_pnl = self.portfolio.realized_pnl();
                let unrealized_pnl: f64 = self.portfolio.positions.values()
                    .map(|p| p.unrealized_pnl)
                    .sum();
                
                self.daily_records.push(DailyRecord {
                    date: date.to_string(),
                    cash: self.portfolio.cash,
                    equity: current_equity,
                    unrealized_pnl,
                    realized_pnl,
                    daily_pnl,
                });
                
                self.results.push(BacktestRecord {
                    timestamp: i as i64,
                    equity: current_equity,
                    cash: self.portfolio.cash,
                    unrealized_pnl,
                    realized_pnl,
                });
                
                prev_equity = current_equity;
            }
        } else {
            // 单线程版本
            let data_height = price_data.values().next().map_or(0, |v| v.len());
            let mut order_id_counter = 0u64;
            let mut prev_equity = self.portfolio.initial_cash;

            // 主回测循环
            for i in 0..data_height {
                let timestamp = i as i64;
                
                // 构建当前价格映射
                let mut current_prices: FastHashMap<String, f64> = FastHashMap::default();
                current_prices.reserve(price_columns.len());
                
                for symbol in &price_columns {
                    if let Some(prices) = price_data.get(symbol) {
                        if i < prices.len() {
                            current_prices.insert(symbol.clone(), prices[i]);
                        }
                    }
                }

                // 处理退出信号
                for symbol in &price_columns {
                    if let Some(exits) = exit_data.get(symbol) {
                        if i < exits.len() && exits[i] {
                            if let Some(position) = self.portfolio.positions.get(symbol) {
                                if position.quantity.abs() > FLOAT_EPSILON {
                                    let side = if position.quantity > 0.0 { Side::Sell } else { Side::Buy };
                                    let quantity = position.quantity.abs();
                                    
                                    order_id_counter += 1;
                                    let order = Order::new_market_order(
                                        format!("order_{}", order_id_counter),
                                        symbol.clone(),
                                        side,
                                        quantity,
                                        timestamp,
                                    );

                                    if let Some(&market_price) = current_prices.get(symbol) {
                                        let _ = self.portfolio.execute_order(order, market_price);
                                    }
                                }
                            }
                        }
                    }
                }

                // 处理入场信号
                let available_cash = self.portfolio.cash * 0.95;
                let num_active_signals = price_columns.iter()
                    .filter(|symbol| {
                        entry_data.get(*symbol)
                            .map_or(false, |entries| i < entries.len() && entries[i])
                    })
                    .count();
                
                if num_active_signals > 0 {
                    let allocation_per_signal = available_cash / num_active_signals as f64;
                    
                    for symbol in &price_columns {
                        if let Some(entries) = entry_data.get(symbol) {
                            if i < entries.len() && entries[i] {
                                if let Some(&market_price) = current_prices.get(symbol) {
                                    if market_price > FLOAT_EPSILON {
                                        let quantity = (allocation_per_signal / market_price).floor();
                                        
                                        if quantity >= 1.0 {
                                            order_id_counter += 1;
                                            let order = Order::new_market_order(
                                                format!("order_{}", order_id_counter),
                                                symbol.clone(),
                                                Side::Buy,
                                                quantity,
                                                timestamp,
                                            );

                                            let _ = self.portfolio.execute_order(order, market_price);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // 更新持仓
                let std_prices: HashMap<String, f64> = current_prices.iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect();
                self.portfolio.update_positions(&std_prices);

                // 记录每日数据
                if i < date_strings.len() {
                    let current_equity = self.portfolio.total_equity_with_prices(&std_prices);
                    let daily_pnl = current_equity - prev_equity;
                    let realized_pnl = self.portfolio.realized_pnl();
                    let unrealized_pnl: f64 = self.portfolio.positions.values()
                        .map(|p| p.unrealized_pnl)
                        .sum();

                    self.daily_records.push(DailyRecord {
                        date: date_strings[i].to_string(),
                        cash: self.portfolio.cash,
                        equity: current_equity,
                        unrealized_pnl,
                        realized_pnl,
                        daily_pnl,
                    });

                    self.results.push(BacktestRecord {
                        timestamp,
                        equity: current_equity,
                        cash: self.portfolio.cash,
                        unrealized_pnl,
                        realized_pnl,
                    });

                    prev_equity = current_equity;
                }
            }
        }

        // 记录性能统计
        self.end_time = Some(std::time::Instant::now());
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            self.backtest_duration_ms = end.duration_since(start).as_millis() as u64;
        }

        Ok(())
    }

    /// 获取回测结果统计摘要
    pub fn get_summary(&self) -> BacktestSummary {
        if self.results.is_empty() {
            eprintln!("警告: results为空，返回默认摘要");
            return BacktestSummary::default();
        }

        let initial_equity = self.portfolio.initial_cash;  // 使用真正的初始资金
        
        // 使用trades的总PNL来计算正确的总收益
        let total_trade_pnl: f64 = self.portfolio.trades.iter().map(|t| t.pnl).sum();
        let total_fees: f64 = self.portfolio.trades.iter().map(|t| t.fees).sum();
        let net_pnl = total_trade_pnl - total_fees;
        
        let corrected_final_equity = initial_equity + net_pnl;
        let total_return = corrected_final_equity - initial_equity;
        let total_return_pct = if initial_equity > 0.0 {
            (total_return / initial_equity) * 100.0
        } else {
            0.0
        };

        // 计算最大回撤
        let mut peak = initial_equity;
        let mut max_drawdown = 0.0;
        let mut max_drawdown_amount = 0.0;
        
        for record in &self.results {
            // 更新峰值
            if record.equity > peak {
                peak = record.equity;
            }
            
            // 计算回撤
            if peak > 0.0 {
                let drawdown = (peak - record.equity) / peak;
                let normalized_drawdown = drawdown.max(0.0).min(1.0);
                let drawdown_amt = peak - record.equity;
                
                if normalized_drawdown > max_drawdown {
                    max_drawdown = normalized_drawdown;
                    max_drawdown_amount = drawdown_amt;
                }
            }
        }

        let num_trades = self.portfolio.trades.len();
        let winning_trades = self.portfolio.trades.iter()
            .filter(|trade| trade.pnl > 0.0)
            .count();
        let win_rate = if num_trades > 0 {
            (winning_trades as f64 / num_trades as f64) * 100.0
        } else {
            0.0
        };

        let total_fees: f64 = self.portfolio.trades.iter()
            .map(|trade| trade.fees)
            .sum();

        BacktestSummary {
            initial_equity,
            final_equity: corrected_final_equity,
            total_return,
            total_return_pct,
            max_drawdown: max_drawdown * 100.0,
            max_drawdown_amount,
            total_trades: num_trades,
            win_rate,
            total_fees,
            realized_pnl: net_pnl,
        }
    }
}

/// 回测摘要报告
#[derive(Debug, Default)]
#[pyclass]
pub struct BacktestSummary {
    #[pyo3(get)]
    pub initial_equity: f64,
    #[pyo3(get)]
    pub final_equity: f64,
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub total_return_pct: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub max_drawdown_amount: f64,
    #[pyo3(get)]
    pub total_trades: usize,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub total_fees: f64,
    #[pyo3(get)]
    pub realized_pnl: f64,
}

impl fmt::Display for BacktestSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== 回测报告 ===")?;
        writeln!(f, "初始资金: {:.2}", self.initial_equity)?;
        writeln!(f, "最终权益: {:.2}", self.final_equity)?;
        writeln!(f, "总收益率: {:.2}%", self.total_return_pct)?;
        writeln!(f, "最大回撤: {:.2}%", self.max_drawdown)?;
        writeln!(f, "交易次数: {}", self.total_trades)?;
        writeln!(f, "胜率: {:.2}%", self.win_rate)?;
        writeln!(f, "总手续费: {:.2}", self.total_fees)?;
        writeln!(f, "已实现盈亏: {:.2}", self.realized_pnl)?;
        Ok(())
    }
}

// ============================================================================
// Python接口
// ============================================================================

/// Python回测类
#[pyclass]
pub struct Backtrade {
    summary: BacktestSummary,
    equity_curve: Vec<f64>,
    trades: Vec<Trade>,
    daily_records: Vec<DailyRecord>,
    backtest_duration_ms: f64,
}

#[pymethods]
impl Backtrade {
    /// 运行回测
    #[staticmethod]
    #[pyo3(signature = (data, entries, exits, init_cash = 100000.0, fee = 0.001))]
    pub fn run(
        data: PyDataFrame,
        entries: PyDataFrame,
        exits: PyDataFrame,
        init_cash: f64,
        fee: f64,
    ) -> PyResult<Self> {
        // 将PyDataFrame转换为DataFrame
        let data_df: DataFrame = data.into();
        let entries_df: DataFrame = entries.into();
        let exits_df: DataFrame = exits.into();

        // 创建回测引擎
        let mut engine = BacktestEngine::new(
            data_df,
            entries_df,
            exits_df,
            init_cash,
            fee,
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // 运行回测
        engine.run().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // 生成摘要
        let summary = engine.get_summary();
        let equity_curve: Vec<f64> = engine.results.iter().map(|r| r.equity).collect();
        
        // 使用BacktestEngine的实际耗时
        let duration_ms = engine.backtest_duration_ms as f64;

        Ok(Self {
            summary,
            equity_curve,
            trades: engine.portfolio.trades,
            daily_records: engine.daily_records,
            backtest_duration_ms: duration_ms,
        })
    }

    /// 显示回测摘要
    pub fn summary(&self) {
        println!("=== 回测性能 ===");
        println!("回测耗时: {:.2}ms", self.backtest_duration_ms);
        let data_points = self.equity_curve.len();
        println!("数据点数: {}", data_points);
        
        if self.backtest_duration_ms > 0.001 { // 如果耗时大于0.001ms
            let bars_per_second = (data_points as f64) / (self.backtest_duration_ms / 1000.0);
            if bars_per_second >= 1000000.0 {
                println!("处理速度: {:.1}M bars/s", bars_per_second / 1000000.0);
            } else if bars_per_second >= 1000.0 {
                println!("处理速度: {:.1}K bars/s", bars_per_second / 1000.0);
            } else {
                println!("处理速度: {:.0} bars/s", bars_per_second);
            }
        } else {
            println!("处理速度: 极快 (>1M bars/s)");
        }
        println!();
        println!("{}", self.summary);
    }

    /// 获取交易记录DataFrame
    pub fn trades_df(&self) -> PyResult<PyDataFrame> {
        let symbols: Vec<String> = self.trades.iter().map(|t| t.symbol.clone()).collect();
        let sides: Vec<String> = self.trades.iter().map(|t| match t.side {
            Side::Buy => "BUY".to_string(),
            Side::Sell => "SELL".to_string(),
        }).collect();
        let quantities: Vec<f64> = self.trades.iter().map(|t| t.quantity).collect();
        let prices: Vec<f64> = self.trades.iter().map(|t| t.price).collect();
        let timestamps: Vec<i64> = self.trades.iter().map(|t| t.timestamp).collect();
        let fees: Vec<f64> = self.trades.iter().map(|t| t.fees).collect();
        let pnls: Vec<f64> = self.trades.iter().map(|t| t.pnl).collect();

        let df = DataFrame::new(vec![
            Column::new("symbol".into(), symbols),
            Column::new("side".into(), sides),
            Column::new("quantity".into(), quantities),
            Column::new("price".into(), prices),
            Column::new("timestamp".into(), timestamps),
            Column::new("fees".into(), fees),
            Column::new("pnl".into(), pnls),
        ]).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyDataFrame(df))
    }

    /// 获取每日资金记录DataFrame
    pub fn daily_records_df(&self) -> PyResult<PyDataFrame> {
        let dates: Vec<String> = self.daily_records.iter().map(|r| r.date.clone()).collect();
        let cash: Vec<f64> = self.daily_records.iter().map(|r| r.cash).collect();
        let equity: Vec<f64> = self.daily_records.iter().map(|r| r.equity).collect();
        let unrealized_pnl: Vec<f64> = self.daily_records.iter().map(|r| r.unrealized_pnl).collect();
        let realized_pnl: Vec<f64> = self.daily_records.iter().map(|r| r.realized_pnl).collect();
        let daily_pnl: Vec<f64> = self.daily_records.iter().map(|r| r.daily_pnl).collect();

        let df = DataFrame::new(vec![
            Column::new("date".into(), dates),
            Column::new("cash".into(), cash),
            Column::new("equity".into(), equity),
            Column::new("unrealized_pnl".into(), unrealized_pnl),
            Column::new("realized_pnl".into(), realized_pnl),
            Column::new("daily_pnl".into(), daily_pnl),
        ]).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyDataFrame(df))
    }

        /// 获取回测耗时（毫秒）
    pub fn backtest_duration(&self) -> f64 {
        self.backtest_duration_ms
    }
}
