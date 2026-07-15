use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

/// Represents a trading order submitted by the strategy.
#[derive(Debug, Clone)]
pub struct Order {
    /// Target asset identifier.
    pub target_asset: String,
    /// Target quantity.
    pub target_quantity: f64,
    /// Price at which the order should execute.
    pub execution_price: f64,
}

/// Represents the execution receipt of an order from the engine.
#[derive(Debug, Clone)]
pub struct FillReceipt {
    /// Asset identifier.
    pub asset: String,
    /// Executed quantity.
    pub quantity: f64,
    /// Executed price.
    pub fill_price: f64,
    /// Commission fee paid.
    pub commission_paid: f64,
    /// Total transaction cost or revenue.
    pub transaction_cost: f64,
    /// Indicates if it was a buy order.
    pub is_buy: bool,
}

/// Execution engine responsible for simulating exchange matching.
pub struct ExecutionEngine {
    /// Buy slippage.
    buy_slippage: f64,
    /// Sell slippage.
    sell_slippage: f64,
    /// Buy commission rate.
    buy_commission_rate: f64,
    /// Sell commission rate.
    sell_commission_rate: f64,
    /// Minimum commission fee.
    minimum_commission_fee: f64,
}

impl ExecutionEngine {
    /// Simulates the execution of a requested order and generates a fill receipt.
    pub fn process_order(
        &self,
        order: &Order,
        available_cash: f64,
        current_position: f64,
    ) -> Option<FillReceipt> {
        if order.target_quantity > 0.0 {
            let fill_price = order.execution_price + self.buy_slippage;
            let transaction_cost = order.target_quantity * fill_price;
            let commission_paid =
                (transaction_cost * self.buy_commission_rate).max(self.minimum_commission_fee);

            if available_cash >= transaction_cost + commission_paid {
                Some(FillReceipt {
                    asset: order.target_asset.clone(),
                    quantity: order.target_quantity,
                    fill_price,
                    commission_paid,
                    transaction_cost,
                    is_buy: true,
                })
            } else {
                None
            }
        } else if order.target_quantity < 0.0 {
            let absolute_quantity = order.target_quantity.abs();
            if current_position >= absolute_quantity {
                let fill_price = order.execution_price - self.sell_slippage;
                let transaction_revenue = absolute_quantity * fill_price;
                let commission_paid = (transaction_revenue * self.sell_commission_rate)
                    .max(self.minimum_commission_fee);

                Some(FillReceipt {
                    asset: order.target_asset.clone(),
                    quantity: order.target_quantity,
                    fill_price,
                    commission_paid,
                    transaction_cost: -transaction_revenue,
                    is_buy: false,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Portfolio manager for tracking balances and holdings.
pub struct Portfolio {
    /// Available cash balance.
    pub available_cash: f64,
    /// Active holdings quantities.
    pub positions: HashMap<String, f64>,
    /// Average entry prices for active holdings.
    pub average_entry_prices: HashMap<String, f64>,
    /// Total number of executed trades.
    pub total_executed_trades: usize,
    /// Total number of profitable trades.
    pub total_winning_trades: usize,
}

impl Portfolio {
    /// Initializes a new Portfolio state.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            available_cash: initial_capital,
            positions: HashMap::new(),
            average_entry_prices: HashMap::new(),
            total_executed_trades: 0,
            total_winning_trades: 0,
        }
    }

    /// Applies a finalized fill receipt to the portfolio.
    pub fn apply_fill(&mut self, fill: &FillReceipt) {
        let current_position = *self.positions.get(&fill.asset).unwrap_or(&0.0);

        if fill.is_buy {
            self.available_cash -= fill.transaction_cost + fill.commission_paid;
            self.positions
                .insert(fill.asset.clone(), current_position + fill.quantity);
            self.average_entry_prices
                .insert(fill.asset.clone(), fill.fill_price);
            self.total_executed_trades += 1;
        } else {
            let revenue = -fill.transaction_cost;
            self.available_cash += revenue - fill.commission_paid;
            self.positions
                .insert(fill.asset.clone(), current_position + fill.quantity);

            if let Some(&entry_price) = self.average_entry_prices.get(&fill.asset) {
                let absolute_quantity = fill.quantity.abs();
                let net_revenue = revenue - fill.commission_paid;
                let cost_basis = absolute_quantity * entry_price;
                if net_revenue > cost_basis {
                    self.total_winning_trades += 1;
                }
            }

            if let Some(&pos) = self.positions.get(&fill.asset)
                && pos <= 1e-8
            {
                self.positions.remove(&fill.asset);
                self.average_entry_prices.remove(&fill.asset);
            }
        }
    }

    /// Calculates the mark-to-market total equity.
    pub fn calculate_equity(&self, price_board: &HashMap<String, f64>) -> f64 {
        let mut holdings_valuation = 0.0;
        for (asset, &quantity) in &self.positions {
            if let Some(&current_price) = price_board.get(asset) {
                holdings_valuation += quantity * current_price;
            } else if let Some(&entry_price) = self.average_entry_prices.get(asset) {
                holdings_valuation += quantity * entry_price;
            }
        }
        self.available_cash + holdings_valuation
    }
}

/// Order context for safely delegating trades from Python strategies.
#[pyclass]
#[derive(Default)]
pub struct OrderContext {
    /// List of pending orders.
    pub pending_orders: Vec<Order>,
}

#[pymethods]
impl OrderContext {
    /// Commits a pending buy order.
    pub fn buy(&mut self, target_asset: String, target_quantity: f64, execution_price: f64) {
        if !execution_price.is_nan() && execution_price > 0.0 && target_quantity > 0.0 {
            self.pending_orders.push(Order {
                target_asset,
                target_quantity,
                execution_price,
            });
        }
    }

    /// Commits a pending sell order.
    pub fn sell(&mut self, target_asset: String, target_quantity: f64, execution_price: f64) {
        if !execution_price.is_nan() && execution_price > 0.0 && target_quantity > 0.0 {
            self.pending_orders.push(Order {
                target_asset,
                target_quantity: -target_quantity,
                execution_price,
            });
        }
    }
}

/// Sequential backtesting environment designed for multi-asset portfolio evaluation.
#[pyclass]
pub struct SequentialBacktester {
    /// Comprehensive historical market data.
    historical_data: DataFrame,
    /// Benchmark DataFrame.
    benchmark: Option<DataFrame>,
    /// Initial allocated capital.
    initial_capital: f64,
    /// Estimated slippage incurred upon buy order execution.
    buy_slippage: f64,
    /// Estimated slippage incurred upon sell order execution.
    sell_slippage: f64,
    /// Standard commission rate charged for buying.
    buy_commission_rate: f64,
    /// Standard commission rate charged for selling.
    sell_commission_rate: f64,
    /// Minimum absolute commission fee per executed trade.
    minimum_commission_fee: f64,
}

#[pymethods]
impl SequentialBacktester {
    /// Creates and configures a new SequentialBacktester instance.
    #[new]
    #[pyo3(signature = (historical_data, benchmark=None, initial_capital=100_000.0, buy_slippage=0.0, sell_slippage=0.0, buy_commission_rate=0.0003, sell_commission_rate=0.0003, minimum_commission_fee=5.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        historical_data: PyDataFrame,
        benchmark: Option<PyDataFrame>,
        initial_capital: f64,
        buy_slippage: f64,
        sell_slippage: f64,
        buy_commission_rate: f64,
        sell_commission_rate: f64,
        minimum_commission_fee: f64,
    ) -> Self {
        Self {
            historical_data: historical_data.into(),
            benchmark: benchmark.map(|b| b.into()),
            initial_capital,
            buy_slippage,
            sell_slippage,
            buy_commission_rate,
            sell_commission_rate,
            minimum_commission_fee,
        }
    }

    /// Executes the event-driven sequential backtesting simulation.
    pub fn run(
        &self,
        py: Python<'_>,
        strategy_callback: Py<PyAny>,
    ) -> PyResult<(PyDataFrame, PyDataFrame, HashMap<String, f64>)> {
        let total_simulation_periods = self.historical_data.height();

        let historical_benchmark: Vec<f64> = if let Some(bench) = &self.benchmark {
            if let Ok(bench_series) = bench.column(bench.get_column_names()[0]) {
                if let Ok(bench_f64) = bench_series.f64() {
                    bench_f64.into_no_null_iter().collect()
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let mut portfolio = Portfolio::new(self.initial_capital);
        let execution_engine = ExecutionEngine {
            buy_slippage: self.buy_slippage,
            sell_slippage: self.sell_slippage,
            buy_commission_rate: self.buy_commission_rate,
            sell_commission_rate: self.sell_commission_rate,
            minimum_commission_fee: self.minimum_commission_fee,
        };

        let mut market_price_board: HashMap<String, f64> = HashMap::new();
        let mut portfolio_equity_curve: Vec<f64> = Vec::with_capacity(total_simulation_periods);

        for current_period in 0..total_simulation_periods {
            let context_instance = Py::new(py, OrderContext::default()).unwrap();

            // Pass the current index (usize) rather than slicing the entire DataFrame on each tick
            let _ = strategy_callback.call1(py, (context_instance.clone_ref(py), current_period));
            let context_reference = context_instance.borrow(py);

            for order in &context_reference.pending_orders {
                market_price_board.insert(order.target_asset.clone(), order.execution_price);

                let current_position =
                    *portfolio.positions.get(&order.target_asset).unwrap_or(&0.0);

                if let Some(fill_receipt) = execution_engine.process_order(
                    order,
                    portfolio.available_cash,
                    current_position,
                ) {
                    portfolio.apply_fill(&fill_receipt);
                }
            }

            let current_period_equity = portfolio.calculate_equity(&market_price_board);
            portfolio_equity_curve.push(current_period_equity);
        }

        let summary_metrics = crate::backtest::metrics::calculate_summary(
            &portfolio_equity_curve,
            &historical_benchmark,
            self.initial_capital,
            portfolio.total_executed_trades,
            portfolio.total_winning_trades,
        );

        let positions_dataframe = DataFrame::empty();
        let capital_dataframe = DataFrame::new(
            total_simulation_periods,
            vec![Series::new("equity".into(), portfolio_equity_curve).into()],
        )
        .unwrap_or_else(|_| DataFrame::empty());

        Ok((
            PyDataFrame(positions_dataframe),
            PyDataFrame(capital_dataframe),
            summary_metrics,
        ))
    }
}
