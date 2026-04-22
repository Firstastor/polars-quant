use itertools::Itertools;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use std::collections::HashMap;

/// Vectorized backtester for a single asset.
#[pyclass]
pub struct VectorizedBacktester {
    /// Price Series.
    price_data: Series,
    /// Buy signal Series.
    buy_signals: Series,
    /// Sell signal Series.
    sell_signals: Series,
    /// Benchmark Series.
    benchmark: Option<Series>,
    /// Initial capital amount.
    initial_capital: f64,
    /// Slippage incurred upon buying.
    buy_slippage: f64,
    /// Slippage incurred upon selling.
    sell_slippage: f64,
    /// Commission rate for buying.
    buy_commission_rate: f64,
    /// Commission rate for selling.
    sell_commission_rate: f64,
    /// Minimum commission fee per trade.
    min_commission: f64,
    /// Target position size (fraction of equity).
    position_size: f64,
}

#[pymethods]
impl VectorizedBacktester {
    /// Creates a new VectorizedBacktester instance.
    #[new]
    #[pyo3(signature = (price, buy_signal, sell_signal, benchmark=None, initial_capital=100_000.0, buy_slippage=0.0, sell_slippage=0.0, buy_commission_rate=0.0003, sell_commission_rate=0.0003, min_commission=5.0, position_size=1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        price: PySeries,
        buy_signal: PySeries,
        sell_signal: PySeries,
        benchmark: Option<PySeries>,
        initial_capital: f64,
        buy_slippage: f64,
        sell_slippage: f64,
        buy_commission_rate: f64,
        sell_commission_rate: f64,
        min_commission: f64,
        position_size: f64,
    ) -> Self {
        Self {
            price_data: price.into(),
            buy_signals: buy_signal.into(),
            sell_signals: sell_signal.into(),
            benchmark: benchmark.map(|b| b.into()),
            initial_capital,
            buy_slippage,
            sell_slippage,
            buy_commission_rate,
            sell_commission_rate,
            min_commission,
            position_size,
        }
    }

    /// Executes the vectorized backtesting process.
    pub fn run(&self) -> PyResult<(PyDataFrame, PyDataFrame, HashMap<String, f64>)> {
        let historical_prices: Vec<f64> = self
            .price_data
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .into_iter()
            .map(|opt| opt.unwrap_or(f64::NAN))
            .collect();

        let historical_buys: Vec<bool> = self
            .buy_signals
            .cast(&DataType::Boolean)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .bool()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .into_iter()
            .map(|opt| opt.unwrap_or(false))
            .collect();

        let historical_sells: Vec<bool> = self
            .sell_signals
            .cast(&DataType::Boolean)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .bool()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .into_iter()
            .map(|opt| opt.unwrap_or(false))
            .collect();

        let total_periods = historical_prices.len();

        let historical_benchmark: Vec<f64> = if let Some(bench) = &self.benchmark {
            if let Ok(casted) = bench.cast(&DataType::Float64) {
                if let Ok(bench_f64) = casted.f64() {
                    bench_f64
                        .into_iter()
                        .map(|opt| opt.unwrap_or(f64::NAN))
                        .collect()
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let mut maximum_drawdown = 0.0;
        let mut total_executed_trades = 0;
        let mut total_winning_trades = 0;

        // State tracking: (current_position_size, available_cash, peak_equity_recorded, last_entry_cost)
        let initial_state = (0.0_f64, self.initial_capital, self.initial_capital, 0.0_f64);

        let (portfolio_positions, portfolio_cash, portfolio_equity): (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        ) = itertools::multizip((
            historical_prices.iter(),
            historical_buys.iter(),
            historical_sells.iter(),
        ))
        .scan(
            initial_state,
            |state, (&current_price, &trigger_buy, &trigger_sell)| {
                let (mut position_size, mut available_cash, mut peak_equity, mut entry_cost) =
                    *state;

                if current_price.is_nan() || current_price <= 0.0 {
                    let current_equity = available_cash + position_size * current_price;
                    return Some((position_size, available_cash, current_equity));
                }

                if trigger_buy && position_size == 0.0 {
                    let execution_price = current_price + self.buy_slippage;
                    let current_equity = available_cash + position_size * current_price;
                    let capital_to_deploy = current_equity * self.position_size;
                    let target_quantity = (capital_to_deploy / execution_price).floor();

                    if target_quantity > 0.0 {
                        let transaction_cost = target_quantity * execution_price;
                        let commission_fee =
                            (transaction_cost * self.buy_commission_rate).max(self.min_commission);

                        position_size += target_quantity;
                        available_cash -= transaction_cost + commission_fee;
                        entry_cost = position_size * current_price;
                        total_executed_trades += 1;
                    }
                } else if trigger_sell && position_size > 0.0 {
                    let execution_price = current_price - self.sell_slippage;
                    let transaction_revenue = position_size * execution_price;
                    let commission_fee =
                        (transaction_revenue * self.sell_commission_rate).max(self.min_commission);

                    let net_revenue = transaction_revenue - commission_fee;
                    if net_revenue > entry_cost {
                        total_winning_trades += 1;
                    }

                    available_cash += net_revenue;
                    position_size = 0.0;
                }

                let current_equity = available_cash + position_size * current_price;

                if current_equity > peak_equity {
                    peak_equity = current_equity;
                }

                let drawdown = (peak_equity - current_equity) / peak_equity;
                if drawdown > maximum_drawdown {
                    maximum_drawdown = drawdown;
                }

                // Update mutable state for the next iteration
                *state = (position_size, available_cash, peak_equity, entry_cost);

                Some((position_size, available_cash, current_equity))
            },
        )
        .multiunzip();

        let summary_metrics = crate::backtest::metrics::calculate_summary(
            &portfolio_equity,
            &historical_benchmark,
            self.initial_capital,
            total_executed_trades,
            total_winning_trades,
        );

        let positions_dataframe = DataFrame::new(
            total_periods,
            vec![Series::new("position".into(), portfolio_positions).into()],
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let capital_dataframe = DataFrame::new(
            total_periods,
            vec![
                Series::new("cash".into(), portfolio_cash).into(),
                Series::new("equity".into(), portfolio_equity).into(),
            ],
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok((
            PyDataFrame(positions_dataframe),
            PyDataFrame(capital_dataframe),
            summary_metrics,
        ))
    }
}
