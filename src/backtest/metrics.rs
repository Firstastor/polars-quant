use std::collections::HashMap;

const TRADING_DAYS_PER_YEAR: f64 = 252.0;
const RISK_FREE_RATE: f64 = 0.03;

/// Calculates comprehensive summary statistics for the backtest.
pub fn calculate_summary(
    equity_curve: &[f64],
    benchmark_curve: &[f64],
    initial_capital: f64,
    total_trades: usize,
    winning_trades: usize,
) -> HashMap<String, f64> {
    let mut summary_metrics = HashMap::new();
    let total_trading_days = equity_curve.len();

    if total_trading_days == 0 {
        return summary_metrics;
    }

    let mut maximum_drawdown = 0.0_f64;
    let mut maximum_recorded_equity = initial_capital;
    let mut previous_equity = initial_capital;
    let mut daily_returns = Vec::with_capacity(total_trading_days);

    for &daily_equity in equity_curve.iter() {
        if daily_equity > maximum_recorded_equity {
            maximum_recorded_equity = daily_equity;
        }

        let current_drawdown = if maximum_recorded_equity > 0.0 {
            (maximum_recorded_equity - daily_equity) / maximum_recorded_equity
        } else {
            0.0
        };

        if current_drawdown > maximum_drawdown {
            maximum_drawdown = current_drawdown;
        }

        let daily_return = if previous_equity > 0.0 {
            (daily_equity - previous_equity) / previous_equity
        } else {
            0.0
        };

        daily_returns.push(daily_return);
        previous_equity = daily_equity;
    }

    let final_equity = *equity_curve.last().unwrap_or(&initial_capital);
    let total_return = (final_equity - initial_capital) / initial_capital;

    let annualized_return = if total_return > -1.0 {
        (1.0 + total_return).powf(TRADING_DAYS_PER_YEAR / total_trading_days as f64) - 1.0
    } else {
        -1.0
    };

    let mean_daily_return = daily_returns.iter().sum::<f64>() / total_trading_days as f64;
    let degrees_of_freedom = (total_trading_days as f64 - 1.0).max(1.0);

    let variance_daily_return = daily_returns
        .iter()
        .map(|&return_rate| (return_rate - mean_daily_return).powi(2))
        .sum::<f64>()
        / degrees_of_freedom;

    let annualized_volatility = variance_daily_return.sqrt() * TRADING_DAYS_PER_YEAR.sqrt();

    let sharpe_ratio = if annualized_volatility > 0.0 {
        (annualized_return - RISK_FREE_RATE) / annualized_volatility
    } else {
        0.0
    };

    let win_rate = if total_trades > 0 {
        winning_trades as f64 / total_trades as f64
    } else {
        0.0
    };

    let mut portfolio_alpha = 0.0;
    let mut portfolio_beta = 0.0;

    if benchmark_curve.len() == total_trading_days {
        let mut benchmark_returns = Vec::with_capacity(total_trading_days);
        let mut previous_benchmark_value = benchmark_curve[0];

        for &benchmark_value in benchmark_curve.iter() {
            let benchmark_daily_return = if previous_benchmark_value > 0.0 {
                (benchmark_value - previous_benchmark_value) / previous_benchmark_value
            } else {
                0.0
            };
            benchmark_returns.push(benchmark_daily_return);
            previous_benchmark_value = benchmark_value;
        }

        let mean_benchmark_return =
            benchmark_returns.iter().sum::<f64>() / total_trading_days as f64;

        let variance_benchmark = benchmark_returns
            .iter()
            .map(|&return_rate| (return_rate - mean_benchmark_return).powi(2))
            .sum::<f64>()
            / degrees_of_freedom;

        let covariance_equity_benchmark = daily_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(&portfolio_return, &benchmark_return)| {
                (portfolio_return - mean_daily_return) * (benchmark_return - mean_benchmark_return)
            })
            .sum::<f64>()
            / degrees_of_freedom;

        if variance_benchmark > 0.0 {
            portfolio_beta = covariance_equity_benchmark / variance_benchmark;
        }

        let benchmark_start_value = *benchmark_curve.first().unwrap_or(&1.0);
        let benchmark_end_value = *benchmark_curve.last().unwrap_or(&1.0);

        let benchmark_total_return = if benchmark_start_value > 0.0 {
            (benchmark_end_value - benchmark_start_value) / benchmark_start_value
        } else {
            0.0
        };

        let benchmark_annualized_return = if benchmark_total_return > -1.0 {
            (1.0 + benchmark_total_return).powf(TRADING_DAYS_PER_YEAR / total_trading_days as f64)
                - 1.0
        } else {
            -1.0
        };

        portfolio_alpha = annualized_return
            - (RISK_FREE_RATE + portfolio_beta * (benchmark_annualized_return - RISK_FREE_RATE));
    }

    summary_metrics.insert("annualized_return".to_string(), annualized_return);
    summary_metrics.insert("max_drawdown".to_string(), maximum_drawdown);
    summary_metrics.insert("alpha".to_string(), portfolio_alpha);
    summary_metrics.insert("beta".to_string(), portfolio_beta);
    summary_metrics.insert("sharpe_ratio".to_string(), sharpe_ratio);
    summary_metrics.insert("max_profit".to_string(), total_return.max(0.0));
    summary_metrics.insert("win_rate".to_string(), win_rate);
    summary_metrics.insert("total_trades".to_string(), total_trades as f64);

    summary_metrics
}
