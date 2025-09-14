use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

#[pyclass]
pub struct Backtrade {
    results: Option<DataFrame>,
    trades: Option<DataFrame>,
    summary: Option<Py<PyDict>>,
    speed: f64,
}

#[pymethods]
impl Backtrade {
    #[new]
    fn new(results: Option<PyDataFrame>, trades: Option<PyDataFrame>) -> PyResult<Self> {
        Ok(Self {
            results: results.map(|df| df.0),
            trades: trades.map(|df| df.0),
            summary: None,
            speed: 0.0,
        })
    }

    #[classmethod]
    #[pyo3(signature = (data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0))]
    fn run(
        _cls: &Bound<PyType>,
        data: PyDataFrame,
        entries: PyDataFrame,
        exits: PyDataFrame,
        init_cash: f64,
        fee: f64,
        slip: f64,
        size: f64,
    ) -> PyResult<Self> {
        let start = std::time::Instant::now();

        let df_data = data.0;
        let df_entries = entries.0;
        let df_exits = exits.0;

        let nrows = df_data.height();
        let ncols = df_data.width();


        if ncols == 0 {
            return Err(PyValueError::new_err("data has no columns"));
        }
        if nrows == 0 {
            return Err(PyValueError::new_err("data has no rows"));
        }

        let dates = df_data[0].as_series().unwrap();

        let per_symbol: Vec<_> = (1..ncols).into_par_iter().map(|col_idx| {
            let col = df_data.select_at_idx(col_idx).unwrap();
            let sym = &col.name().to_string();

            let prices: Vec<f64> =col.f64().unwrap()
            .into_iter()
            .map(|opt| opt.unwrap_or(0.0))
            .collect();

            let entry_col = df_entries.select_at_idx(col_idx)
                .ok_or_else(|| PyValueError::new_err(format!("entries missing column index {}", col_idx)))?;
            let exit_col = df_exits.select_at_idx(col_idx)
                .ok_or_else(|| PyValueError::new_err(format!("exits missing column index {}", col_idx)))?;

            let entry_flags: Vec<bool> = match entry_col.dtype() {
                DataType::Boolean => entry_col.bool().unwrap()
                    .into_iter()
                    .map(|o| o.unwrap_or(false))
                    .collect(),
                _ => vec![false; nrows],
            };
            let exit_flags: Vec<bool> = match exit_col.dtype() {
                DataType::Boolean => exit_col.bool().unwrap()
                    .into_iter()
                    .map(|o| o.unwrap_or(false))
                    .collect(),
                _ => vec![false; nrows],
            };

            let mut res_symbol: Vec<String>   = Vec::with_capacity(nrows);
            let mut res_date:   Vec<String>   = Vec::with_capacity(nrows);
            let mut res_cash:   Vec<f64>      = Vec::with_capacity(nrows);
            let mut res_equity: Vec<f64>      = Vec::with_capacity(nrows);

            let mut trades: Vec<(String, String, f64, i64, Option<String>, Option<f64>)> =
                Vec::with_capacity(nrows / 2);

            let mut cash = init_cash;
            let mut positions: i64 = 0;
            for i in 0..nrows {
                let today = dates.get(i).map(|v| v.to_string()).unwrap_or_default();

                let p = prices[i];

                // Optimize condition checks
                let can_entry = entry_flags[i] && positions == 0 && p > 0.01;
                let can_exit = exit_flags[i] && positions > 0 && p > 0.01;

                if can_entry {
                    let buy_price = (1.0 + slip) * p;
                    let max_share = (((1.0 - fee) * cash / buy_price) / 100.0).floor() * 100.0;
                    let buy_share = (max_share * size).floor() as i64;

                    if buy_share > 0 {
                        cash -= buy_price * buy_share as f64;
                        positions = buy_share;
                        trades.push((
                            sym.clone(),
                            today.clone(),
                            buy_price,
                            buy_share,
                            None,
                            None,
                        ));
                    }
                } else if can_exit {
                    let sell_price = (1.0 - fee) * (1.0 - slip) * p;
                    cash += sell_price * positions as f64;
                    if let Some(last) = trades.last_mut() {
                        last.4 = Some(today.clone());
                        last.5 = Some(sell_price);
                    }
                    positions = 0;
                }

                let equity = cash + p * positions as f64;
                res_symbol.push(sym.clone());
                res_date.push(today);
                res_cash.push(cash);
                res_equity.push(equity);
            }

            Ok::<_, PyErr>((res_symbol, res_date, res_cash, res_equity, trades))
        }).collect::<Result<Vec<_>, PyErr>>()?;

        let total_rows: usize = per_symbol.iter().map(|(s, _, _, _, _)| s.len()).sum();
        let mut res_symbol: Vec<String>   = Vec::with_capacity(total_rows);
        let mut res_date:   Vec<String>   = Vec::with_capacity(total_rows);
        let mut res_cash:   Vec<f64>      = Vec::with_capacity(total_rows);
        let mut res_equity: Vec<f64>      = Vec::with_capacity(total_rows);
        let mut trades_all: Vec<(String, String, f64, i64, Option<String>, Option<f64>)> = Vec::new();

        for (s, d, c, e, t) in per_symbol {
            res_symbol.extend(s);
            res_date.extend(d);
            res_cash.extend(c);
            res_equity.extend(e);
            trades_all.extend(t);
        }

        let results_df = DataFrame::new(vec![
            Column::new("Symbol".into(), res_symbol),
            Column::new("Date".into(),   res_date),
            Column::new("Cash".into(),   res_cash),
            Column::new("Equity".into(), res_equity),
        ]).unwrap();

        let mut t_sym:    Vec<String>      = Vec::with_capacity(trades_all.len());
        let mut t_bdate:  Vec<String>      = Vec::with_capacity(trades_all.len());
        let mut t_bprice: Vec<f64>         = Vec::with_capacity(trades_all.len());
        let mut t_share:  Vec<i64>         = Vec::with_capacity(trades_all.len());
        let mut t_sdate:  Vec<Option<String>> = Vec::with_capacity(trades_all.len());
        let mut t_sprice: Vec<Option<f64>> = Vec::with_capacity(trades_all.len());

        for (s, bd, bp, sh, sd, sp) in trades_all.into_iter() {
            t_sym.push(s);
            t_bdate.push(bd);
            t_bprice.push(bp);
            t_share.push(sh);
            t_sdate.push(sd);
            t_sprice.push(sp);
        }

        let trades_df = DataFrame::new(vec![
            Column::new("symbol".into(),     t_sym),
            Column::new("buy_date".into(),   t_bdate),
            Column::new("buy_price".into(),  t_bprice),
            Column::new("share".into(),      t_share),
            Column::new("sell_date".into(),  t_sdate),
            Column::new("sell_price".into(), t_sprice),
        ]).unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let total_bars = nrows * (ncols - 1);
        let speed = if elapsed > 0.0 {
            total_bars as f64 / elapsed
        } else {
            0.0
        };

        Ok(Self {
            results: Some(results_df),
            trades: Some(trades_df),
            summary: None,
            speed,
        })
    }

    fn results(&self) -> PyResult<Option<PyDataFrame>> {
        Ok(self.results.clone().map(PyDataFrame))
    }

    fn summary<'py>(&'py mut self, py: Python<'py>) -> PyResult<()> {
        if self.summary.is_none() {
            self.cal_summary(py)?;
        }

        let root: Bound<'py, PyDict> = self.summary
            .as_ref()
            .unwrap()
            .as_ref()
            .extract::<Bound<'py, PyDict>>(py)?;

        let overall: Bound<'py, PyDict> = root.get_item("overall")?
            .ok_or_else(|| PyValueError::new_err("missing overall summary"))?
            .extract()?;

        let per_symbol: Bound<'py, PyDict> = root.get_item("per_symbol")?
            .ok_or_else(|| PyValueError::new_err("missing per_symbol summary"))?
            .extract()?;

        fn get_f64<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<f64> {
            Ok(d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("missing {}", key)))?
                .extract::<f64>()?)
        }
        fn get_usize<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<usize> {
            Ok(d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("missing {}", key)))?
                .extract::<usize>()?)
        }

        let init     = get_f64(&overall, "initial_capital")?;
        let final_eq = get_f64(&overall, "final_equity")?;
        let tot_ret  = get_f64(&overall, "total_return")? * 100.0;
        let ann_ret  = get_f64(&overall, "annualized_return")? * 100.0;
        let max_dd   = get_f64(&overall, "max_drawdown")? * 100.0;
        let sharpe   = get_f64(&overall, "sharpe_ratio")?;
        let speed    = self.speed;

        let mut out = String::new();
        out.push_str(&format!(
            r#"
================ Backtest Summary (Overall) =================
Avg Initial Capital      : {init:.2}
Avg Final Equity         : {final_eq:.2}
Avg Total Return         : {tot_ret:.2}%
Avg Annualized Return    : {ann_ret:.2}%
Avg Max Drawdown         : {max_dd:.2}%
Avg Sharpe Ratio         : {sharpe:.2}

Backtest Speed       : {speed:.2} bars/s
==============================================================
"#,
        ));

        out.push_str(&format!("\n=============== Per-Symbol Evaluation ===============\n"));
        for (sym, val) in per_symbol.iter() {
            let sym: String = sym.extract()?;
            let d: Bound<'py, PyDict> = val.extract()?;

            let trades   = get_usize(&d, "total_trades")?;
            let win      = get_usize(&d, "winning_trades")?;
            let loss     = get_usize(&d, "losing_trades")?;
            let win_rate = get_f64(&d, "win_rate")? * 100.0;
            let pf       = get_f64(&d, "profit_factor")?;
            let avg      = get_f64(&d, "avg_trade_return")? * 100.0;

            out.push_str(&format!(
                r#"
Symbol: {sym}
Total Trades     : {trades}
Winning Trades   : {win}
Losing Trades    : {loss}
Win Rate         : {win_rate:.2}%
Profit Factor    : {pf:.2}
Avg Trade Return : {avg:.2}%
-------------------------------------------------------
"#,
            ));
        }

        let builtins = py.import("builtins")?;
        builtins.getattr("print")?.call1((out,))?;

        Ok(())
    }

    fn trades(&self) -> PyResult<Option<PyDataFrame>> {
        Ok(self.trades.clone().map(PyDataFrame))
    }
}

impl Backtrade {
    fn cal_summary(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.results.is_none() || self.trades.is_none() {
            return Err(PyValueError::new_err("No results or trades data available"));
        }

        let results = self.results.as_ref().unwrap();
        let trades = self.trades.as_ref().unwrap();
        let symbols = results.column("Symbol").unwrap().str().unwrap();
        let equity_col = results.column("Equity").unwrap().f64().unwrap();

        use std::collections::HashMap;
        let mut equity_map: HashMap<String, Vec<f64>> = HashMap::new();
        for i in 0..results.height() {
            if let (Some(sym), Some(eq)) = (symbols.get(i), equity_col.get(i)) {
                equity_map.entry(sym.to_string()).or_default().push(eq);
            }
        }

        let mut overall_init = Vec::new();
        let mut overall_final = Vec::new();
        let mut overall_total_return = Vec::new();
        let mut overall_annualized = Vec::new();
        let mut overall_max_dd = Vec::new();
        let mut overall_sharpe = Vec::new();

        for (_sym, eqs) in &equity_map {
            if eqs.is_empty() { continue; }

            let initial = eqs.first().unwrap_or(&0.0);
            let final_eq = eqs.last().unwrap_or(&0.0);
            let total_return = if *initial != 0.0 { (final_eq - initial) / initial } else { 0.0 };
            let n_days = eqs.len();
            let years = if n_days > 1 { (n_days - 1) as f64 / 252.0 } else { 0.0 };
            let annualized = if years > 0.0 { (1.0 + total_return).powf(1.0 / years) - 1.0 } else { 0.0 };

            let mut cum_max = f64::NEG_INFINITY;
            let mut max_dd = 0.0;
            for &v in eqs {
                if v > cum_max { cum_max = v; }
                if cum_max > 0.0 {
                    let dd = 1.0 - v / cum_max;
                    if dd > max_dd { max_dd = dd; }
                }
            }

            let mut rets: Vec<f64> = Vec::with_capacity(eqs.len().saturating_sub(1));
            for i in 1..eqs.len() {
                let p0 = eqs[i - 1];
                let p1 = eqs[i];
                if p0 != 0.0 { rets.push((p1 - p0) / p0); }
            }
            let sharpe = if !rets.is_empty() {
                let mean = rets.iter().sum::<f64>() / rets.len() as f64;
                let var = rets.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rets.len() as f64;
                let std = var.sqrt();
                if std > 0.0 { mean / std * 252.0_f64.sqrt() } else { 0.0 }
            } else { 0.0 };

            overall_init.push(*initial);
            overall_final.push(*final_eq);
            overall_total_return.push(total_return);
            overall_annualized.push(annualized);
            overall_max_dd.push(max_dd);
            overall_sharpe.push(sharpe);
        }

        let avg = |v: &Vec<f64>| if !v.is_empty() { v.iter().sum::<f64>() / v.len() as f64 } else { 0.0 };
        
        let overall = PyDict::new(py);
        overall.set_item("initial_capital", avg(&overall_init))?;
        overall.set_item("final_equity", avg(&overall_final))?;
        overall.set_item("total_return", avg(&overall_total_return))?;
        overall.set_item("annualized_return", avg(&overall_annualized))?;
        overall.set_item("max_drawdown", avg(&overall_max_dd))?;
        overall.set_item("sharpe_ratio", avg(&overall_sharpe))?;

        let bp = trades.column("buy_price").unwrap().f64().unwrap();
        let sp = trades.column("sell_price").unwrap().f64().unwrap();
        let syms = trades.column("symbol").unwrap().str().unwrap();

        let mut sym_map: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        for i in 0..trades.height() {
            if let (Some(b), Some(s), Some(sym)) = (bp.get(i), sp.get(i), syms.get(i)) {
                sym_map.entry(sym.to_string()).or_default().push((b, s));
            }
        }

        let per_symbol = PyDict::new(py);
        for (sym, arr) in sym_map {
            let mut t_trades = 0usize;
            let mut w_trades = 0usize;
            let mut l_trades = 0usize;
            let mut t_profit = 0.0f64;
            let mut t_loss = 0.0f64;
            let mut sum_ret = 0.0f64;

            for (b, s) in arr.iter() {
                t_trades += 1;
                let profit = s - b;
                let r = if *b != 0.0 { profit / b } else { 0.0 };
                sum_ret += r;
                if profit > 0.0 { w_trades += 1; t_profit += profit; } 
                else { l_trades += 1; t_loss += profit; }
            }

            let w_rate = if t_trades > 0 { w_trades as f64 / t_trades as f64 } else { 0.0 };
            let pf = if t_loss < 0.0 { t_profit / (-t_loss) } else { f64::INFINITY };
            let avg_ret = if t_trades > 0 { sum_ret / t_trades as f64 } else { 0.0 };

            let dict = PyDict::new(py);
            dict.set_item("total_trades", t_trades)?;
            dict.set_item("winning_trades", w_trades)?;
            dict.set_item("losing_trades", l_trades)?;
            dict.set_item("win_rate", w_rate)?;
            dict.set_item("profit_factor", pf)?;
            dict.set_item("avg_trade_return", avg_ret)?;
            per_symbol.set_item(sym, dict)?;
        }

        // --- 汇总到 summary ---
        let summary = PyDict::new(py);
        summary.set_item("overall", overall)?;
        summary.set_item("per_symbol", per_symbol)?;

        self.summary = Some(summary.into());
        Ok(())
    }
}

#[pyclass]
pub struct Portfolio {
    results: Option<DataFrame>,
    trades: Option<DataFrame>,
    summary: Option<Py<PyDict>>,
    speed: f64,
}

#[pymethods]
impl Portfolio {
    #[new]
    fn new(results: Option<PyDataFrame>, trades: Option<PyDataFrame>) -> PyResult<Self> {
        Ok(Self {
            results: results.map(|df| df.0),
            trades: trades.map(|df| df.0),
            summary: None,
            speed: 0.0,
        })
    }

    #[classmethod]
    #[pyo3(signature = (data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0))]
    fn run(
        _cls: &Bound<PyType>,
        data: PyDataFrame,
        entries: PyDataFrame,
        exits: PyDataFrame,
        init_cash: f64,
        fee: f64,
        slip: f64,
        size: f64,
    ) -> PyResult<Self> {

        let start = std::time::Instant::now();

        let df_data = data.0;
        let df_entries = entries.0;
        let df_exits = exits.0;

        let nrows = df_data.height();
        let ncols = df_data.width();

        if ncols <= 1 {
            return Err(PyValueError::new_err(
                "data must have at least 1 symbol column besides date",
            ));
        }
        if nrows == 0 {
            return Err(PyValueError::new_err("data has no rows"));
        }

        let dates: Vec<String> = (0..nrows)
            .map(|i| {df_data[0].get(i).map(|v| v.to_string()).unwrap_or_default()})
            .collect();

        let mut symbols: Vec<String> = Vec::with_capacity(ncols - 1);
        let mut prices_all: Vec<Vec<f64>> = Vec::with_capacity(ncols - 1);
        let mut entries_all: Vec<Vec<bool>> = Vec::with_capacity(ncols - 1);
        let mut exits_all: Vec<Vec<bool>> = Vec::with_capacity(ncols - 1);

        for col_idx in 1..ncols {
            let col = df_data.select_at_idx(col_idx).unwrap();
            symbols.push(col.name().to_string());

            let prices: Vec<f64> = col
            .f64()
            .unwrap()
            .into_iter()
            .map(|opt| opt.unwrap_or(0.0))
            .collect()
            ;
            prices_all.push(prices);

            let entry_col = df_entries.select_at_idx(col_idx).ok_or_else(|| {
                PyValueError::new_err(format!("entries missing column index {}", col_idx))
            })?;
            let exit_col = df_exits.select_at_idx(col_idx).ok_or_else(|| {
                PyValueError::new_err(format!("exits missing column index {}", col_idx))
            })?;

            let entry_flags: Vec<bool> = match entry_col.dtype() {
                DataType::Boolean => entry_col
                    .bool()
                    .unwrap()
                    .into_iter()
                    .map(|o| o.unwrap_or(false))
                    .collect(),
                _ => vec![false; nrows],
            };
            let exit_flags: Vec<bool> = match exit_col.dtype() {
                DataType::Boolean => exit_col
                    .bool()
                    .unwrap()
                    .into_iter()
                    .map(|o| o.unwrap_or(false))
                    .collect(),
                _ => vec![false; nrows],
            };

            entries_all.push(entry_flags);
            exits_all.push(exit_flags);
        }
        let mut cash = init_cash;
        let mut equity = cash;
        let mut positions = 0i64;
        
        let mut res_date = Vec::with_capacity(nrows);
        let mut res_cash = Vec::with_capacity(nrows);
        let mut res_equity = Vec::with_capacity(nrows);

        let mut trades: Vec<(String, String, f64, i64, Option<String>, Option<f64>)> = Vec::new();

        for i in 0..nrows {
            let today = &dates[i];
            
            for (col_idx, sym) in symbols.iter().enumerate() {
                let p = prices_all[col_idx][i];
                let entry_flag = entries_all[col_idx][i];
                let exit_flag = exits_all[col_idx][i];
                let symbol = &trades.last().map(|(first, _, _, _, _, _)| first.clone()).unwrap_or("None".into());

                // Optimize condition checks
                let can_entry = entry_flag && positions == 0 && p > 0.01;
                let can_exit = positions > 0 && sym == symbol && p > 0.01 && exit_flag;

                if can_entry {
                    let buy_price = (1.0 + slip) * p;
                    let max_share = (((1.0 - fee) * cash / buy_price) / 100.0).floor() * 100.0;
                    let buy_share = (max_share * size).floor() as i64;

                    if buy_share > 0 {
                        cash -= buy_price * buy_share as f64;
                        positions = buy_share;
                        trades.push((
                            sym.clone(),
                            today.clone(),
                            buy_price,
                            buy_share,
                            None,
                            None,
                        ));
                    }
                }

                if can_exit {
                    let sell_price = (1.0 - fee) * (1.0 - slip) * p;
                    cash += sell_price * positions as f64;
                    if let Some(last) = trades.last_mut() {
                        last.4 = Some(today.clone());
                        last.5 = Some(sell_price);
                    }
                    positions = 0;
                }
                equity = cash + p * positions as f64;
            }
            
            res_date.push(today.clone());
            res_cash.push(cash);
            res_equity.push(equity);
        }

        let results_df = DataFrame::new(vec![
            Column::new("Date".into(), res_date),
            Column::new("Cash".into(), res_cash),
            Column::new("Equity".into(), res_equity),
        ])
        .unwrap();

        let (mut t_sym, mut t_bdate, mut t_bprice, mut t_share, mut t_sdate, mut t_sprice) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

        for (s, bd, bp, sh, sd, sp) in trades.into_iter() {
            t_sym.push(s);
            t_bdate.push(bd);
            t_bprice.push(bp);
            t_share.push(sh);
            t_sdate.push(sd);
            t_sprice.push(sp);
        }

        let trades_df = DataFrame::new(vec![
            Column::new("symbol".into(), t_sym),
            Column::new("buy_date".into(), t_bdate),
            Column::new("buy_price".into(), t_bprice),
            Column::new("share".into(), t_share),
            Column::new("sell_date".into(), t_sdate),
            Column::new("sell_price".into(), t_sprice),
        ])
        .unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let total_bars = nrows * (ncols - 1);
        let speed = if elapsed > 0.0 {
            total_bars as f64 / elapsed
        } else {
            0.0
        };

        Ok(Self {
            results: Some(results_df),
            trades: Some(trades_df),
            summary: None,
            speed,
        })
    }

    fn results(&self) -> PyResult<Option<PyDataFrame>> {
        Ok(self.results.clone().map(PyDataFrame))
    }

    fn summary<'py>(&'py mut self, py: Python<'py>) -> PyResult<()> {
        if self.summary.is_none() {
            self.cal_summary(py)?;
        }

        let root: Bound<'py, PyDict> = self.summary
            .as_ref()
            .unwrap()
            .as_ref()
            .extract::<Bound<'py, PyDict>>(py)?;

        let overall: Bound<'py, PyDict> = root.get_item("overall")?
            .ok_or_else(|| PyValueError::new_err("missing overall summary"))?
            .extract()?;

        fn get_f64<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<f64> {
            Ok(d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("missing {}", key)))?
                .extract::<f64>()?)
        }
        fn get_usize<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<usize> {
            Ok(d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("missing {}", key)))?
                .extract::<usize>()?)
        }

        let init     = get_f64(&overall, "initial_capital")?;
        let final_eq = get_f64(&overall, "final_equity")?;
        let tot_ret  = get_f64(&overall, "total_return")? * 100.0;
        let ann_ret  = get_f64(&overall, "annualized_return")? * 100.0;
        let max_dd   = get_f64(&overall, "max_drawdown")? * 100.0;
        let sharpe   = get_f64(&overall, "sharpe_ratio")?;
        let trades   = get_usize(&overall, "total_trades")?;
        let win      = get_usize(&overall, "winning_trades")?;
        let loss     = get_usize(&overall, "losing_trades")?;
        let win_rate = get_f64(&overall, "win_rate")? * 100.0;
        let pf       = get_f64(&overall, "profit_factor")?;
        let avg      = get_f64(&overall, "avg_trade_return")? * 100.0;
        let speed    = self.speed;

        let mut out = String::new();
        out.push_str(&format!(
            r#"
        ================= Backtest Summary (Overall) =================
        Initial Capital      : {init:.2}
        Final Equity         : {final_eq:.2}
        Total Return         : {tot_ret:.2}%
        Annualized Return    : {ann_ret:.2}%
        Max Drawdown         : {max_dd:.2}%
        Sharpe Ratio         : {sharpe:.2}

        Total Trades         : {trades}
        Winning Trades       : {win}
        Losing Trades        : {loss}
        Win Rate             : {win_rate:.2}%
        Profit Factor        : {pf:.2}
        Average Trade Return : {avg:.2}%

        Backtest Speed       : {speed:.2} bars/s
        ===============================================================
        "#,
        ));

        let builtins = py.import("builtins")?;
        builtins.getattr("print")?.call1((out,))?;

        Ok(())
    }

    fn trades(&self) -> PyResult<Option<PyDataFrame>> {
        Ok(self.trades.clone().map(PyDataFrame))
    }
}

impl Portfolio {
    fn cal_summary(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.results.is_none() || self.trades.is_none() {
            return Err(PyValueError::new_err("No results or trades data available"));
        }
        let results = self.results.as_ref().unwrap();
        let trades = self.trades.as_ref().unwrap();

        let equity = results.column("Equity").unwrap().f64().unwrap();
        let n = results.column("Date").unwrap().unique().unwrap().len();
        if n == 0 {
            return Err(PyValueError::new_err("results has empty Equity"));
        }
        let initial_capital = equity.get(0).unwrap_or(0.0);
        let final_equity = equity.get(n - 1).unwrap_or(0.0);
        let total_return = if initial_capital != 0.0 {
            (final_equity - initial_capital) / initial_capital
        } else {
            0.0
        };
        let years = if n > 1 {
            (n as f64 - 1.0) / 252.0
        } else {
            0.0
        };
        let annualized_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        let mut total_trades = 0;
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut total_profit = 0.0;
        let mut total_loss = 0.0;
        let mut sum_trade_return = 0.0;

        let bp = trades.column("buy_price").unwrap().f64().unwrap();
        let sp = trades.column("sell_price").unwrap().f64().unwrap();

        for i in 0..trades.height() {
            if let (Some(b), Some(s)) = (bp.get(i), sp.get(i)) {
                total_trades += 1;
                let profit = s - b;
                let r = if b != 0.0 { (s - b) / b } else { 0.0 };
                sum_trade_return += r;
                if profit > 0.0 {
                    winning_trades += 1;
                    total_profit += profit;
                } else {
                    losing_trades += 1;
                    total_loss += profit;
                }
            }
        }

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        let profit_factor = if total_loss < 0.0 {
            total_profit / (-total_loss)
        } else {
            f64::INFINITY
        };
        let avg_trade_return = if total_trades > 0 {
            sum_trade_return / total_trades as f64
        } else {
            0.0
        };

        let mut cum_max = f64::NEG_INFINITY;
        let mut max_dd = 0.0f64;
        for i in 0..n {
            if let Some(v) = equity.get(i) {
                if v > cum_max {
                    cum_max = v;
                }
                if cum_max > 0.0 {
                    let dd = 1.0 - v / cum_max;
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }
            }
        }
        let mut rets: Vec<f64> = Vec::with_capacity(n.saturating_sub(1));
        for i in 1..n {
            if let (Some(p1), Some(p0)) = (equity.get(i), equity.get(i - 1)) {
                if p0 != 0.0 {
                    rets.push((p1 - p0) / p0);
                }
            }
        }
        let sharpe_ratio = if !rets.is_empty() {
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rets.len() as f64;
            let std = var.sqrt();
            if std > 0.0 {
                mean / std * 252.0_f64.sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        let out = PyDict::new(py);
        let overall = PyDict::new(py);
        overall.set_item("initial_capital", initial_capital)?;
        overall.set_item("final_equity", final_equity)?;
        overall.set_item("total_return", total_return)?;
        overall.set_item("annualized_return", annualized_return)?;
        overall.set_item("total_trades", total_trades)?;
        overall.set_item("winning_trades", winning_trades)?;
        overall.set_item("losing_trades", losing_trades)?;
        overall.set_item("win_rate", win_rate)?;
        overall.set_item("profit_factor", profit_factor)?;
        overall.set_item("avg_trade_return", avg_trade_return)?;
        overall.set_item("max_drawdown", max_dd)?;
        overall.set_item("sharpe_ratio", sharpe_ratio)?;

        out.set_item("overall", overall)?;

        self.summary = Some(out.into());
        Ok(())
    }
}
