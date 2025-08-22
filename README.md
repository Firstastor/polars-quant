# polars_quant

polars_quant is a Rust-backed Python extension exposing a small set of real APIs implemented in this repository:

- polars_quant.history(stock_code: str, scale: int = 240, datalen: int = 3650, timeout: int = 10)
  - Fetch historical OHLCV from the remote source. Returns a list of records (each having keys: day, open, close, high, low, volume) or None.

- class polars_quant.Backtrade
  - Backtrade.run(data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0)
  - Backtrade.portfolio(data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0)
  - Instance attributes/methods: results, trades, summary(), speed

Quick usage examples

1) Fetch history

```python
import polars as pl
import polars_quant

items = polars_quant.history("sh600519", scale=240, datalen=365, timeout=10)
if items is None:
    print("No data")
else:
    df = pl.DataFrame(items)
    print(df.head())
```

2) Backtest (single symbol)

```python
import polars as pl
import polars_quant as plqt

data = pl.DataFrame({
    "date": ["2024-01-01","2024-01-02","2024-01-03"],
    "SYM": [100.0, 101.5, 99.0],
})
entries = pl.DataFrame({"date": data["date"], "SYM": [False, True, False]})
exits = pl.DataFrame({"date": data["date"], "SYM": [False, False, True]})

bt = plqt.Backtrade.run(data, entries, exits, init_cash=100000.0, fee=0.0005)
print(bt.summary())
if getattr(bt, "results", None) is not None:
    print(bt.results.head())
```

Notes
- Keep `data`, `entries`, `exits` aligned: column 0 = date, columns 1..N = symbols (one column per symbol).
- `entries`/`exits` columns accept booleans or integers as flags.
- GitHub repository README is static; no automatic locale switching. Use this pattern to maintain parallel language files.