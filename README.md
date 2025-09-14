# polars-quant


# polars-quant 🧮📊

A compact, code-first toolkit of technical indicators and simple backtesting utilities implemented in Rust and exposed to Python via PyO3. Designed for research and small-scale backtests using Polars DataFrames.

Key points

- License: MIT (see `LICENSE`).
- No GPU/CUDA features are included — CPU-only Rust + Polars implementation.
- The Python API signatures are authoritative in `python/polars_quant/polars_quant.pyi`.

Install (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install polars polars-quant
```

From source (dev)

```powershell
git clone https://github.com/Firstastor/polars-quant.git
cd polars-quant
pip install -e .
# If building native extensions you will need Rust toolchain
# rustup and MSVC build tools on Windows
```

Quick examples (detailed) ⚡

1) Compute a 3-period moving average on a price series

```python
import polars as pl
import polars_quant as plqt

df = pl.DataFrame({'close': [100.0, 101.0, 102.0, 103.0, 104.0]})

# `ma` returns a list of Series (one per numeric column).
ma_list = plqt.ma(df, 3)
print(type(ma_list), len(ma_list))
# To attach results back to a DataFrame:
res_df = df.with_columns(ma_list)
print(res_df)
```

2) MACD example (single series)

```python
df = pl.DataFrame({'close': [100.0, 101.0, 102.5, 101.0, 103.0, 104.0]})
macd_series = plqt.macd(df, fast=12, slow=26, signal=9)
# macd returns [dif, dea, macd] series
res = df.with_columns(macd_series)
print(res)
```

3) ADX (expects OHLC columns: lowercase `high`, `low`, `close`)

```python
df = pl.DataFrame({
	'high': [10.0, 10.5, 11.0],
	'low': [9.5, 9.8, 10.2],
	'close': [10.0, 10.4, 10.8]
})
adx_df = plqt.adx(df, timeperiod=14)  # returns a DataFrame with `adx` column added
print(adx_df)
```

Backtesting (small example) 🧾

```python
import polars as pl
from polars_quant import Backtrade

# sample price data for one symbol
data = pl.DataFrame({
	'Date': ['2023-01-01','2023-01-02','2023-01-03','2023-01-04'],
	'AAPL': [100.0, 102.0, 101.0, 105.0]
})

# entry/exit signals must match the data shape (boolean)
entries = pl.DataFrame({
	'Date': data['Date'],
	'AAPL': [True, False, False, True]
})
exits = pl.DataFrame({
	'Date': data['Date'],
	'AAPL': [False, True, True, False]
})

bt = Backtrade.run(data, entries, exits, init_cash=100000.0, fee=0.001)
bt.summary()
```

Docs

- API reference: `docs/api.md` (short, code-driven)
- Examples: `docs/start/usage.md`
- Exact Python signatures: `python/polars_quant/polars_quant.pyi`

If you want a full parity table (function → pyi signature → Rust output columns), tell me and I'll generate it from the codebase.

