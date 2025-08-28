import polars as pl 
import polars_quant as plqt
from pathlib import Path

data = []
data_entries = []
data_exits = []
i = 0
for file in Path("Data/").glob('*.parquet'):
    df = pl.read_parquet(file)
    if df.height < 100:
        continue
    df_date = df["date"]
    df_ma5 = plqt.ma(df["date", "close"], 5)["close_ma5"].alias(f"{file.stem}_ma5")
    df_ma10 = plqt.ma(df["date", "close"], 10)["close_ma10"].alias(f"{file.stem}_ma10")
    df_rsi = plqt.rsi(df)["rsi"].alias(f"{file.stem}_rsi")
    df = df["date","close"].rename({"close":file.stem})

    # 5日均线交10日均线
    entries = (df_ma5 > df_ma10) & (df_ma10.shift(1) > df_ma5.shift(1))
    exits = (df_ma5 < df_ma10) & (df_ma10.shift(1) < df_ma5.shift(1))

    # RSI突破上下限
    # entries = df_rsi < 30
    # exits = df_rsi > 70
    entries = pl.DataFrame([df_date, entries.shift(1)])
    exits = pl.DataFrame([df_date, exits.shift(1)])
    data.append(df)
    data_entries.append(entries)
    data_exits.append(exits)

def merge(df) -> pl.DataFrame:
    dfs = []
    for i in range(0, len(df), 100):
        batch = df[i:i+100]
        dfs.append(pl.concat(batch, how="align"))
    return pl.concat(dfs, how="align")

data = merge(data)
data_entries = merge(data_entries)
data_exits = merge(data_exits)
# print(plqt.Portfolio.run(data, data_entries, data_exits).summary())
print(plqt.Backtrade.run(data, data_entries, data_exits).summary())