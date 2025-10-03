import polars as pl
import polars_quant as pq

df = pl.read_parquet("data/sh.000001.parquet")

df = df["date", "close"].rename({"date": "Date"})

ma5 = pq.ma(df["close"], 5).alias("ma5")
ma10 = pq.ma(df["close"], 10).alias("ma10")

entries = (ma5 > ma10) & (ma5.shift(1) <= ma10.shift(1))
exits = (ma5 < ma10) & (ma5.shift(1) >= ma10.shift(1))

entries = pl.DataFrame({"Date:": df["Date"], "close": entries})
exits = pl.DataFrame({"Date:": df["Date"], "close": exits})

bt = pq.Backtrade.run(df, entries, exits)
bt.summary()
