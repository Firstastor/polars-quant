import polars as pl
import polars_quant as pq
from timeit import timeit

df = pl.read_parquet("test.parquet")
df = df["datetime","close"]
entries = pq.MA.run(df,[5,10]).cross(5,10)["close"].rename({"close_ma5_cross_ma10":"close"})
exits = pq.MA.run(df,[5,10]).cross(10,5)["close"].rename({"close_ma10_cross_ma5":"close"})

df = pl.read_parquet("test.parquet")
df = df["datetime","open"]
entries_1 = pq.MA.run(df,[5,10]).cross(5,10)["open"].rename({"open_ma5_cross_ma10":"open"})
exits_1 = pq.MA.run(df,[5,10]).cross(10,5)["open"].rename({"open_ma10_cross_ma5":"open"})
df = pl.read_parquet("test.parquet")
df = df["datetime","close","open"]
entries = entries.join(entries_1, "datetime")
exits = exits.join(exits_1,"datetime")

df = pl.read_parquet("test.parquet")
df = df["datetime","high"]
entries_2 = pq.MA.run(df,[5,10]).cross(5,10)["high"].rename({"high_ma5_cross_ma10":"high"})
exits_2 = pq.MA.run(df,[5,10]).cross(10,5)["high"].rename({"high_ma10_cross_ma5":"high"})
df = pl.read_parquet("test.parquet")
df = df["datetime","close","open","high"]
entries = entries.join(entries_2, "datetime")
exits = exits.join(exits_2,"datetime")

df = pl.read_parquet("test.parquet")
df = df["datetime","low"]
entries_2 = pq.MA.run(df,[5,10]).cross(5,10)["low"].rename({"low_ma5_cross_ma10":"low"})
exits_2 = pq.MA.run(df,[5,10]).cross(10,5)["low"].rename({"low_ma10_cross_ma5":"low"})
df = pl.read_parquet("test.parquet")
df = df["datetime","close","open","high","low"]
entries = entries.join(entries_2, "datetime")
exits = exits.join(exits_2,"datetime")

print(pq.Backtrade.run(df,entries,exits,100000,0,0,1).summary())
