import polars as pl
import polars_quant as plqt
from timeit import timeit

df = pl.read_parquet("test.parquet")
df = df["datetime","close"]
entries = plqt.MA.run(df,[5,10]).cross(5,10)["close"].rename({"close_ma5_cross_ma10":"close"})
exits = plqt.MA.run(df,[5,10]).cross(10,5)["close"].rename({"close_ma10_cross_ma5":"close"})

print(timeit(lambda:plqt.Backtrade.run(df,entries,exits), number=3))

