import polars as pl
import polars_quant as plqt
from timeit import timeit

df = pl.read_parquet("test.parquet")
df = df["datetime","close"]
entries = plqt.MA.run(df,[5,10]).cross(5,10)["close"]
exits = plqt.MA.run(df,[5,10]).cross(10,5)["close"]
bt = plqt.Backtrade.run(df,entries,exits)
bt.summary()
# print(timeit(lambda:plqt.Backtrade.run(df,entries,exits), number=1))

