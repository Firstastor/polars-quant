import polars as pl
import polars_quant as plqt
from timeit import timeit


# df_1 = pl.read_parquet("example/sz000001.parquet")[["date","close"]].rename({"close": "sz000001"})
# df_2 = pl.read_parquet("example/sz000002.parquet")[["date","close"]].rename({"close": "sz000002"})
# df_3 = pl.read_parquet("example/sz000004.parquet")["date","close"].rename({"close": "sz000004"})
# df = (df_1
#       .join(df_2, "date", "full",coalesce=True)
#       .join(df_3, "date", "full",coalesce=True)).sort("date").drop_nulls()

# print(timeit(lambda:plqt.wma(df),number=100))
# print(timeit(lambda:plqt.WMA.run(df, 20), number=100))
# print(plqt.mavp(df).head(40))
# print(plqt.TRIMA.run(df, 20).frame)

info = pl.read_parquet("example/info.parquet")["symbol"]
for symbol in info:
    plqt.history_save(symbol)
