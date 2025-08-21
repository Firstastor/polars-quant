from genericpath import exists
import polars as pl
import polars_quant as plqt
from timeit import timeit


df_1 = pl.read_parquet("example/sz000001.parquet")[["date","close"]].rename({"close": "sz000001"})
# entries_1 = plqt.MA.run(df_1,[5,10]).cross(5,10)["close"].rename({'close_ma5_cross_ma10':'sz000001'})
# exits_1 = plqt.MA.run(df_1,[5,10]).cross(10,5)["close"].rename({'close_ma10_cross_ma5':'sz000001'})
# df_1 = df_1.rename({"close": "sz000001"})

df_2 = pl.read_parquet("example/sz000002.parquet")[["date","close"]].rename({"close": "sz000002"})
# entries_2 = plqt.MA.run(df_2,[5,10]).cross(5,10)["close"].rename({'close_ma5_cross_ma10':'sz000002'})
# exits_2 = plqt.MA.run(df_2,[5,10]).cross(10,5)["close"].rename({'close_ma10_cross_ma5':'sz000002'})
# df_2 = df_2.rename({"close": "sz000002"})

df_3 = pl.read_parquet("example/sz000004.parquet")["date","close"].rename({"close": "sz000004"})
# entries_3 = plqt.MA.run(df_3,[5,10]).cross(5,10)["close"].rename({'close_ma5_cross_ma10':'sz000004'})
# exits_3 = plqt.MA.run(df_3,[5,10]).cross(10,5)["close"].rename({'close_ma10_cross_ma5':'sz000004'})


df = (df_1.join(df_2, "date", "outer",coalesce=True).join(df_3, "date", "outer",coalesce=True)).sort("date")
# entries = (entries_1.join(entries_2, "date").join(entries_3, "date"))
# exits = (exits_1.join(exits_2, "date").join(exits_3, "date"))
entries = plqt.MA.run(df,[5,10]).cross(5,10)
exits = plqt.MA.run(df,[5,10]).cross(10,5)
# print(timeit(lambda:plqt.Backtrade.run(df,entries,exits), number=100))
# print(timeit(lambda:plqt.Backtrade.portfolio(df,entries,exits),number=100))


# print(plqt.Backtrade.portfolio(df,entries,exits).summary())
# print(plqt.Backtrade.run(df,entries,exits).summary())