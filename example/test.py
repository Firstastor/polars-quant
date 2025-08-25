import polars as pl
import polars_quant as plqt
from timeit import timeit
import talib

# df_1 = pl.read_parquet("example/sz000001.parquet")[["date","close"]].rename({"close": "sz000001"})
# df_2 = pl.read_parquet("example/sz000002.parquet")[["date","close"]].rename({"close": "sz000002"})
# df_3 = pl.read_parquet("example/sz000004.parquet")["date","close"].rename({"close": "sz000004"})
# df = (df_1
#       .join(df_2, "date", "full",coalesce=True)
#       .join(df_3, "date", "full",coalesce=True)).sort("date").drop_nulls()

df = pl.read_parquet("example/sz000001.parquet")
plqt.stoch(df)
talib.STOCH(df['high'], df['low'],df['close'])
print(timeit(lambda:plqt.obv(df),number=1000))
print(timeit(lambda:pl.Series(talib.OBV(df['close'],df['volume'])),number=1000))
print(plqt.obv(df))
print(pl.Series(talib.OBV(df['close'],df['volume'])))
# print(pl.read_parquet("example/sz000001.parquet"))

# info = pl.read_parquet("example/info.parquet")["symbol"]
# for symbol in info:
#     plqt.history_save(symbol)
