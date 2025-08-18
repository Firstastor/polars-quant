import polars as pl
import polars_quant as plqt


df = pl.read_parquet("test.parquet")
bt = plqt.KAMA.run(df).frame
# entries = plqt.MA.run(df,[5,10]).cross(5,10)["close"].shift(1)
# exits = plqt.MA.run(df,[5,10]).cross(10,5)["close"].shift(1)
# bt = plqt.Backtrade.run(df,entries,exits).trades
print(bt)

