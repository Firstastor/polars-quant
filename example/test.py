import polars as pl
import polars_quant as plqt

df = plqt.history("sz000001")
print(plqt.ma(df))