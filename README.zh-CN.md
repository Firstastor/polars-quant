# polars_quant

polars_quant 是一个由 Rust 支持的 Python 原生扩展，本仓库已实现并导出的主要接口为：

- polars_quant.history(stock_code: str, scale: int = 240, datalen: int = 3650, timeout: int = 10)
  - 从远端获取 OHLCV 历史数据。返回记录列表（每条记录包含 day, open, close, high, low, volume）或 None。

- 类 polars_quant.Backtrade
  - Backtrade.run(data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0)
  - Backtrade.portfolio(data, entries, exits, init_cash=100000.0, fee=0.0, slip=0.0, size=1.0)
  - 实例提供的属性/方法：results、trades、summary()、speed

快速使用示例

1) 抓取历史数据

```python
import polars as pl
import polars_quant

items = polars_quant.history("sh600519", scale=240, datalen=365, timeout=10)
if items is None:
    print("没有数据")
else:
    df = pl.DataFrame(items)
    print(df.head())
```

2) 单标的回测示例

```python
import polars as pl
from polars_quant import Backtrade

data = pl.DataFrame({
    "date": ["2024-01-01","2024-01-02","2024-01-03"],
    "SYM": [100.0, 101.5, 99.0],
})
entries = pl.DataFrame({"date": data["date"], "SYM": [False, True, False]})
exits = pl.DataFrame({"date": data["date"], "SYM": [False, False, True]})

bt = Backtrade.run(data, entries, exits, init_cash=100000.0, fee=0.0005)
print(bt.summary())
if getattr(bt, "results", None) is not None:
    print(bt.results.head())
```

注意
- 保持 data、entries、exits 三者列对齐：第 0 列为日期，后续列为每个标的（每列一个标的）。
- entries / exits 列可以使用布尔或整数标记。
- GitHub 不会自动根据浏览器语言切换 README；使用并维护两个语言文件是一种简单可维护的做法。