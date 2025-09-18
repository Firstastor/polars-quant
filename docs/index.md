# Polars-Quant

基于 Rust + Polars 的量化研究与小型回测工具集。提供高性能技术指标（TA）与简洁的向量化回测接口，专注“代码即研究”。

- 指标：MA/EMA/KAMA、MACD、RSI、ADX、布林带等常见技术指标
- 回测：支持单标的独立回测与组合级共享资金回测
- 数据：便捷获取与保存 A 股历史数据与基础信息
- 生态：Python API，底层 Rust 并行，面向 Polars DataFrame

快速开始：

1) 安装

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install polars polars-quant
```

2) 一个最小示例

```python
import polars as pl
import polars_quant as plqt

# 计算 3 日均线
df = pl.DataFrame({'close': [100.0, 101.0, 102.0, 103.0, 104.0]})
ma_series = plqt.ma(df, timeperiod=3)
res = df.with_columns(ma_series)
print(res)
```

继续阅读：
- 特性概览：见“开始/特性”
- 安装说明：见“开始/安装”
- 使用示例：见“开始/使用”
- API 参考：见“API”
