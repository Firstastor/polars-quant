# polars-quant 🧮📊

> 基于 Rust + Polars 的量化研究与小型回测工具集，提供常用技术指标与简洁高效的向量化回测接口，适合快速原型和中小规模回测。

🔗 在线文档（中文）：https://firstastor.github.io/polars-quant/

---

## ✨ 特性

- 🧠 指标丰富：MA/EMA/KAMA、MACD、RSI、ADX、布林带等常见技术指标
- ⚡ 高性能：Rust 实现 + 并行加速，面向 Polars DataFrame 的向量化计算
- 📈 回测两种模式：
  - 单标的独立回测（每个标的使用独立资金）
  - 组合级回测（共享资金池，更贴近实盘）
- 🧰 数据便捷：内置 A 股历史数据和基础信息的获取/保存函数
- 🧩 轻量依赖：仅需 Polars 与 PyArrow，无 GPU 依赖

> Python API 以 `python/polars_quant/polars_quant.pyi` 为准。

---

## 🚀 安装

Windows PowerShell（推荐）

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install polars polars-quant
```

从源码安装（开发）

```powershell
git clone https://github.com/Firstastor/polars-quant.git
cd polars-quant
pip install -e .
# Windows 构建原生扩展需 Rust 工具链 + MSVC 构建工具
```

---

## 🧪 快速上手

1) 计算 3 日均线（MA）

```python
import polars as pl
import polars_quant as plqt

df = pl.DataFrame({'close': [100.0, 101.0, 102.0, 103.0, 104.0]})
# `ma` 返回若干 Series，可直接挂回 DataFrame
ma_list = plqt.ma(df, timeperiod=3)
res_df = df.with_columns(ma_list)
print(res_df)
```

2) MACD（返回 [dif, dea, macd] 三列）

```python
df = pl.DataFrame({'close': [100.0, 101.0, 102.5, 101.0, 103.0, 104.0]})
macd_cols = plqt.macd(df, fast=12, slow=26, signal=9)
res = df.with_columns(macd_cols)
print(res)
```

3) ADX（需要列名：`high`、`low`、`close`）

```python
df = pl.DataFrame({
    'high': [10.0, 10.5, 11.0],
    'low': [9.5, 9.8, 10.2],
    'close': [10.0, 10.4, 10.8]
})
adx_df = plqt.adx(df, timeperiod=14)  # 返回带 `adx` 列的 DataFrame
print(adx_df)
```

---

## 📈 回测示例

- Backtrade.run：每个标的独立使用一份初始资金，互不影响
- Portfolio.run：多标的共享资金池，更贴近组合实盘

```python
import polars as pl
from polars_quant import Backtrade

# 简单单标的示例
data = pl.DataFrame({
    'Date': ['2023-01-01','2023-01-02','2023-01-03','2023-01-04'],
    'AAPL': [100.0, 102.0, 101.0, 105.0]
})
entries = pl.DataFrame({'Date': data['Date'], 'AAPL': [True, False, False, True]})
exits   = pl.DataFrame({'Date': data['Date'], 'AAPL': [False, True, True, False]})

bt = Backtrade.run(data, entries, exits, init_cash=100_000.0, fee=0.001)
bt.summary()         # 控制台打印绩效摘要
print(bt.results())  # 资金曲线 / 现金
print(bt.trades())   # 交易明细
```

---

## 🔧 数据函数速览（A 股）

```python
import polars_quant as plqt

# 历史数据（来自新浪）
df = plqt.history('sz000001', scale=240, datalen=100)

# 保存到 Parquet
plqt.history_save('sz000001', datalen=500)

# 全市场基础信息
i = plqt.info()
plqt.info_save('stocks.parquet')
```

---

## 📚 文档导航

- 🏠 首页与概览：https://firstastor.github.io/polars-quant/
- ✨ 特性介绍：https://firstastor.github.io/polars-quant/start/features/
- 🛠 安装指南：https://firstastor.github.io/polars-quant/start/installation/
- 🚀 使用示例：https://firstastor.github.io/polars-quant/start/usage/
- 🔎 API 参考：https://firstastor.github.io/polars-quant/api/

---

## 📦 项目信息

- 许可证：MIT（见 `LICENSE`）
- 不包含 GPU/CUDA 加速，CPU 版 Rust + Polars 实现
- Python API 签名以 `python/polars_quant/polars_quant.pyi` 为准
- 仓库地址：https://github.com/Firstastor/polars-quant


