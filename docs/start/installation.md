# 安装

推荐环境：Python 3.9+，Windows/Linux/macOS，已安装 Rust 工具链可编译开发版。

## 通过 PyPI 安装

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install polars polars-quant
```

## 从源码安装（开发）

```powershell
git clone https://github.com/Firstastor/polars-quant.git
cd polars-quant
pip install -e .
# Windows 构建原生扩展需 Rust + MSVC 构建工具
```

## 快速自测

```python
import polars as pl
import polars_quant as plqt

print(pl.__version__)
# 简单调用，若执行正常说明安装成功
_ = plqt.sma(pl.DataFrame({'close':[1,2,3,4]}), timeperiod=2)
```
