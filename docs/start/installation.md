# Installation Guide

This guide will help you install and set up Polars-Quant for your development environment.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.9 or later
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB for installation

### Dependencies
Polars-Quant requires the following core dependencies:
- **Polars**: Fast DataFrame library for Python
- **PyArrow**: Apache Arrow integration for data processing
- **Rust**: For building the core engine (automatically handled)

## 🚀 Installation Methods
# Installation

This page describes simple, practical steps to install and test polars-quant.

Requirements

- Python 3.9+
- A supported OS (Windows, macOS, Linux)
- Rust toolchain only needed if building from source

Quick install (recommended)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install polars polars-quant
```

From source (development)

```powershell
git clone https://github.com/Firstastor/polars-quant.git
cd polars-quant
pip install -e .
```

Build notes (when developing native code)

- Install Rust via rustup if you plan to build the Rust extension locally.
- On Windows, ensure MSVC build tools are installed when building native crates.

Quick test

```python
import polars as pl
import polars_quant as plqt

print('polars-quant import OK')
df = pl.DataFrame({'close': [100, 101, 102, 103]})
res = plqt.ma(df, 2)
print('ma result type:', type(res))
```

Troubleshooting pointers

- If installation fails on `cargo` steps, run `rustup update` and ensure build tools are available.
- If importing fails in Python, verify you installed `polars` and that the virtualenv is active.

Support

- Open an issue on GitHub with a short reproduction and environment details.