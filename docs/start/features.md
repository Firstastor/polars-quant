# 特性

- 纯向量化，贴合 Polars 风格：以 DataFrame 为核心，返回 Series 或带指标列的 DataFrame。
- 高性能：底层 Rust 实现 + 多线程并行，适合中小规模回测与策略研究。
- 简明 API：以 `python/polars_quant/polars_quant.pyi` 为准，函数签名清晰统一。
- 两种回测形态：
  - Backtrade.run：各标的独立回测，独立资金曲线。
  - Backtrade.portfolio / Portfolio.run：组合级回测，共享资金池。
- 贴心数据函数：抓取并保存 A 股历史数据与基础信息，快速落地研究。
- 轻量依赖：仅依赖 Polars 与 PyArrow，无 GPU 需求。

适用场景

- 技术指标研究、信号原型验证
- 单标的或多标的简易回测、快节奏迭代
- 与 Polars/Python 科研生态无缝衔接

不是什么

- 这不是全功能交易平台；不含订单撮合、组合优化、风控引擎等重量模块。
- 更偏向“研究驱动”，而非“生产交易系统”。
