"""
polars-quant 类型提示文件
提供所有技术分析函数的类型注释和使用示例

主要功能模块：
1. 数据处理 (Data Processing) - 收益率计算、数据加载
2. 技术指标 (Technical Analysis) - 各种技术分析指标函数
3. 回测引擎 (Backtesting) - 高性能多线程回测系统
4. 选股器 (Stock Selector) - 多维度股票筛选
5. 因子分析 (Factor Analysis) - 因子计算与评估

安装使用：
    pip install polars-quant

基本用法：
    import polars as pl
    import polars_quant as pq
    
    # 计算技术指标
    prices = pl.Series([1, 2, 3, 4, 5])
    sma_result = pq.sma(prices, 3)
    
    # 计算收益率
    df = pl.DataFrame({"date": [...], "close": [...]})
    df = pq.returns(df, price_col="close", period=1, method="simple")
    
    # 加载数据
    data = pq.load("./data", file_type=["parquet", "csv"])
    
    # 运行回测
    result = pq.Backtest.run(data, entries, exits)
    result.summary()
"""

import polars as pl
from typing import Tuple, Optional, List, Literal

# ====================================================================
# 数据处理模块 (Data Processing)
# ====================================================================

def returns(
    df: pl.DataFrame,
    price_col: str = "close",
    period: int = 1,
    method: Literal["simple", "log"] = "simple",
    return_col: str = "return"
) -> pl.DataFrame:
    """
    计算收益率
    
    Args:
        df: 输入DataFrame
        price_col: 价格列名，默认 "close"
        period: 计算周期，默认 1
        method: 计算方法
            - "simple": 简单收益率 (price[t] - price[t-period]) / price[t-period]
            - "log": 对数收益率 ln(price[t] / price[t-period])
        return_col: 返回列名，默认 "return"
    
    Returns:
        添加了收益率列的 DataFrame
    
    Examples:
        >>> import polars as pl
        >>> import polars_quant as pq
        >>> 
        >>> # 计算简单收益率
        >>> df = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        ...     "close": [100.0, 102.0, 101.0]
        ... })
        >>> result = pq.returns(df, price_col="close", period=1, method="simple")
        >>> # 结果包含原始列 + return列: [None, 0.02, -0.0098]
        >>> 
        >>> # 计算对数收益率
        >>> result = pq.returns(df, method="log", return_col="log_return")
    """
    ...

def load(
    folder: str,
    file_type: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    has_header: bool = True
) -> pl.DataFrame:
    """
    从文件夹批量加载股票数据
    
    Args:
        folder: 数据文件夹路径
        file_type: 文件类型列表，支持 ["parquet", "csv", "xlsx", "xls", "json", "feather", "ipc"]
                  None 表示支持所有格式
        prefix: 文件名前缀过滤（可选）
        suffix: 文件名后缀过滤（可选）
        has_header: CSV/Excel 文件是否包含表头，默认 True
    
    Returns:
        合并后的 DataFrame，列格式: date, {symbol}_open, {symbol}_high, {symbol}_low, {symbol}_close, {symbol}_volume
    
    Examples:
        >>> import polars_quant as pq
        >>> 
        >>> # 加载所有 parquet 文件
        >>> data = pq.load("./stock_data", file_type=["parquet"])
        >>> 
        >>> # 加载特定前缀的 CSV 文件
        >>> data = pq.load("./stock_data", file_type=["csv"], prefix="SH")
        >>> 
        >>> # 加载所有支持格式的文件
        >>> data = pq.load("./stock_data")
    
    Notes:
        - 每个文件应包含 date, open, high, low, close, volume 列
        - 文件名（去除扩展名）将作为股票代码
        - 加载后列名格式为: {股票代码}_{列名}，如 AAPL_close, TSLA_volume
        - 所有数据按 date 列进行外连接 (Full Join)
        - 结果按日期排序
    """
    ...

# ====================================================================
# 回测引擎 (Backtesting Engine) - 独立资金池回测系统
# ====================================================================

class Backtest:
    """
    独立资金池量化回测引擎
    
    特点:
    - 每只股票使用独立资金池进行回测
    - 支持自动多线程并行处理
    - 真实模拟交易规则（佣金、最低佣金、滑点、整百股交易）
    - 完整的交易记录和绩效统计
    - 支持单股票深度分析和全局汇总
    
    Examples:
        基本回测示例：
        >>> import polars as pl
        >>> import polars_quant as pq
        >>> 
        >>> # 准备价格数据（第一列为日期，其余列为各股票价格）
        >>> prices = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        ...     "AAPL": [150.0, 152.0, 148.0],
        ...     "GOOGL": [2800.0, 2850.0, 2820.0]
        ... })
        >>> 
        >>> # 准备买入信号（True表示买入信号）
        >>> buy_signals = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        ...     "AAPL": [True, False, False],
        ...     "GOOGL": [False, True, False]
        ... })
        >>> 
        >>> # 准备卖出信号（True表示卖出信号）
        >>> sell_signals = pl.DataFrame({
        ...     "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        ...     "AAPL": [False, False, True],
        ...     "GOOGL": [False, False, True]
        ... })
        >>> 
        >>> # 创建回测实例
        >>> bt = pq.Backtest(
        ...     prices=prices,
        ...     buy_signals=buy_signals,
        ...     sell_signals=sell_signals,
        ...     initial_capital=100000.0,  # 初始资金10万
        ...     commission_rate=0.0003,    # 佣金费率万三
        ...     min_commission=5.0,        # 最低佣金5元
        ...     slippage=0.0               # 滑点0%
        ... )
        >>> 
        >>> # 运行回测
        >>> bt.run()
        >>> 
        >>> # 查看全局统计摘要
        >>> bt.summary()
        >>> 
        >>> # 获取所有股票的每日资金记录
        >>> daily_records = bt.get_daily_records()
        >>> 
        >>> # 获取所有交易记录
        >>> positions = bt.get_position_records()
        >>> 
        >>> # 查看单只股票的每日资金记录
        >>> aapl_daily = bt.get_stock_daily("AAPL")
        >>> 
        >>> # 查看单只股票的交易记录
        >>> aapl_positions = bt.get_stock_positions("AAPL")
        >>> 
        >>> # 查看单只股票的统计摘要
        >>> aapl_summary = bt.get_stock_summary("AAPL")
        >>> print(aapl_summary)
        
        高级配置示例（含滑点和佣金）：
        >>> bt = pq.Backtest(
        ...     prices=prices,
        ...     buy_signals=buy_signals,
        ...     sell_signals=sell_signals,
        ...     initial_capital=1000000.0,   # 初始资金100万
        ...     commission_rate=0.0003,      # 佣金费率万三
        ...     min_commission=5.0,          # 最低佣金5元
        ...     slippage=0.001               # 滑点0.1%
        ... )
        >>> bt.run()
        >>> bt.summary()
    """
    
    def __init__(
        self,
        prices: pl.DataFrame,
        buy_signals: pl.DataFrame,
        sell_signals: pl.DataFrame,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        slippage: float = 0.0,
        position_size: float = 1.0,
        leverage: float = 1.0,
        margin_call_threshold: float = 0.3,
        interest_rate: float = 0.06,
        benchmark: Optional[pl.DataFrame] = None
    ) -> None:
        """
        创建回测实例
        
        Args:
            prices: 价格数据DataFrame，第一列为日期，其余列为各股票价格
                格式: {"date": [...], "STOCK1": [...], "STOCK2": [...]}
            buy_signals: 买入信号DataFrame，第一列为日期，其余列为布尔值
                格式: {"date": [...], "STOCK1": [True/False, ...], "STOCK2": [True/False, ...]}
            sell_signals: 卖出信号DataFrame，第一列为日期，其余列为布尔值
                格式: {"date": [...], "STOCK1": [True/False, ...], "STOCK2": [True/False, ...]}
            initial_capital: 初始资金，默认100000.0（每只股票独立使用该资金）
            commission_rate: 佣金费率，默认0.0003（万三）
            min_commission: 最低佣金，默认5.0元
            slippage: 滑点，默认0.0（0.001表示0.1%）
            position_size: 仓位大小，默认1.0（满仓），取值范围(0.0, 1.0]
                例如：0.5表示每次使用50%的可用资金买入
            leverage: 杠杆倍数，默认1.0（不使用杠杆），取值范围[1.0, ∞)
                例如：2.0表示使用2倍杠杆，可买入2倍本金的股票
                注意：使用杠杆会产生融资利息成本
            margin_call_threshold: 保证金维持率阈值，默认0.3（30%）
                当保证金维持率低于此值时触发强制平仓
                保证金维持率 = (总资产 - 负债) / 总资产
            interest_rate: 融资年化利率，默认0.06（6%）
                使用杠杆时的借款成本，每日计息
            benchmark: 基准指数数据DataFrame（可选），必须恰好有2列（日期和价格）
                格式: {"date": [...], "benchmark": [...]}
                所有股票将使用此基准进行对比分析
            
        Raises:
            ValueError: 当DataFrame列数不一致或prices列数少于2时
            ValueError: 当仓位大小不在(0.0, 1.0]范围内时
            ValueError: 当杠杆倍数 < 1.0时
            ValueError: 当保证金维持率不在[0.0, 1.0)范围内时
        
        Examples:
            基本回测（无杠杆）：
            >>> bt = Backtest(
            ...     prices=prices_df,
            ...     buy_signals=buy_df,
            ...     sell_signals=sell_df,
            ...     initial_capital=100000.0
            ... )
            
            使用2倍杠杆回测：
            >>> bt = Backtest(
            ...     prices=prices_df,
            ...     buy_signals=buy_df,
            ...     sell_signals=sell_df,
            ...     initial_capital=100000.0,
            ...     leverage=2.0,              # 2倍杠杆
            ...     margin_call_threshold=0.3, # 保证金维持率30%以下强平
            ...     interest_rate=0.06         # 年化6%融资利率
            ... )

            ValueError: 当基准数据不是恰好2列时
            
        Examples:
            >>> # 创建带基准对比和半仓策略的回测
            >>> benchmark = pl.DataFrame({
            ...     "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            ...     "SH000001": [3100.0, 3120.0, 3090.0]  # 上证指数
            ... })
            >>> bt = pq.Backtest(
            ...     prices=prices,
            ...     buy_signals=buy_signals,
            ...     sell_signals=sell_signals,
            ...     initial_capital=100000.0,
            ...     position_size=0.5,  # 每次使用50%资金
            ...     benchmark=benchmark
            ... )
        """
        ...
    
    def run(self) -> None:
        """
        运行回测
        
        执行回测计算，使用多线程并行处理所有股票。
        每只股票使用独立的资金池进行回测。
        
        Raises:
            ValueError: 当date列不存在或格式不正确时
        """
        ...
    
    def get_daily_records(self) -> pl.DataFrame:
        """
        获取所有股票的每日资金记录
        
        Returns:
            DataFrame，包含以下列：
            - symbol: 股票代码
            - date: 日期
            - cash: 当日现金
            - stock_value: 当日持仓市值
            - total_value: 当日总资产（现金+持仓市值）
            
        Raises:
            ValueError: 当未运行run()方法时
        """
        ...
    
    def get_position_records(self) -> pl.DataFrame:
        """
        获取所有交易记录
        
        Returns:
            DataFrame，包含以下列：
            - symbol: 股票代码
            - entry_date: 入场日期
            - entry_price: 入场价格
            - quantity: 交易数量
            - exit_date: 出场日期
            - exit_price: 出场价格
            - pnl: 盈亏金额
            - pnl_pct: 盈亏百分比
            - holding_days: 持仓天数
            
        Raises:
            ValueError: 当没有交易记录时
        """
        ...
    
    def get_performance_metrics(self) -> pl.DataFrame:
        """
        获取每日绩效指标（包括每日盈亏、累计收益、与基准对比）
        
        Returns:
            DataFrame，包含以下列：
            - date: 日期
            - portfolio_value: 组合总市值
            - daily_pnl: 当日盈亏金额
            - daily_return_pct: 当日收益率（%）
            - cumulative_pnl: 累计盈亏金额
            - cumulative_return_pct: 累计收益率（%）
            
            如果提供了基准数据，还包括：
            - benchmark_return_pct: 基准当日收益率（%）
            - alpha_pct: 超额收益率（策略收益 - 基准收益）（%）
            - relative_return_pct: 相对基准的累计收益率（%）
            - beta: Beta系数（策略相对基准的系统性风险敞口）
              * Beta < 1: 策略波动小于基准，防御性
              * Beta = 1: 策略波动与基准一致
              * Beta > 1: 策略波动大于基准，进攻性
            
        Raises:
            ValueError: 当未运行run()方法时
            
        Examples:
            >>> bt.run()
            >>> metrics = bt.get_performance_metrics()
            >>> print(metrics.head())
            >>> 
            >>> # 查看Beta值
            >>> if "beta" in metrics.columns:
            ...     beta = metrics["beta"][0]
            ...     print(f"策略Beta系数: {beta:.4f}")
            >>> 
            >>> # 绘制累计收益对比图
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(metrics["date"], metrics["cumulative_return_pct"], label="策略")
            >>> if "benchmark_return_pct" in metrics.columns:
            ...     # 计算基准累计收益
            ...     benchmark_cum = metrics.select([
            ...         pl.col("benchmark_return_pct").cum_sum()
            ...     ])
            ...     plt.plot(metrics["date"], benchmark_cum, label="基准")
            >>> plt.legend()
            >>> plt.show()
            >>> 
            >>> # 分析每日盈亏分布
            >>> print(metrics.select([
            ...     pl.col("daily_pnl").mean().alias("平均日盈亏"),
            ...     pl.col("daily_pnl").std().alias("日盈亏标准差"),
            ...     pl.col("daily_pnl").max().alias("最大日盈利"),
            ...     pl.col("daily_pnl").min().alias("最大日亏损")
            ... ]))
        """
        ...
    
    def get_stock_performance(self, symbol: str) -> pl.DataFrame:
        """
        获取单只股票的每日绩效指标
        
        Args:
            symbol: 股票代码
            
        Returns:
            DataFrame，包含以下列：
            - symbol: 股票代码
            - date: 日期
            - stock_value: 股票总资产（现金+持仓市值）
            - daily_pnl: 当日盈亏金额
            - daily_return_pct: 当日收益率（%）
            - cumulative_pnl: 累计盈亏金额
            - cumulative_return_pct: 累计收益率（%）
            
            如果提供了基准数据，还包括：
            - benchmark_return_pct: 基准当日收益率（%）
            - alpha_pct: 超额收益率（策略收益 - 基准收益）（%）
            - relative_return_pct: 相对基准的累计收益率（%）
            
        Raises:
            ValueError: 当未运行run()方法时或股票代码不存在时
            
        Examples:
            >>> bt.run()
            >>> # 获取AAPL的每日绩效
            >>> aapl_perf = bt.get_stock_performance("AAPL")
            >>> print(aapl_perf)
            >>> 
            >>> # 分析AAPL的盈亏统计
            >>> aapl_stats = aapl_perf.select([
            ...     pl.col("daily_pnl").mean().alias("平均日盈亏"),
            ...     pl.col("daily_pnl").max().alias("最大日盈利"),
            ...     pl.col("daily_pnl").min().alias("最大日亏损"),
            ...     pl.col("cumulative_return_pct").tail(1).alias("总收益率")
            ... ])
            >>> print(aapl_stats)
            >>> 
            >>> # 对比多只股票的表现
            >>> aapl_perf = bt.get_stock_performance("AAPL")
            >>> googl_perf = bt.get_stock_performance("GOOGL")
            >>> comparison = pl.DataFrame({
            ...     "AAPL收益": aapl_perf["cumulative_return_pct"],
            ...     "GOOGL收益": googl_perf["cumulative_return_pct"]
            ... })
            >>> print(comparison.tail())
        """
        ...
    
    def get_stock_daily(self, symbol: str) -> pl.DataFrame:
        """
        获取单只股票的每日资金记录
        
        Args:
            symbol: 股票代码
            
        Returns:
            DataFrame，包含以下列：
            - symbol: 股票代码
            - date: 日期
            - cash: 当日现金
            - stock_value: 当日持仓市值
            - total_value: 当日总资产
            
        Raises:
            ValueError: 当未运行run()方法时
        """
        ...
    
    def get_stock_positions(self, symbol: str) -> pl.DataFrame:
        """
        获取单只股票的交易记录
        
        Args:
            symbol: 股票代码
            
        Returns:
            DataFrame，包含以下列：
            - symbol: 股票代码
            - entry_date: 入场日期
            - entry_price: 入场价格
            - quantity: 交易数量
            - exit_date: 出场日期
            - exit_price: 出场价格
            - pnl: 盈亏金额
            - pnl_pct: 盈亏百分比
            - holding_days: 持仓天数
            
        Raises:
            ValueError: 当没有交易记录时
        """
        ...
    
    def get_stock_summary(self, symbol: str) -> str:
        """
        获取单只股票的统计摘要
        
        Args:
            symbol: 股票代码
            
        Returns:
            格式化的统计摘要字符串，包含：
            - 【绩效总览】初始资金、最终资金、总收益率、最大回撤、夏普比率
            - 【交易统计】交易次数、盈利/亏损交易、胜率
            - 【盈利分析】总盈利、平均盈利、最大单笔盈利
            - 【亏损分析】总亏损、平均亏损、最大单笔亏损
            - 【基准对比】(仅当提供了benchmark时)
              * Alpha、Beta、跑赢基准比例
            
        Raises:
            ValueError: 当未运行run()方法时
        
        Example:
            >>> bt = Backtest(df, initial_capital=100000, benchmark=benchmark_df)
            >>> bt.run()
            >>> print(bt.get_stock_summary("000001"))
            # 输出包含绩效总览和基准对比分析
        """
        ...
    
    def summary(self) -> None:
        """
        打印全局统计摘要报告
        
        包含内容：
        - 【基本信息】回测期间、初始资金、最终资金、总盈亏
        - 【收益指标】总收益率、年化收益率、日均收益率
        - 【风险指标】最大回撤、最大回撤持续、日波动率、年化波动率
        - 【风险调整收益】夏普比率、索提诺比率、卡尔马比率
        - 【交易统计】总交易次数、盈利/亏损交易、胜率、盈亏比
        - 【盈利分析】总盈利、平均盈利、最大单笔盈利、平均盈利持仓
        - 【亏损分析】总亏损、平均亏损、最大单笔亏损、平均亏损持仓
        - 【持仓分析】平均持仓周期、总持仓天数、最大连续盈利/亏损
        - 【交易成本】总交易额、总手续费、手续费占比
        - 【资金使用】平均单笔交易额、资金使用率
        - 【日收益分析】正收益天数、负收益天数、日胜率
        - 【股票维度】交易股票数量、表现最好/最差的股票
        - 【基准对比】(仅当提供了benchmark时)
          * 收益对比：策略 vs 基准累计收益率、超额收益
          * 风险分析：Alpha、Beta、IR
          * 相对表现：跑赢基准天数比例
        
        Raises:
            ValueError: 当未运行run()方法时
        
        Example:
            >>> bt = Backtest(df, initial_capital=100000, benchmark=benchmark_df)
            >>> bt.run()
            >>> bt.summary()
            # 输出包含基准对比分析的完整统计摘要
        """
        ...

# ====================================================================
# 股票选择器 (Stock Selector) - 链式筛选系统
# ====================================================================

class Selector:
    """
    股票选择器 - 支持链式调用的股票筛选工具
    
    功能特点:
    - 链式调用支持多条件组合筛选
    - 单一 filter() 方法支持 30+ 筛选参数
    - 内置技术指标计算（MA、RSI、MACD、KDJ 等）
    - 支持排序和取 TopN
    
    Examples:
        基本用法：
        >>> import polars as pl
        >>> import polars_quant as pq
        >>> 
        >>> # 使用 load 函数加载数据
        >>> data = pq.load("data/stocks", file_type=["parquet"])
        >>> 
        >>> # 创建 Selector
        >>> selector = pq.Selector(data)
        >>> 
        >>> # 链式筛选示例
        >>> result = (
        ...     selector
        ...     .filter(price_min=10, price_max=100)           # 价格区间
        ...     .filter(volume_min=1000000)                     # 成交量筛选
        ...     .filter(ma_above=20)                            # 价格在20日均线之上
        ...     .filter(rsi_min=30, rsi_max=70)                # RSI 超卖超买
        ...     .filter(volatility_min=15, volatility_max=40)  # 波动率区间
        ...     .sort(by="return_5d", ascending=False, top_n=10) # 按5日收益率排序取前10
        ...     .result()                                       # 获取股票代码列表
        ... )
        >>> print(result)  # ['SH600000', 'SH600001', ...]
        >>> 
        >>> # 查看详细信息
        >>> info_df = selector.info()
        >>> print(info_df)
        
        技术指标筛选：
        >>> # MACD 金叉
        >>> result = selector.filter(macd="golden_cross").result()
        >>> 
        >>> # KDJ 超卖
        >>> result = selector.filter(kdj="oversold").result()
        >>> 
        >>> # 涨停板筛选
        >>> result = selector.filter(limit_type="limit_up").result()
        >>> 
        >>> # 放量突破
        >>> result = selector.filter(
        ...     volume_change="volume_surge",
        ...     volume_multiplier=2.0,
        ...     breakout="breakout_high"
        ... ).result()
    """
    
    def __init__(self, ohlcv_data: pl.DataFrame) -> None:
        """
        创建股票选择器实例
        
        Args:
            ohlcv_data: OHLCV 数据的 DataFrame
                要求包含列: date, {symbol}_open, {symbol}_high, {symbol}_low, 
                           {symbol}_close, {symbol}_volume
        
        Raises:
            ValueError: OHLCV 数据至少需要 2 列
        """
        ...
    
    def filter(
        self,
        # 价格筛选
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        
        # 成交量筛选
        volume_min: Optional[float] = None,
        volume_avg_days: Optional[int] = None,
        
        # 收益率筛选
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        return_period: Optional[int] = None,
        
        # 波动率筛选
        volatility_min: Optional[float] = None,
        volatility_max: Optional[float] = None,
        volatility_period: Optional[int] = None,
        
        # 均线筛选
        ma_above: Optional[int] = None,
        ma_below: Optional[int] = None,
        
        # RSI 筛选
        rsi_min: Optional[float] = None,
        rsi_max: Optional[float] = None,
        rsi_period: Optional[int] = None,
        
        # MACD 筛选
        macd: Optional[str] = None,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        
        # KDJ 筛选
        kdj: Optional[str] = None,
        kdj_period: int = 9,
        
        # 涨跌停筛选
        limit_type: Optional[str] = None,
        limit_threshold: float = 9.9,
        
        # 成交量变化筛选
        volume_change: Optional[str] = None,
        volume_multiplier: float = 2.0,
        volume_change_days: int = 5,
        
        # 连续上涨/下跌筛选
        consecutive: Optional[str] = None,
        consecutive_days: int = 3,
        
        # 突破筛选
        breakout: Optional[str] = None,
        breakout_period: int = 20
    ) -> 'Selector':
        """
        筛选股票（支持链式调用）
        
        Args:
            价格筛选:
                price_min: 最低价格
                price_max: 最高价格
            
            成交量筛选:
                volume_min: 最小成交量
                volume_avg_days: 平均成交量天数（None 表示使用最新成交量）
            
            收益率筛选:
                return_min: 最小收益率（百分比）
                return_max: 最大收益率（百分比）
                return_period: 收益率周期（默认 1 天）
            
            波动率筛选:
                volatility_min: 最小年化波动率（百分比）
                volatility_max: 最大年化波动率（百分比）
                volatility_period: 波动率计算周期（默认 20 天）
            
            均线筛选:
                ma_above: 价格在 N 日均线之上
                ma_below: 价格在 N 日均线之下
            
            RSI 筛选:
                rsi_min: 最小 RSI 值
                rsi_max: 最大 RSI 值
                rsi_period: RSI 周期（默认 14）
            
            MACD 筛选:
                macd: MACD 条件 ("golden_cross"=金叉, "death_cross"=死叉, 
                      "above_zero"=在零轴上方, "below_zero"=在零轴下方)
                macd_fast: 快线周期（默认 12）
                macd_slow: 慢线周期（默认 26）
                macd_signal: 信号线周期（默认 9）
            
            KDJ 筛选:
                kdj: KDJ 条件 ("golden_cross"=金叉, "death_cross"=死叉,
                     "oversold"=超卖, "overbought"=超买)
                kdj_period: KDJ 周期（默认 9）
            
            涨跌停筛选:
                limit_type: 涨跌停类型 ("limit_up"=涨停, "limit_down"=跌停,
                           "near_limit_up"=接近涨停, "near_limit_down"=接近跌停)
                limit_threshold: 涨跌停阈值（默认 9.9%）
            
            成交量变化筛选:
                volume_change: 成交量变化类型 ("volume_surge"=放量, "volume_shrink"=缩量)
                volume_multiplier: 成交量倍数（默认 2.0）
                volume_change_days: 平均成交量天数（默认 5）
            
            连续上涨/下跌筛选:
                consecutive: 连续类型 ("consecutive_up"=连续上涨, "consecutive_down"=连续下跌)
                consecutive_days: 连续天数（默认 3）
            
            突破筛选:
                breakout: 突破类型 ("breakout_high"=突破新高, "breakdown_low"=跌破新低)
                breakout_period: 突破周期（默认 20 天）
        
        Returns:
            self (支持链式调用)
        
        Examples:
            >>> # 价格和成交量筛选
            >>> selector.filter(price_min=10, price_max=50, volume_min=1000000)
            >>> 
            >>> # 技术指标筛选
            >>> selector.filter(
            ...     ma_above=20,
            ...     rsi_min=30, rsi_max=70,
            ...     macd="golden_cross"
            ... )
            >>> 
            >>> # 链式调用
            >>> selector.filter(price_min=10).filter(volume_min=1000000).filter(rsi_min=30)
        """
        ...
    
    def result(self) -> List[str]:
        """
        获取筛选结果（股票代码列表）
        
        Returns:
            股票代码列表
        
        Example:
            >>> symbols = selector.filter(price_min=10).result()
            >>> print(symbols)  # ['SH600000', 'SH600001', ...]
        """
        ...
    
    def reset(self) -> 'Selector':
        """
        重置筛选条件
        
        Returns:
            self (支持链式调用)
        
        Example:
            >>> selector.filter(price_min=10).reset().filter(volume_min=1000000)
        """
        ...
    
    def sort(
        self,
        by: str,
        ascending: bool = False,
        top_n: Optional[int] = None
    ) -> 'Selector':
        """
        对股票进行排序并可选择取 TopN
        
        注意: 如果 top_n 大于可用股票数量，将返回所有可用股票（不会报错）
        
        Args:
            by: 排序指标，可选值:
                - "price": 价格
                - "return_1d": 1日收益率
                - "return_5d": 5日收益率
                - "return_20d": 20日收益率
                - "volume": 成交量
                - "volatility": 波动率
            ascending: 是否升序（默认 False，降序）
            top_n: 取前 N 个股票（可选）。如果 N 大于股票总数，返回所有股票
        
        Returns:
            self (支持链式调用)
        
        Examples:
            >>> # 按5日收益率降序排序，取前10（如果只有5只股票，返回全部5只）
            >>> selector.sort(by="return_5d", ascending=False, top_n=10)
            >>> 
            >>> # 按波动率升序排序
            >>> selector.sort(by="volatility", ascending=True)
        """
        ...
    
    def info(self) -> pl.DataFrame:
        """
        获取股票详细信息
        
        Returns:
            包含以下列的 DataFrame:
            - symbol: 股票代码
            - price: 最新收盘价
            - open: 最新开盘价
            - high: 最新最高价
            - low: 最新最低价
            - volume: 最新成交量
            - return_1d: 1日收益率（%）
            - return_5d: 5日收益率（%）
            - return_20d: 20日收益率（%）
            - volatility: 年化波动率（%，基于20日数据）
            - ma_5: 5日均线
            - ma_10: 10日均线
            - ma_20: 20日均线
            - volume_ratio: 量比（当前成交量 / 5日平均成交量）
            - amplitude: 振幅（%，当日最高最低价差 / 收盘价）
        
        Example:
            >>> df = selector.filter(price_min=10).info()
            >>> print(df)
            # 输出15列详细信息
        """
        ...

# ====================================================================
# 重叠研究指标 (Overlap Studies) - 移动平均线及相关指标
# ====================================================================

def bband(series: pl.Series, period: int, std_dev: float) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """
    布林带 (Bollinger Bands)
    
    Args:
        series: 价格序列
        period: 计算周期
        std_dev: 标准差倍数
    
    Returns:
        Tuple[上轨, 中轨, 下轨]
    
    Example:
        >>> import polars as pl, polars_quant as pq
        >>> prices = pl.Series([20, 21, 19, 22, 20, 21, 23])
        >>> upper, middle, lower = pq.bband(prices, 5, 2.0)
    """
    ...

def sma(series: pl.Series, period: int) -> pl.Series:
    """
    简单移动平均线 (Simple Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        SMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.sma(prices, 3)
    """
    ...

def ema(series: pl.Series, period: int) -> pl.Series:
    """
    指数移动平均线 (Exponential Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        EMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.ema(prices, 3)
    """
    ...

def wma(series: pl.Series, period: int) -> pl.Series:
    """
    加权移动平均线 (Weighted Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        WMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.wma(prices, 3)
    """
    ...

def dema(series: pl.Series, period: int) -> pl.Series:
    """
    双指数移动平均线 (Double Exponential Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        DEMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.dema(prices, 5)
    """
    ...

def tema(series: pl.Series, period: int) -> pl.Series:
    """
    三指数移动平均线 (Triple Exponential Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        TEMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.tema(prices, 5)
    """
    ...

def trima(series: pl.Series, period: int) -> pl.Series:
    """
    三角移动平均线 (Triangular Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        TRIMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.trima(prices, 5)
    """
    ...

def kama(series: pl.Series, period: int) -> pl.Series:
    """
    考夫曼自适应移动平均线 (Kaufman Adaptive Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        KAMA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.kama(prices, 10)
    """
    ...

def mama(series: pl.Series, fast_limit: float, slow_limit: float) -> Tuple[pl.Series, pl.Series]:
    """
    MESA自适应移动平均线 (MESA Adaptive Moving Average)
    
    Args:
        series: 价格序列
        fast_limit: 快速限制 (通常为0.5)
        slow_limit: 慢速限制 (通常为0.05)
    
    Returns:
        Tuple[MAMA序列, FAMA序列]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> mama, fama = pq.mama(prices, 0.5, 0.05)
    """
    ...

def t3(series: pl.Series, period: int, volume_factor: float) -> pl.Series:
    """
    T3移动平均线 (T3 Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
        volume_factor: 体积因子 (通常为0.7)
    
    Returns:
        T3序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.t3(prices, 5, 0.7)
    """
    ...

def ma(series: pl.Series, period: int = 20, ma_type: str = "SMA") -> pl.Series:
    """
    移动平均 (Moving Average) - 支持 SMA、EMA、WMA
    
    Args:
        series: 价格序列
        period: 计算周期，默认 20
        ma_type: 移动平均类型，支持：
            - "SMA": 简单移动平均 (Simple Moving Average)
            - "EMA": 指数移动平均 (Exponential Moving Average)
            - "WMA": 加权移动平均 (Weighted Moving Average)
            默认 "SMA"
    
    Returns:
        MA 序列
    
    Notes:
        - SMA: 所有值权重相等的简单平均
        - EMA: 对近期数据赋予更高权重的指数平滑
        - WMA: 线性加权，最近的值权重最大（权重为 1, 2, 3, ..., period）
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> sma = pq.ma(prices, 5)  # 默认 SMA
        >>> ema = pq.ma(prices, 5, "EMA")
        >>> wma = pq.ma(prices, 5, "WMA")
    """
    ...

def mavp(series: pl.Series, periods: pl.Series, min_period: int, max_period: int) -> pl.Series:
    """
    可变周期移动平均线 (Moving Average with Variable Period)
    
    Args:
        series: 价格序列
        periods: 周期序列
        min_period: 最小周期
        max_period: 最大周期
    
    Returns:
        MAVP序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> periods = pl.Series([3, 3, 4, 4, 5, 5])
        >>> result = pq.mavp(prices, periods, 2, 30)
    """
    ...

def midpoint(series: pl.Series, period: int) -> pl.Series:
    """
    中点价格 (Midpoint over period)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        中点价格序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.midpoint(prices, 3)
    """
    ...

def midprice_hl(high: pl.Series, low: pl.Series, period: int) -> pl.Series:
    """
    最高最低价中点 (Midpoint Price over period)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期
    
    Returns:
        中点价格序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.midprice_hl(high, low, 3)
    """
    ...

def sar(high: pl.Series, low: pl.Series, acceleration: float, maximum: float) -> pl.Series:
    """
    抛物线SAR (Parabolic Stop and Reverse)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        acceleration: 加速因子 (通常为0.02)
        maximum: 最大值 (通常为0.2)
    
    Returns:
        SAR序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.sar(high, low, 0.02, 0.2)
    """
    ...

def sarext(
    high: pl.Series, 
    low: pl.Series, 
    startvalue: Optional[float] = None, 
    offsetonreverse: Optional[float] = None,
    accelerationinitlong: Optional[float] = None,
    accelerationlong: Optional[float] = None,
    accelerationmaxlong: Optional[float] = None,
    accelerationinitshort: Optional[float] = None,
    accelerationshort: Optional[float] = None,
    accelerationmaxshort: Optional[float] = None
) -> pl.Series:
    """
    抛物线SAR扩展版 (Parabolic Stop and Reverse - Extended)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        startvalue: 起始值
        offsetonreverse: 反转偏移
        accelerationinitlong: 多头初始加速
        accelerationlong: 多头加速
        accelerationmaxlong: 多头最大加速
        accelerationinitshort: 空头初始加速
        accelerationshort: 空头加速
        accelerationmaxshort: 空头最大加速
    
    Returns:
        SAR扩展序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.sarext(high, low)
    """
    ...

# ====================================================================
# 动量指标 (Momentum Indicators) - 趋势强度和方向指标
# ====================================================================

def adx(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    平均趋向指标 (Average Directional Movement Index)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
    
    Returns:
        ADX序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.adx(high, low, close, 14)
    """
    ...

def adxr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    平均趋向指标评级 (Average Directional Movement Index Rating)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
    
    Returns:
        ADXR序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.adxr(high, low, close, 14)
    """
    ...

def dx(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    方向性指标 (Directional Movement Index)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
    
    Returns:
        DX序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.dx(high, low, close, 14)
    """
    ...

def plus_di(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    正方向指标 (Plus Directional Indicator)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
    
    Returns:
        +DI序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.plus_di(high, low, close, 14)
    """
    ...

def minus_di(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    负方向指标 (Minus Directional Indicator)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期
    
    Returns:
        -DI序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.minus_di(high, low, close, 14)
    """
    ...

def plus_dm(high: pl.Series, low: pl.Series, period: int) -> pl.Series:
    """
    正方向移动 (Plus Directional Movement)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期
    
    Returns:
        +DM序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.plus_dm(high, low, 14)
    """
    ...

def minus_dm(high: pl.Series, low: pl.Series, period: int) -> pl.Series:
    """
    负方向移动 (Minus Directional Movement)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期
    
    Returns:
        -DM序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.minus_dm(high, low, 14)
    """
    ...

def macd(series: pl.Series, fast: int, slow: int, signal: int) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """
    MACD指标 (Moving Average Convergence Divergence)
    
    Args:
        series: 价格序列
        fast: 快速EMA周期 (通常为12)
        slow: 慢速EMA周期 (通常为26)
        signal: 信号线EMA周期 (通常为9)
    
    Returns:
        Tuple[MACD线, 信号线, 柱状图]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> macd_line, signal_line, histogram = pq.macd(prices, 12, 26, 9)
    """
    ...

def macdext(
    series: pl.Series, 
    fast: int, 
    slow: int, 
    signal: int, 
    fast_matype: str, 
    slow_matype: str, 
    signal_matype: str
) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """
    MACD扩展版 (MACD with controllable MA type)
    
    Args:
        series: 价格序列
        fast: 快速MA周期
        slow: 慢速MA周期
        signal: 信号线周期
        fast_matype: 快速MA类型 ("sma", "ema", 等)
        slow_matype: 慢速MA类型
        signal_matype: 信号线MA类型
    
    Returns:
        Tuple[MACD线, 信号线, 柱状图]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> macd_line, signal_line, histogram = pq.macdext(prices, 12, 26, 9, "ema", "ema", "sma")
    """
    ...

def macdfix(series: pl.Series, signal: int) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """
    MACD固定参数版 (Moving Average Convergence Divergence - Fix 12/26)
    
    Args:
        series: 价格序列
        signal: 信号线周期 (通常为9)
    
    Returns:
        Tuple[MACD线, 信号线, 柱状图]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> macd_line, signal_line, histogram = pq.macdfix(prices, 9)
    """
    ...

def rsi(series: pl.Series, period: int) -> pl.Series:
    """
    相对强弱指标 (Relative Strength Index)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为14)
    
    Returns:
        RSI序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rsi(prices, 14)
    """
    ...

def stoch(high: pl.Series, low: pl.Series, close: pl.Series, k_period: int, d_period: int) -> Tuple[pl.Series, pl.Series]:
    """
    随机指标 (Stochastic)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: K值周期 (通常为5)
        d_period: D值周期 (通常为3)
    
    Returns:
        Tuple[K值序列, D值序列]
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> k_values, d_values = pq.stoch(high, low, close, 5, 3)
    """
    ...

def stochf(high: pl.Series, low: pl.Series, close: pl.Series, k_period: int, d_period: int) -> Tuple[pl.Series, pl.Series]:
    """
    快速随机指标 (Stochastic Fast)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: K值周期
        d_period: D值周期
    
    Returns:
        Tuple[快速K值, 快速D值]
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> fast_k, fast_d = pq.stochf(high, low, close, 5, 3)
    """
    ...

def stochrsi(series: pl.Series, period: int, k_period: int, d_period: int) -> Tuple[pl.Series, pl.Series]:
    """
    RSI随机指标 (Stochastic Relative Strength Index)
    
    Args:
        series: 价格序列
        period: RSI周期
        k_period: K值周期
        d_period: D值周期
    
    Returns:
        Tuple[StochRSI K值, StochRSI D值]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> k_values, d_values = pq.stochrsi(prices, 14, 5, 3)
    """
    ...

def willr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    威廉指标 (Williams %R)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期 (通常为14)
    
    Returns:
        Williams %R序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.willr(high, low, close, 14)
    """
    ...

def ultosc(high: pl.Series, low: pl.Series, close: pl.Series, period1: int, period2: int, period3: int) -> pl.Series:
    """
    终极摆动指标 (Ultimate Oscillator)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period1: 第一周期 (通常为7)
        period2: 第二周期 (通常为14)
        period3: 第三周期 (通常为28)
    
    Returns:
        终极摆动指标序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.ultosc(high, low, close, 7, 14, 28)
    """
    ...

def aroon(high: pl.Series, low: pl.Series, period: int) -> Tuple[pl.Series, pl.Series]:
    """
    阿隆指标 (Aroon)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期 (通常为14)
    
    Returns:
        Tuple[Aroon Up, Aroon Down]
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> aroon_up, aroon_down = pq.aroon(high, low, 14)
    """
    ...

def aroonosc(high: pl.Series, low: pl.Series, period: int) -> pl.Series:
    """
    阿隆摆动指标 (Aroon Oscillator)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        period: 计算周期 (通常为14)
    
    Returns:
        Aroon摆动指标序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.aroonosc(high, low, 14)
    """
    ...

def apo(series: pl.Series, fast_period: int, slow_period: int) -> pl.Series:
    """
    绝对价格摆动指标 (Absolute Price Oscillator)
    
    Args:
        series: 价格序列
        fast_period: 快速周期 (通常为12)
        slow_period: 慢速周期 (通常为26)
    
    Returns:
        APO序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.apo(prices, 12, 26)
    """
    ...

def ppo(series: pl.Series, fast_period: int, slow_period: int) -> pl.Series:
    """
    价格摆动百分比 (Percentage Price Oscillator)
    
    Args:
        series: 价格序列
        fast_period: 快速周期 (通常为12)
        slow_period: 慢速周期 (通常为26)
    
    Returns:
        PPO序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ppo(prices, 12, 26)
    """
    ...

def bop(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    均势指标 (Balance Of Power)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        BOP序列
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.bop(open, high, low, close)
    """
    ...

def cci(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    顺势指标 (Commodity Channel Index)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期 (通常为14)
    
    Returns:
        CCI序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cci(high, low, close, 14)
    """
    ...

def cmo(series: pl.Series, period: int) -> pl.Series:
    """
    钱德动量摆动指标 (Chande Momentum Oscillator)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为14)
    
    Returns:
        CMO序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.cmo(prices, 14)
    """
    ...

def mfi(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, period: int) -> pl.Series:
    """
    资金流量指标 (Money Flow Index)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        period: 计算周期 (通常为14)
    
    Returns:
        MFI序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.mfi(high, low, close, volume, 14)
    """
    ...

def mom(series: pl.Series, period: int) -> pl.Series:
    """
    动量指标 (Momentum)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        动量序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.mom(prices, 10)
    """
    ...

def trix(series: pl.Series, period: int) -> pl.Series:
    """
    TRIX指标 (1-day Rate-Of-Change (ROC) of a Triple Smooth EMA)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为14)
    
    Returns:
        TRIX序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.trix(prices, 14)
    """
    ...

def roc(series: pl.Series, period: int) -> pl.Series:
    """
    变化率 (Rate of change)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROC序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.roc(prices, 10)
    """
    ...

def rocp(series: pl.Series, period: int) -> pl.Series:
    """
    变化率百分比 (Rate of change Percentage)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROCP序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rocp(prices, 10)
    """
    ...

def rocr(series: pl.Series, period: int) -> pl.Series:
    """
    变化率比率 (Rate of change ratio)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROCR序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rocr(prices, 10)
    """
    ...

def rocr100(series: pl.Series, period: int) -> pl.Series:
    """
    变化率比率*100 (Rate of change ratio 100 scale)
    
    Args:
        series: 价格序列
        period: 计算周期 (通常为10)
    
    Returns:
        ROCR100序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.rocr100(prices, 10)
    """
    ...

# ====================================================================
# 成交量指标 (Volume Indicators) - 成交量相关指标
# ====================================================================

def ad(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series) -> pl.Series:
    """
    累积/派发线 (Accumulation/Distribution Line)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
    
    Returns:
        A/D线序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.ad(high, low, close, volume)
    """
    ...

def adosc(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, fast_period: int, slow_period: int) -> pl.Series:
    """
    累积/派发摆动指标 (Accumulation/Distribution Oscillator)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        fast_period: 快速周期 (通常为3)
        slow_period: 慢速周期 (通常为10)
    
    Returns:
        A/D摆动指标序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.adosc(high, low, close, volume, 3, 10)
    """
    ...

def obv(close: pl.Series, volume: pl.Series) -> pl.Series:
    """
    能量潮指标 (On Balance Volume)
    
    Args:
        close: 收盘价序列
        volume: 成交量序列
    
    Returns:
        OBV序列
    
    Example:
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> volume = pl.Series([100, 200, 300, 400, 500])
        >>> result = pq.obv(close, volume)
    """
    ...

# ====================================================================
# 波动率指标 (Volatility Indicators) - 价格波动测量
# ====================================================================

def atr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    平均真实波幅 (Average True Range)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期 (通常为14)
    
    Returns:
        ATR序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.atr(high, low, close, 14)
    """
    ...

def natr(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """
    标准化平均真实波幅 (Normalized Average True Range)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 计算周期 (通常为14)
    
    Returns:
        NATR序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.natr(high, low, close, 14)
    """
    ...

def trange(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    真实波幅 (True Range)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        真实波幅序列
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.trange(high, low, close)
    """
    ...

# ====================================================================
# 价格变换函数 (Price Transform) - 价格数据变换
# ====================================================================

def avgprice(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    平均价格 (Average Price)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        平均价格序列 (OHLC/4)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.avgprice(open, high, low, close)
    """
    ...

def medprice(high: pl.Series, low: pl.Series) -> pl.Series:
    """
    中位价格 (Median Price)
    
    Args:
        high: 最高价序列
        low: 最低价序列
    
    Returns:
        中位价格序列 (HL/2)
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> result = pq.medprice(high, low)
    """
    ...

def typprice(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    典型价格 (Typical Price)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        典型价格序列 (HLC/3)
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.typprice(high, low, close)
    """
    ...

def wclprice(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    加权收盘价格 (Weighted Close Price)
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        加权收盘价格序列 (HLC2/4)
    
    Example:
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.wclprice(high, low, close)
    """
    ...

# ====================================================================
# 周期指标 (Cycle Indicators) - 希尔伯特变换系列
# ====================================================================

def ht_trendline(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-趋势线 (Hilbert Transform - Instantaneous Trendline)
    
    Args:
        series: 价格序列
    
    Returns:
        瞬时趋势线序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_trendline(prices)
    """
    ...

def ht_dcperiod(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-主导周期 (Hilbert Transform - Dominant Cycle Period)
    
    Args:
        series: 价格序列
    
    Returns:
        主导周期序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_dcperiod(prices)
    """
    ...

def ht_dcphase(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-主导周期相位 (Hilbert Transform - Dominant Cycle Phase)
    
    Args:
        series: 价格序列
    
    Returns:
        主导周期相位序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_dcphase(prices)
    """
    ...

def ht_phasor(series: pl.Series) -> Tuple[pl.Series, pl.Series]:
    """
    希尔伯特变换-相量分量 (Hilbert Transform - Phasor Components)
    
    Args:
        series: 价格序列
    
    Returns:
        Tuple[同相分量, 正交分量]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> inphase, quadrature = pq.ht_phasor(prices)
    """
    ...

def ht_sine(series: pl.Series) -> Tuple[pl.Series, pl.Series]:
    """
    希尔伯特变换-正弦波 (Hilbert Transform - SineWave)
    
    Args:
        series: 价格序列
    
    Returns:
        Tuple[正弦波, 余弦波]
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> sine, leadsine = pq.ht_sine(prices)
    """
    ...

def ht_trendmode(series: pl.Series) -> pl.Series:
    """
    希尔伯特变换-趋势模式 (Hilbert Transform - Trend vs Cycle Mode)
    
    Args:
        series: 价格序列
    
    Returns:
        趋势模式序列 (0=周期模式, 1=趋势模式)
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = pq.ht_trendmode(prices)
    """
    ...

# ====================================================================
# 蜡烛图模式识别 (Candlestick Pattern Recognition) - K线形态
# ====================================================================

def cdldoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    十字星 (Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldoji(open, high, low, close)
    """
    ...

def cdldojistar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    十字星 (Doji Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldojistar(open, high, low, close)
    """
    ...

def cdldragonflydoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    蜻蜓十字 (Dragonfly Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldragonflydoji(open, high, low, close)
    """
    ...

def cdlgravestonedoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    墓石十字 (Gravestone Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlgravestonedoji(open, high, low, close)
    """
    ...

def cdllongleggeddoji(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    长腿十字 (Long Legged Doji)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdllongleggeddoji(open, high, low, close)
    """
    ...

def cdlhammer(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    锤头线 (Hammer)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhammer(open, high, low, close)
    """
    ...

def cdlhangingman(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    上吊线 (Hanging Man)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhangingman(open, high, low, close)
    """
    ...

def cdlinvertedhammer(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    倒锤头线 (Inverted Hammer)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlinvertedhammer(open, high, low, close)
    """
    ...

def cdlshootingstar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    射击之星 (Shooting Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlshootingstar(open, high, low, close)
    """
    ...

def cdlengulfing(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    吞没形态 (Engulfing Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlengulfing(open, high, low, close)
    """
    ...

def cdlharami(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    孕线形态 (Harami Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlharami(open, high, low, close)
    """
    ...

def cdlharamicross(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    孕线十字 (Harami Cross Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlharamicross(open, high, low, close)
    """
    ...

def cdlmorningstar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    早晨之星 (Morning Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmorningstar(open, high, low, close, 0.3)
    """
    ...

def cdleveningstar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    黄昏之星 (Evening Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdleveningstar(open, high, low, close, 0.3)
    """
    ...

def cdlmorningdojistar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    早晨十字星 (Morning Doji Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmorningdojistar(open, high, low, close, 0.3)
    """
    ...

def cdleveningdojistar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    黄昏十字星 (Evening Doji Star)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透率 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdleveningdojistar(open, high, low, close, 0.3)
    """
    ...

def cdl3blackcrows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三只乌鸦 (Three Black Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3blackcrows(open, high, low, close)
    """
    ...

def cdl3whitesoldiers(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三个白武士 (Three Advancing White Soldiers)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3whitesoldiers(open, high, low, close)
    """
    ...

def cdlpiercing(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    穿透形态 (Piercing Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlpiercing(open, high, low, close)
    """
    ...

def cdldarkcloudcover(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    乌云盖顶 (Dark Cloud Cover)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透百分比 (通常为0.5)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdldarkcloudcover(open, high, low, close, 0.5)
    """
    ...

def cdlabandonedbaby(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    弃婴形态 (Abandoned Baby)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透百分比 (通常为0.3)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlabandonedbaby(open, high, low, close, 0.3)
    """
    ...

def cdltristar(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三星形态 (Tri-Star Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdltristar(open, high, low, close)
    """
    ...

def cdl3inside(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三内部上升/下降 (Three Inside Up/Down)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3inside(open, high, low, close)
    """
    ...

def cdl3outside(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三外部上升/下降 (Three Outside Up/Down)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3outside(open, high, low, close)
    """
    ...

def cdl3linestrike(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    三线攻击 (Three-Line Strike)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3linestrike(open, high, low, close)
    """
    ...

def cdl3starsinsouth(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    南方三星 (Three Stars In The South)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl3starsinsouth(open, high, low, close)
    """
    ...

def cdlidentical3crows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    相同三乌鸦 (Identical Three Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlidentical3crows(open, high, low, close)
    """
    ...

def cdlgapsidesidewhite(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    向上跳空并列阳线 (Up-side Gap Two Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlgapsidesidewhite(open, high, low, close)
    """
    ...

def cdlupsidegap2crows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    向上跳空两只乌鸦 (Upside Gap Two Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlupsidegap2crows(open, high, low, close)
    """
    ...

def cdltasukigap(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    跳空并列阴阳线 (Tasuki Gap)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdltasukigap(open, high, low, close)
    """
    ...

def cdlxsidegap3methods(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    上升/下降跳空三法 (Upside/Downside Gap Three Methods)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlxsidegap3methods(open, high, low, close)
    """
    ...

def cdl2crows(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    两只乌鸦 (Two Crows)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdl2crows(open, high, low, close)
    """
    ...

def cdladvanceblock(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    前进阻挡 (Advance Block)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdladvanceblock(open, high, low, close)
    """
    ...

def cdlbelthold(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    捉腰带线 (Belt-hold)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlbelthold(open, high, low, close)
    """
    ...

def cdlbreakaway(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    脱离形态 (Breakaway)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlbreakaway(open, high, low, close)
    """
    ...

def cdlclosingmarubozu(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    收盘秃头 (Closing Marubozu)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlclosingmarubozu(open, high, low, close)
    """
    ...

def cdlconcealbabyswall(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    藏婴吞没 (Concealing Baby Swallow)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlconcealbabyswall(open, high, low, close)
    """
    ...

def cdlcounterattack(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    反击线 (Counterattack)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlcounterattack(open, high, low, close)
    """
    ...

def cdlhighwave(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    长影线 (High-Wave Candle)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhighwave(open, high, low, close)
    """
    ...

def cdlhikkake(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    陷阱形态 (Hikkake Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhikkake(open, high, low, close)
    """
    ...

def cdlhikkakemod(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    修正陷阱形态 (Modified Hikkake Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhikkakemod(open, high, low, close)
    """
    ...

def cdlhomingpigeon(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    家鸽形态 (Homing Pigeon)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlhomingpigeon(open, high, low, close)
    """
    ...

def cdlinneck(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    颈内线 (In-Neck Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlinneck(open, high, low, close)
    """
    ...

def cdlkicking(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    反冲形态 (Kicking)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlkicking(open, high, low, close)
    """
    ...

def cdlkickingbylength(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    由长度决定的反冲形态 (Kicking - bull/bear determined by the longer marubozu)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlkickingbylength(open, high, low, close)
    """
    ...

def cdlladderbottom(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    梯底形态 (Ladder Bottom)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlladderbottom(open, high, low, close)
    """
    ...

def cdllongline(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    长线形态 (Long Line Candle)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdllongline(open, high, low, close)
    """
    ...

def cdlmarubozu(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    秃头形态 (Marubozu)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmarubozu(open, high, low, close)
    """
    ...

def cdlmatchinglow(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    相同低价 (Matching Low)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmatchinglow(open, high, low, close)
    """
    ...

def cdlmathold(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series, penetration: float) -> pl.Series:
    """
    铺垫形态 (Mat Hold)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        penetration: 穿透百分比 (通常为0.5)
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlmathold(open, high, low, close, 0.5)
    """
    ...

def cdlonneck(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    颈上线 (On-Neck Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlonneck(open, high, low, close)
    """
    ...

def cdlrickshawman(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    黄包车夫 (Rickshaw Man)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlrickshawman(open, high, low, close)
    """
    ...

def cdlrisefall3methods(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    上升/下降三法 (Rising/Falling Three Methods)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlrisefall3methods(open, high, low, close)
    """
    ...

def cdlseparatinglines(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    分离线 (Separating Lines)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlseparatinglines(open, high, low, close)
    """
    ...

def cdlshortline(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    短线形态 (Short Line Candle)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlshortline(open, high, low, close)
    """
    ...

def cdlspinningtop(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    纺锤形态 (Spinning Top)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlspinningtop(open, high, low, close)
    """
    ...

def cdlstalledpattern(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    停顿形态 (Stalled Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlstalledpattern(open, high, low, close)
    """
    ...

def cdlsticksandwich(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    条形三明治 (Stick Sandwich)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlsticksandwich(open, high, low, close)
    """
    ...

def cdltakuri(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    探水竿 (Takuri - Dragonfly Doji with very long lower shadow)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdltakuri(open, high, low, close)
    """
    ...

def cdlthrusting(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    插入形态 (Thrusting Pattern)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlthrusting(open, high, low, close)
    """
    ...

def cdlunique3river(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """
    奇特三河床 (Unique 3 River)
    
    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
    
    Returns:
        信号序列 (100=看涨, -100=看跌, 0=无信号)
    
    Example:
        >>> open = pl.Series([2, 3, 4, 5, 6])
        >>> high = pl.Series([5, 6, 7, 8, 9])
        >>> low = pl.Series([1, 2, 3, 4, 5])
        >>> close = pl.Series([3, 4, 5, 6, 7])
        >>> result = pq.cdlunique3river(open, high, low, close)
    """
    ...


# ====================================================================
# 交易策略 (Trading Strategy)
# ====================================================================

class Strategy:
    """
    交易策略模块
    
    提供多种常用交易策略，每个策略返回包含 buy_signal 和 sell_signal 列的 DataFrame
    
    主要策略:
    - MA均线策略（支持多种均线类型、趋势过滤、斜率过滤）
    - MACD策略（经典MACD金叉死叉）
    - RSI策略（超买超卖区间）
    - 布林带策略（突破上下轨）
    - KDJ策略（随机指标）
    - CCI策略（顺势指标）
    - ADX策略（趋势强度）
    - 突破策略（新高新低突破）
    - 反转策略（均值回归）
    - 成交量策略（放量突破）
    - 网格策略（区间震荡）
    - 跳空策略（缺口交易）
    - 形态策略（K线形态识别）
    - 趋势策略（多均线趋势判断）
    
    Examples:
        >>> import polars as pl
        >>> from polars_quant import Strategy
        >>> 
        >>> df = pl.DataFrame({
        ...     "date": ["2024-01-01", "2024-01-02"],
        ...     "open": [100.0, 102.0],
        ...     "high": [105.0, 106.0],
        ...     "low": [99.0, 101.0],
        ...     "close": [103.0, 104.0],
        ...     "volume": [1000000, 1200000]
        ... })
        >>> 
        >>> strategy = Strategy()
        >>> 
        >>> # MA均线策略
        >>> signals = strategy.ma(df, fast_period=10, slow_period=20)
        >>> 
        >>> # MACD策略
        >>> signals = strategy.macd(df)
        >>> 
        >>> # 组合多个策略
        >>> ma_signals = strategy.ma(df, fast_period=5, slow_period=10)
        >>> rsi_signals = strategy.rsi(df, period=14, oversold=30, overbought=70)
        >>> # 取交集（同时满足）
        >>> combined = ma_signals.with_columns([
        ...     (pl.col("buy_signal") & rsi_signals["buy_signal"]).alias("buy_signal"),
        ...     (pl.col("sell_signal") | rsi_signals["sell_signal"]).alias("sell_signal")
        ... ])
    """
    
    def __init__(self) -> None:
        """创建Strategy实例"""
        ...
    
    def ma(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        fast_period: int = 10,
        slow_period: int = 20,
        ma_type: str = "sma",
        trend_period: int = 0,
        trend_filter: bool = False,
        slope_filter: bool = False,
        distance_pct: float = 0.0
    ) -> pl.DataFrame:
        """
        MA均线策略（支持多种过滤条件）
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            fast_period: 快线周期（默认10）
            slow_period: 慢线周期（默认20）
            ma_type: 均线类型（默认"sma"），可选: "sma", "ema", "wma", "dema", "tema"
            trend_period: 趋势过滤均线周期（默认0，不使用）
            trend_filter: 是否启用趋势过滤（价格在趋势线上方才买入）
            slope_filter: 是否启用斜率过滤（均线斜率向上才买入）
            distance_pct: 价格与均线最小距离百分比过滤（默认0.0，不使用）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 快线上穿慢线时为True
            - sell_signal: 快线下穿慢线时为True
        
        Examples:
            >>> # 简单MA策略
            >>> signals = strategy.ma(df, fast_period=5, slow_period=10)
            >>> 
            >>> # 带趋势过滤的MA策略
            >>> signals = strategy.ma(df, fast_period=10, slow_period=20,
            ...                       trend_period=60, trend_filter=True)
            >>> 
            >>> # EMA策略
            >>> signals = strategy.ma(df, ma_type="ema", fast_period=12, slow_period=26)
        """
        ...
    
    def macd(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pl.DataFrame:
        """
        MACD策略
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            fast_period: 快线周期（默认12）
            slow_period: 慢线周期（默认26）
            signal_period: 信号线周期（默认9）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: MACD金叉（MACD线上穿信号线）
            - sell_signal: MACD死叉（MACD线下穿信号线）
        """
        ...
    
    def rsi(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0
    ) -> pl.DataFrame:
        """
        RSI策略
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: RSI周期（默认14）
            oversold: 超卖阈值（默认30）
            overbought: 超买阈值（默认70）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: RSI从超卖区向上突破时为True
            - sell_signal: RSI从超买区向下突破时为True
        """
        ...
    
    def bband(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20,
        std_dev: float = 2.0
    ) -> pl.DataFrame:
        """
        布林带策略
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 周期（默认20）
            std_dev: 标准差倍数（默认2.0）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 价格突破下轨后回归时为True
            - sell_signal: 价格突破上轨后回落时为True
        """
        ...
    
    def stoch(
        self,
        df: pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0
    ) -> pl.DataFrame:
        """
        KDJ/Stochastic策略
        
        Args:
            df: 包含价格数据的DataFrame
            high_col: 最高价列名（默认"high"）
            low_col: 最低价列名（默认"low"）
            close_col: 收盘价列名（默认"close"）
            fastk_period: FastK周期（默认14）
            slowk_period: SlowK周期（默认3）
            slowd_period: SlowD周期（默认3）
            oversold: 超卖阈值（默认20）
            overbought: 超买阈值（默认80）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: K线从超卖区向上突破D线
            - sell_signal: K线从超买区向下突破D线
        """
        ...
    
    def cci(
        self,
        df: pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        period: int = 20,
        oversold: float = -100.0,
        overbought: float = 100.0
    ) -> pl.DataFrame:
        """
        CCI顺势指标策略
        
        Args:
            df: 包含价格数据的DataFrame
            high_col: 最高价列名（默认"high"）
            low_col: 最低价列名（默认"low"）
            close_col: 收盘价列名（默认"close"）
            period: 周期（默认20）
            oversold: 超卖阈值（默认-100）
            overbought: 超买阈值（默认100）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
        """
        ...
    
    def adx(
        self,
        df: pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        period: int = 14,
        threshold: float = 25.0
    ) -> pl.DataFrame:
        """
        ADX趋势强度策略
        
        Args:
            df: 包含价格数据的DataFrame
            high_col: 最高价列名（默认"high"）
            low_col: 最低价列名（默认"low"）
            close_col: 收盘价列名（默认"close"）
            period: 周期（默认14）
            threshold: ADX阈值（默认25，ADX>25表示趋势明显）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: ADX>threshold且+DI>-DI
            - sell_signal: ADX>threshold且-DI>+DI
        """
        ...
    
    def breakout(
        self,
        df: pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        period: int = 20,
        atr_multiplier: float = 2.0
    ) -> pl.DataFrame:
        """
        突破策略（Donchian Channel + ATR过滤）
        
        Args:
            df: 包含价格数据的DataFrame
            high_col: 最高价列名（默认"high"）
            low_col: 最低价列名（默认"low"）
            close_col: 收盘价列名（默认"close"）
            period: 通道周期（默认20）
            atr_multiplier: ATR止损倍数（默认2.0）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 突破N日最高价
            - sell_signal: 跌破N日最低价
        """
        ...
    
    def reversion(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5
    ) -> pl.DataFrame:
        """
        均值回归策略
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 均值周期（默认20）
            entry_std: 入场标准差倍数（默认2.0）
            exit_std: 出场标准差倍数（默认0.5）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 价格低于均值-entry_std*std
            - sell_signal: 价格回归至均值-exit_std*std以上
        """
        ...
    
    def volume(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
        volume_period: int = 20,
        volume_multiplier: float = 2.0,
        price_period: int = 20
    ) -> pl.DataFrame:
        """
        成交量突破策略
        
        Args:
            df: 包含价格和成交量数据的DataFrame
            price_col: 价格列名（默认"close"）
            volume_col: 成交量列名（默认"volume"）
            volume_period: 成交量均值周期（默认20）
            volume_multiplier: 成交量倍数（默认2.0）
            price_period: 价格突破周期（默认20）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 放量突破N日最高价
            - sell_signal: 放量跌破N日最低价
        """
        ...
    
    def grid(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        grid_size: float = 0.05,
        base_price: Optional[float] = None
    ) -> pl.DataFrame:
        """
        网格交易策略
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            grid_size: 网格间距（百分比，默认0.05=5%）
            base_price: 基准价格（默认None，使用第一个价格）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 价格下跌至下一个网格线
            - sell_signal: 价格上涨至上一个网格线
        """
        ...
    
    def gap(
        self,
        df: pl.DataFrame,
        open_col: str = "open",
        close_col: str = "close",
        gap_threshold: float = 0.02
    ) -> pl.DataFrame:
        """
        跳空缺口策略
        
        Args:
            df: 包含价格数据的DataFrame
            open_col: 开盘价列名（默认"open"）
            close_col: 收盘价列名（默认"close"）
            gap_threshold: 跳空阈值（默认0.02=2%）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 向上跳空
            - sell_signal: 向下跳空
        """
        ...
    
    def pattern(
        self,
        df: pl.DataFrame,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        pattern_type: str = "hammer"
    ) -> pl.DataFrame:
        """
        K线形态策略
        
        Args:
            df: 包含价格数据的DataFrame
            open_col: 开盘价列名（默认"open"）
            high_col: 最高价列名（默认"high"）
            low_col: 最低价列名（默认"low"）
            close_col: 收盘价列名（默认"close"）
            pattern_type: 形态类型（默认"hammer"），可选:
                - "hammer": 锤子线
                - "shooting_star": 流星线
                - "engulfing": 吞没形态
                - "doji": 十字星
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
        """
        ...
    
    def trend(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        short_period: int = 5,
        medium_period: int = 10,
        long_period: int = 20
    ) -> pl.DataFrame:
        """
        多均线趋势策略
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            short_period: 短期均线周期（默认5）
            medium_period: 中期均线周期（默认10）
            long_period: 长期均线周期（默认20）
        
        Returns:
            包含 buy_signal 和 sell_signal 列的DataFrame
            - buy_signal: 短期>中期>长期（多头排列）
            - sell_signal: 短期<中期<长期（空头排列）
        """
        ...


# ====================================================================
# 因子挖掘和评估 (Factor Mining & Evaluation)
# ====================================================================

class Factor:
    """
    因子挖掘和评估模块
    
    提供因子计算、因子评估、IC分析等功能，用于量化选股和因子研究
    
    主要功能:
    - 15+ 技术因子计算（动量、反转、波动率、成交量等）
    - 8种专业因子评估指标（IC、IR、Rank IC、分层分析等）
    - 支持多种动量计算方法（简单收益率、对数收益率、残差动量、动量加速度）
    - 支持自定义因子列名，可在同一DataFrame中计算多个因子
    
    Examples:
        基本因子分析流程：
        >>> import polars as pl
        >>> from polars_quant import Factor
        >>> 
        >>> # 准备数据
        >>> df = pl.DataFrame({
        ...     "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        ...     "symbol": ["AAPL", "AAPL", "AAPL"],
        ...     "close": [150.0, 152.0, 148.0],
        ...     "volume": [1000000, 1200000, 900000]
        ... })
        >>> 
        >>> # 创建Factor实例
        >>> factor = Factor()
        >>> 
        >>> # 计算动量因子
        >>> df = factor.momentum(df, period=20)
        >>> 
        >>> # 计算波动率因子
        >>> df = factor.volatility(df, period=20)
        >>> 
        >>> # 评估因子（需要先有收益率列）
        >>> ic = factor.ic(df, "momentum", "return")
        >>> ir = factor.ir(df, "momentum", "return")
        
        多种动量计算方法：
        >>> # 简单收益率动量（默认）
        >>> df = factor.momentum(df, period=20)
        >>> 
        >>> # 对数收益率动量
        >>> df = factor.momentum(df, period=60, method="log", factor_col="log_momentum")
        >>> 
        >>> # 残差动量（去除市场整体趋势）
        >>> df = factor.momentum(df, period=20, method="residual", factor_col="residual_mom")
        >>> 
        >>> # 动量加速度（捕捉趋势变化）
        >>> df = factor.momentum(df, period=20, method="acceleration", factor_col="mom_accel")
    """
    
    def __init__(self) -> None:
        """创建Factor实例"""
        ...
    
    # ================================================================
    # 因子计算方法
    # ================================================================
    
    def momentum(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20,
        method: str = "return",
        factor_col: str = "momentum"
    ) -> pl.DataFrame:
        """
        动量因子（增强版）
        
        计算价格动量，支持多种计算方式
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认20天）
            method: 计算方法（默认"return"）
                - "return": 简单收益率 (current - past) / past
                - "log": 对数收益率 ln(current / past)
                - "residual": 残差动量（去除市场整体趋势）
                - "acceleration": 动量加速度（动量的变化率）
            factor_col: 因子列名（默认"momentum"）
        
        Returns:
            添加了动量因子列的DataFrame
        
        Examples:
            >>> # 简单收益率动量（默认）
            >>> df = factor.momentum(df, period=20)
            >>> 
            >>> # 对数收益率动量（适合长周期）
            >>> df = factor.momentum(df, period=60, method="log")
            >>> 
            >>> # 残差动量（去除市场整体趋势）
            >>> df = factor.momentum(df, period=20, method="residual")
            >>> 
            >>> # 动量加速度（捕捉趋势变化）
            >>> df = factor.momentum(df, period=20, method="acceleration")
        """
        ...
    
    def reversal(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 5
    ) -> pl.DataFrame:
        """
        反转因子
        
        短期反转效应：过去短期表现差的股票未来可能反转
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认5天）
        
        Returns:
            添加了reversal因子列的DataFrame
        """
        ...
    
    def volatility(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20
    ) -> pl.DataFrame:
        """
        波动率因子
        
        计算价格的标准差作为波动率指标
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了volatility因子列的DataFrame
        """
        ...
    
    def volume_factor(
        self,
        df: pl.DataFrame,
        volume_col: str = "volume",
        period: int = 20
    ) -> pl.DataFrame:
        """
        成交量因子
        
        计算成交量相对于均值的偏离程度
        
        Args:
            df: 包含成交量数据的DataFrame
            volume_col: 成交量列名（默认"volume"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了volume_factor因子列的DataFrame
        """
        ...
    
    def price_volume_corr(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
        period: int = 20
    ) -> pl.DataFrame:
        """
        价量相关性因子
        
        计算价格和成交量的滚动相关系数
        
        Args:
            df: 包含价格和成交量数据的DataFrame
            price_col: 价格列名（默认"close"）
            volume_col: 成交量列名（默认"volume"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了price_volume_corr因子列的DataFrame
        """
        ...
    
    def price_acceleration(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20
    ) -> pl.DataFrame:
        """
        价格加速度因子
        
        计算价格变化的加速度（二阶导数）
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了price_acceleration因子列的DataFrame
        """
        ...
    
    def skewness(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20
    ) -> pl.DataFrame:
        """
        偏度因子
        
        计算收益率分布的偏度
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了skewness因子列的DataFrame
        """
        ...
    
    def kurtosis(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20
    ) -> pl.DataFrame:
        """
        峰度因子
        
        计算收益率分布的峰度
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了kurtosis因子列的DataFrame
        """
        ...
    
    def max_drawdown(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 20
    ) -> pl.DataFrame:
        """
        最大回撤因子
        
        计算过去N天的最大回撤
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了max_drawdown因子列的DataFrame
        """
        ...
    
    def turnover_factor(
        self,
        df: pl.DataFrame,
        volume_col: str = "volume",
        period: int = 20
    ) -> pl.DataFrame:
        """
        换手率因子
        
        计算成交量的变化率
        
        Args:
            df: 包含成交量数据的DataFrame
            volume_col: 成交量列名（默认"volume"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了turnover_factor因子列的DataFrame
        """
        ...
    
    def amplitude_factor(
        self,
        df: pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        period: int = 20
    ) -> pl.DataFrame:
        """
        振幅因子
        
        计算平均振幅
        
        Args:
            df: 包含价格数据的DataFrame
            high_col: 最高价列名（默认"high"）
            low_col: 最低价列名（默认"low"）
            close_col: 收盘价列名（默认"close"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了amplitude_factor因子列的DataFrame
        """
        ...
    
    def price_volume_divergence(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
        period: int = 20
    ) -> pl.DataFrame:
        """
        价量背离因子
        
        检测价格和成交量的背离情况
        
        Args:
            df: 包含价格和成交量数据的DataFrame
            price_col: 价格列名（默认"close"）
            volume_col: 成交量列名（默认"volume"）
            period: 回看周期（默认20天）
        
        Returns:
            添加了price_volume_divergence因子列的DataFrame
        """
        ...
    
    def rsi_factor(
        self,
        df: pl.DataFrame,
        price_col: str = "close",
        period: int = 14
    ) -> pl.DataFrame:
        """
        RSI因子
        
        相对强弱指标，标准化到[-1, 1]区间
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名（默认"close"）
            period: 回看周期（默认14天）
        
        Returns:
            添加了rsi_factor因子列的DataFrame
        """
        ...
    
    # ================================================================
    # 因子评估方法
    # ================================================================
    
    def ic(
        self,
        df: pl.DataFrame,
        factor_col: str,
        return_col: str = "return"
    ) -> float:
        """
        IC值（信息系数）
        
        IC = 因子值与未来收益率的Pearson相关系数
        衡量因子对未来收益的线性预测能力
        
        Args:
            df: 包含因子和收益率数据的DataFrame
            factor_col: 因子列名
            return_col: 未来收益率列名（默认"return"）
        
        Returns:
            IC值（范围 -1 到 1）
        
        评价标准:
            - |IC| > 0.03: 因子有一定预测能力
            - |IC| > 0.05: 因子较优秀
            - |IC| > 0.10: 因子非常优秀
        
        Examples:
            >>> ic = factor.ic(df, "momentum")
            >>> print(f"IC值: {ic:.4f}")
        """
        ...
    
    def ir(
        self,
        df: pl.DataFrame,
        factor_col: str,
        return_col: str = "return",
        period: int = 20
    ) -> float:
        """
        IR值（信息比率）
        
        IR = IC均值 / IC标准差
        衡量因子收益的稳定性和持续性
        
        Args:
            df: 包含因子和收益率数据的DataFrame
            factor_col: 因子列名
            return_col: 未来收益率列名（默认"return"）
            period: 滚动计算IC的窗口期（默认20）
        
        Returns:
            IR值
        
        评价标准:
            - IR > 0.5: 因子较稳定
            - IR > 1.0: 因子很稳定
            - IR > 2.0: 因子非常优秀
        """
        ...
    
    def rank_ic(
        self,
        df: pl.DataFrame,
        factor_col: str,
        return_col: str = "return"
    ) -> float:
        """
        Rank IC值（秩相关系数）
        
        计算因子排名与收益排名的相关性，相比IC更稳健
        
        Args:
            df: 包含因子和收益率数据的DataFrame
            factor_col: 因子列名
            return_col: 未来收益率列名（默认"return"）
        
        Returns:
            Rank IC值（范围 -1 到 1）
        
        评价标准:
            - |Rank IC| > 0.03: 因子有一定预测能力
            - |Rank IC| > 0.05: 因子较优秀
            - |Rank IC| > 0.10: 因子非常优秀
        """
        ...
    
    def quantile(
        self,
        df: pl.DataFrame,
        factor_col: str,
        return_col: str = "return",
        n_quantiles: int = 5
    ) -> pl.DataFrame:
        """
        分层分析
        
        按因子值分层，统计各层的平均收益
        
        Args:
            df: 包含因子和收益率数据的DataFrame
            factor_col: 因子列名
            return_col: 未来收益率列名（默认"return"）
            n_quantiles: 分层数量（默认5）
        
        Returns:
            包含quantile和mean_return列的DataFrame
        
        评价标准:
            - 单调性：分层收益应呈现单调递增/递减
            - 多空收益：最高层与最低层收益差
            - 区分度：各层收益差异越大越好
        """
        ...
    
    def coverage(
        self,
        df: pl.DataFrame,
        factor_col: str
    ) -> float:
        """
        因子覆盖率
        
        计算非空因子值的比例
        
        Args:
            df: 包含因子数据的DataFrame
            factor_col: 因子列名
        
        Returns:
            覆盖率（0-1之间的比例）
        
        评价标准:
            - 覆盖率 > 0.8: 因子覆盖充分
            - 覆盖率 > 0.9: 因子覆盖良好
        """
        ...
    
    def ic_win_rate(
        self,
        df: pl.DataFrame,
        factor_col: str,
        return_col: str = "return",
        period: int = 20
    ) -> float:
        """
        IC胜率
        
        IC胜率 = IC>0的次数 / 总次数
        衡量因子预测方向的准确率
        
        Args:
            df: 包含因子和收益率数据的DataFrame
            factor_col: 因子列名
            return_col: 未来收益率列名（默认"return"）
            period: 滚动窗口（默认20）
        
        Returns:
            IC胜率（0-1之间的比例）
        
        评价标准:
            - IC胜率 > 0.5: 因子有预测能力
            - IC胜率 > 0.6: 因子较强
            - IC胜率 > 0.7: 因子很强
        """
        ...
    
    def long_short(
        self,
        df: pl.DataFrame,
        factor_col: str,
        return_col: str = "return",
        n_quantiles: int = 5
    ) -> float:
        """
        多空收益
        
        多空收益 = 最高分位收益 - 最低分位收益
        衡量因子的盈利能力
        
        Args:
            df: 包含因子和收益率数据的DataFrame
            factor_col: 因子列名
            return_col: 未来收益率列名（默认"return"）
            n_quantiles: 分层数量（默认5）
        
        Returns:
            多空收益（最高层收益 - 最低层收益）
        """
        ...
    
    def turnover(
        self,
        df: pl.DataFrame,
        factor_col: str,
        date_col: str = "date",
        group_col: str = "symbol",
        n_quantiles: int = 5
    ) -> float:
        """
        因子换手率
        
        换手率 = 相邻两期分层组合中股票变化的比例
        衡量因子的稳定性，换手率越低表示因子越稳定
        
        Args:
            df: 包含因子数据的DataFrame（需要包含时间列和分组列）
            factor_col: 因子列名
            date_col: 时间列名（默认"date"）
            group_col: 分组列名（如股票代码，默认"symbol"）
            n_quantiles: 分层数量（默认5）
        
        Returns:
            平均换手率（0-2之间，单边换手率）
        
        评价标准:
            - 换手率 < 0.3: 因子很稳定
            - 换手率 < 0.5: 因子较稳定
            - 换手率 > 0.7: 因子不稳定
        
        说明:
            换手率的计算方法：
            1. 对每个时间截面，按因子值将股票分为n_quantiles组
            2. 对于相邻两期，计算每组中股票的变化比例
            3. 换手率 = (新增股票数 + 减少股票数) / (2 × 组内股票总数)
        """
        ...
