"""
polars-quant 类型提示文件
提供所有技术分析函数的类型注释和使用示例

主要功能模块：
1. 技术指标 (Technical Analysis) - 各种技术分析指标函数
2. 回测引擎 (Backtesting) - 高性能多线程回测系统

安装使用：
    pip install polars-quant

基本用法：
    import polars as pl
    import polars_quant as pq
    
    # 计算技术指标
    prices = pl.Series([1, 2, 3, 4, 5])
    sma_result = pq.sma(prices, 3)
    
    # 运行回测
    result = pq.Backtrade.run(data, entries, exits)
    result.summary()
"""

import polars as pl
from typing import Tuple, Optional, List

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
            benchmark: 基准指数数据DataFrame（可选），必须恰好有2列（日期和价格）
                格式: {"date": [...], "benchmark": [...]}
                所有股票将使用此基准进行对比分析
            
        Raises:
            ValueError: 当DataFrame列数不一致或prices列数少于2时
            ValueError: 当仓位大小不在(0.0, 1.0]范围内时
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

class StockSelector:
    """
    股票选择器 - 支持链式调用的股票筛选工具
    
    功能特点:
    - 从文件夹批量加载股票数据（支持 parquet/csv/xlsx/xls/json/feather/ipc）
    - 单一 filter() 方法支持 30+ 筛选参数
    - 支持链式调用进行多条件组合筛选
    - 内置技术指标计算（MA、RSI、MACD、KDJ 等）
    - 支持排序和取 TopN
    
    Examples:
        基本用法：
        >>> import polars_quant as pq
        >>> 
        >>> # 从文件夹加载数据
        >>> selector = pq.StockSelector.from_folder(
        ...     "data/stocks",
        ...     file_type="parquet",  # 可选: parquet/csv/xlsx/xls/json/feather/ipc
        ...     prefix="SH",          # 可选: 只加载以 SH 开头的文件
        ...     has_header=True       # CSV/Excel 是否有表头
        ... )
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
    
    @staticmethod
    def from_folder(
        folder: str,
        file_type: Optional[str | List[str]] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        has_header: bool = True
    ) -> 'StockSelector':
        """
        从文件夹批量加载股票数据
        
        Args:
            folder: 数据文件夹路径
            file_type: 文件类型，可选值: "parquet", "csv", "xlsx", "xls", "json", 
                      "feather", "ipc" 或列表。默认支持所有格式
            prefix: 文件名前缀过滤（可选）
            suffix: 文件名后缀过滤（可选）
            has_header: CSV/Excel 文件是否包含表头（默认 True）
        
        Returns:
            StockSelector 实例
        
        Raises:
            ValueError: 不是有效的目录
            ValueError: file_type 必须是字符串或字符串列表
            ValueError: 未找到匹配的文件
            ValueError: 文件缺少必需的列 (date/open/high/low/close/volume)
        
        Examples:
            >>> # 加载所有支持格式的文件
            >>> selector = pq.StockSelector.from_folder("data/stocks")
            >>> 
            >>> # 只加载 parquet 文件
            >>> selector = pq.StockSelector.from_folder("data/stocks", file_type="parquet")
            >>> 
            >>> # 只加载上海股票（SH 开头）
            >>> selector = pq.StockSelector.from_folder("data/stocks", prefix="SH")
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
    ) -> 'StockSelector':
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
    
    def reset(self) -> 'StockSelector':
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
    ) -> 'StockSelector':
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
