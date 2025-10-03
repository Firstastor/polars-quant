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
# 回测引擎 (Backtesting Engine) - 高性能多线程回测系统
# ====================================================================

class Backtrade:
    """
    高性能量化回测引擎
    
    特点:
    - 支持多线程并行处理
    - 向量化计算优化
    - 支持多股票组合回测
    - 完整的交易记录和绩效统计
    
    Examples:
        基本回测示例：
        >>> import polars as pl
        >>> import polars_quant as pq
        >>> 
        >>> # 准备价格数据 (Date, AAPL, GOOGL, ...)
        >>> data = pl.DataFrame({
        ...     "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        ...     "AAPL": [150.0, 152.0, 148.0],
        ...     "GOOGL": [2800.0, 2850.0, 2820.0]
        ... })
        >>> 
        >>> # 准备入场信号 (True表示买入信号)
        >>> entries = pl.DataFrame({
        ...     "AAPL": [True, False, False],
        ...     "GOOGL": [False, True, False]
        ... })
        >>> 
        >>> # 准备出场信号 (True表示卖出信号)
        >>> exits = pl.DataFrame({
        ...     "AAPL": [False, False, True],
        ...     "GOOGL": [False, False, True]
        ... })
        >>> 
        >>> # 运行回测
        >>> bt = pq.Backtrade.run(
        ...     data=data,
        ...     entries=entries, 
        ...     exits=exits,
        ...     init_cash=100000.0,  # 初始资金10万
        ...     fee=0.001            # 手续费率0.1%
        ... )
        >>> 
        >>> # 查看回测结果
        >>> bt.summary()             # 打印详细报告
        >>> returns = bt.total_return()    # 获取总收益率
        >>> max_dd = bt.max_drawdown()     # 获取最大回撤
        >>> equity = bt.equity_curve()     # 获取权益曲线
        >>> trades_df = bt.trades_df()     # 获取交易记录
        >>> daily_df = bt.daily_records_df()  # 获取每日资金记录
        
        高级配置示例：
        >>> # 对于大量股票，自动启用多线程并行处理
        >>> bt = pq.Backtrade.run(
        ...     data=large_universe_data,  # 包含100+只股票的数据
        ...     entries=entry_signals,
        ...     exits=exit_signals,
        ...     init_cash=1000000.0,       # 初始资金100万
        ...     fee=0.0005                 # 手续费率0.05%
        ... )
        >>> # 系统会自动检测股票数量并启用并行处理
    """
    
    @staticmethod
    def run(
        data: pl.DataFrame,
        entries: pl.DataFrame, 
        exits: pl.DataFrame,
        init_cash: float = 100000.0,
        fee: float = 0.001
    ) -> 'Backtrade':
        """
        运行回测
        
        Args:
            data: 价格数据DataFrame，必须包含Date列和各股票价格列
                格式: {"Date": [...], "STOCK1": [...], "STOCK2": [...]}
            entries: 入场信号DataFrame，布尔值矩阵
                格式: {"STOCK1": [True/False, ...], "STOCK2": [True/False, ...]}
            exits: 出场信号DataFrame，布尔值矩阵  
                格式: {"STOCK1": [True/False, ...], "STOCK2": [True/False, ...]}
            init_cash: 初始资金，默认10万
            fee: 手续费率，默认0.1%
            
        Returns:
            Backtrade实例，包含所有回测结果
            
        Note:
            - 当股票数量≥10时自动启用多线程并行处理
            - 系统会自动进行向量化优化以提升计算性能
            - 资金会均等分配给所有股票进行回测
        """
        ...
    
    def summary(self) -> None:
        """
        打印回测摘要报告
        
        包含内容：
        - 回测性能统计（耗时、处理速度）
        - 收益率统计
        - 风险指标（最大回撤等）
        - 交易统计（次数、胜率等）
        - 手续费统计
        """
        ...
    
    def equity_curve(self) -> List[float]:
        """
        获取权益曲线
        
        Returns:
            每个时间点的账户总权益值列表
            
        Example:
            >>> equity = bt.equity_curve()
            >>> # 可用于绘制权益曲线图
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(equity)
            >>> plt.title("账户权益曲线")
            >>> plt.show()
        """
        ...
    
    def total_return(self) -> float:
        """
        获取总收益率（百分比）
        
        Returns:
            总收益率百分比，如15.5表示15.5%的收益
        """
        ...
    
    def max_drawdown(self) -> float:
        """
        获取最大回撤（百分比）
        
        Returns:
            最大回撤百分比，如-8.5表示最大回撤8.5%
        """
        ...
    
    def num_trades(self) -> int:
        """
        获取交易次数
        
        Returns:
            总交易次数（买入+卖出）
        """
        ...
    
    def win_rate(self) -> float:
        """
        获取胜率（百分比）
        
        Returns:
            胜率百分比，如65.0表示65%的胜率
        """
        ...
    
    def trades_df(self) -> pl.DataFrame:
        """
        获取详细交易记录DataFrame
        
        Returns:
            包含所有交易记录的DataFrame，列包括：
            - symbol: 股票代码
            - side: 交易方向 (BUY/SELL)
            - quantity: 交易数量
            - price: 交易价格
            - timestamp: 交易时间戳
            - fees: 手续费
            - pnl: 该笔交易盈亏
            
        Example:
            >>> trades = bt.trades_df()
            >>> print(trades.head())
            >>> # 筛选盈利交易
            >>> profitable = trades.filter(pl.col("pnl") > 0)
            >>> # 按股票分组统计
            >>> by_symbol = trades.group_by("symbol").agg([
            ...     pl.col("pnl").sum().alias("total_pnl"),
            ...     pl.col("pnl").count().alias("trade_count")
            ... ])
        """
        ...
    
    def daily_records_df(self) -> pl.DataFrame:
        """
        获取每日资金记录DataFrame
        
        Returns:
            包含每日账户状态的DataFrame，列包括：
            - date: 日期
            - cash: 现金余额
            - equity: 总权益
            - unrealized_pnl: 未实现盈亏
            - realized_pnl: 已实现盈亏
            - daily_pnl: 当日盈亏
            
        Example:
            >>> daily = bt.daily_records_df()
            >>> print(daily.head())
            >>> # 计算每日收益率
            >>> daily_returns = daily.with_columns([
            ...     (pl.col("equity").pct_change() * 100).alias("daily_return_pct")
            ... ])
            >>> # 计算夏普比率等风险指标
            >>> sharpe = daily_returns["daily_return_pct"].mean() / daily_returns["daily_return_pct"].std()
        """
        ...
    
    def backtest_duration(self) -> float:
        """
        获取回测耗时（毫秒）
        
        Returns:
            回测总耗时（毫秒）
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

def ma(series: pl.Series, period: int) -> pl.Series:
    """
    通用移动平均线 (Generic Moving Average)
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        MA序列
    
    Example:
        >>> prices = pl.Series([1, 2, 3, 4, 5, 6])
        >>> result = pq.ma(prices, 5)
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