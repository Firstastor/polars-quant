import polars as pl
import pytest
from polars_quant import VectorizedBacktester
from polars_quant.talib import overlap as pq_ta


def setup_vector_data(stock_data):
    """Prepare signals for the vector backtester."""
    df = stock_data.clone()

    # Generate simple signals: SMA crossover
    sma_20 = pq_ta.SMA(df["close"], 20)
    sma_50 = pq_ta.SMA(df["close"], 50)

    df = df.with_columns([sma_20.alias("sma20"), sma_50.alias("sma50")])

    # Buy when SMA20 > SMA50, Sell when SMA20 < SMA50
    df = df.with_columns(
        [
            (pl.col("sma20") > pl.col("sma50")).alias("buy"),
            (pl.col("sma20") < pl.col("sma50")).alias("sell"),
        ]
    )

    # Fill nulls in signals to prevent missing boolean values
    df = df.with_columns(
        [pl.col("buy").fill_null(False), pl.col("sell").fill_null(False)]
    )
    return df


def test_vector_backtester_correctness(stock_data):
    """Test VectorBacktester correctness."""
    df = setup_vector_data(stock_data)

    backtester = VectorizedBacktester(
        price=df["close"],
        buy_signal=df["buy"],
        sell_signal=df["sell"],
        initial_capital=100000.0,
        position_size=1.0,
    )

    pos_df, cap_df, summary = backtester.run()

    assert pos_df.shape[0] == df.shape[0]
    assert cap_df.shape[0] == df.shape[0]
    assert "annualized_return" in summary
    assert "max_drawdown" in summary
    assert "total_trades" in summary


@pytest.mark.benchmark(group="vector")
def test_vector_backtester_benchmark(benchmark, stock_data):
    """Test VectorBacktester performance using pytest-benchmark."""
    df = setup_vector_data(stock_data)

    def run_backtester():
        backtester = VectorizedBacktester(
            price=df["close"],
            buy_signal=df["buy"],
            sell_signal=df["sell"],
            initial_capital=100000.0,
            position_size=1.0,
        )
        return backtester.run()

    # Run the benchmark
    benchmark.pedantic(run_backtester, rounds=1, iterations=1)
