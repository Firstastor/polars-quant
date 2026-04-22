import pytest
from polars_quant import SequentialBacktester


def test_sequential_backtester_correctness(stock_data):
    """Test SequentialBacktester correctness."""
    df = stock_data.head(2000).clone()

    def strategy(ctx, i):
        if i > 20:
            last_price = df["close"][i]
            if last_price > 15.0:
                ctx.buy("sh.600000", 100, last_price)
            else:
                ctx.sell("sh.600000", 100, last_price)

    backtester = SequentialBacktester(
        historical_data=df,
        initial_capital=100000.0,
    )

    pos_df, cap_df, summary = backtester.run(strategy)

    assert cap_df.shape[0] == df.shape[0]
    assert "total_trades" in summary
    assert "annualized_return" in summary


@pytest.mark.benchmark(group="sequential")
def test_sequential_backtester_benchmark(benchmark, stock_data):
    """Test SequentialBacktester performance using pytest-benchmark."""
    # Use a smaller dataset slice to prevent the sequential benchmark from running excessively long
    df = stock_data.head(2000).clone()

    def strategy(ctx, i):
        if i > 20:
            last_price = df["close"][i]
            if last_price > 15.0:
                ctx.buy("sh.600000", 100, last_price)
            else:
                ctx.sell("sh.600000", 100, last_price)

    def run_backtester():
        backtester = SequentialBacktester(
            historical_data=df,
            initial_capital=100000.0,
        )
        return backtester.run(strategy)

    benchmark.pedantic(run_backtester, rounds=1, iterations=1)
