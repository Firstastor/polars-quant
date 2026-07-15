import sys
import timeit

import numpy as np
import polars as pl
import polars_quant as pq

import talib

SEED = 42
SIZE = 10_000
RTOL = 1e-8
ATOL = 1e-8
WARMUP = 20
REPEATS = 50

rng = np.random.default_rng(SEED)
open_price = rng.uniform(10, 50, SIZE)
high_price = rng.uniform(10, 50, SIZE)
low_price = rng.uniform(10, 50, SIZE)
close_price = rng.uniform(10, 50, SIZE)
volume = rng.uniform(1000, 10000, SIZE)
for i in range(SIZE):
    sorted_prices = sorted([open_price[i], high_price[i], low_price[i], close_price[i]])
    low_price[i], open_price[i], close_price[i], high_price[i] = sorted_prices
data_frame = pl.DataFrame(
    {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
    }
)


def compare(pq_values, tl_values, label):
    pq_nan = int(np.isnan(pq_values).sum())
    tl_nan = int(np.isnan(tl_values).sum())
    if pq_nan != tl_nan:
        return f"  FAIL {label}: NaN mismatch PQ={pq_nan} TL={tl_nan}"
    valid = ~np.isnan(pq_values) & ~np.isnan(tl_values)
    if valid.sum() == 0:
        return f"  SKIP {label}: no non-NaN values"
    diff = np.abs(pq_values[valid] - tl_values[valid])
    max_diff = diff.max()
    if max_diff < max(RTOL, ATOL):
        return f"  PASS {label}: max_diff={max_diff:.2e}"
    return f"  FAIL {label}: max_diff={max_diff:.2e} mean_diff={diff.mean():.2e}"


def run_single(name, pq_expr, tl_callable, data_frame):
    pq_result = data_frame.with_columns(pq_expr.alias("x"))["x"].to_numpy()
    tl_result = np.asarray(tl_callable()).ravel()
    return compare(pq_result, tl_result, name)


modules = {}

modules["overlap"] = {}

bbands_upper, bbands_middle, bbands_lower = pq.BBANDS(
    pl.col("close"), timeperiod=5, nbdevup=2.0, nbdevdn=2.0
)

for entry_name, pq_val, tl_val in [
    (
        "BBANDS_lower",
        bbands_lower,
        lambda: np.asarray(
            talib.BBANDS(
                data_frame["close"].to_numpy(), timeperiod=5, nbdevup=2.0, nbdevdn=2.0
            )[2]
        ).ravel(),
    ),
    (
        "BBANDS_middle",
        bbands_middle,
        lambda: np.asarray(
            talib.BBANDS(
                data_frame["close"].to_numpy(), timeperiod=5, nbdevup=2.0, nbdevdn=2.0
            )[1]
        ).ravel(),
    ),
    (
        "BBANDS_upper",
        bbands_upper,
        lambda: np.asarray(
            talib.BBANDS(
                data_frame["close"].to_numpy(), timeperiod=5, nbdevup=2.0, nbdevdn=2.0
            )[0]
        ).ravel(),
    ),
]:
    modules["overlap"][entry_name] = (pq_val, tl_val)

for entry_name, pq_val, tl_val in [
    (
        "DEMA",
        pq.DEMA(pl.col("close"), timeperiod=30),
        lambda: talib.DEMA(data_frame["close"].to_numpy(), timeperiod=30),
    ),
    (
        "EMA",
        pq.EMA(pl.col("close"), timeperiod=30),
        lambda: talib.EMA(data_frame["close"].to_numpy(), timeperiod=30),
    ),
    (
        "MIDPOINT",
        pq.MIDPOINT(pl.col("close"), timeperiod=14),
        lambda: talib.MIDPOINT(data_frame["close"].to_numpy(), timeperiod=14),
    ),
    (
        "MIDPRICE",
        pq.MIDPRICE(pl.col("high"), pl.col("low"), timeperiod=14),
        lambda: talib.MIDPRICE(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy(), timeperiod=14
        ),
    ),
    (
        "SMA",
        pq.SMA(pl.col("close"), timeperiod=30),
        lambda: talib.SMA(data_frame["close"].to_numpy(), timeperiod=30),
    ),
    (
        "T3",
        pq.T3(pl.col("close"), timeperiod=5, vfactor=0.7),
        lambda: talib.T3(data_frame["close"].to_numpy(), timeperiod=5, vfactor=0.7),
    ),
    (
        "TEMA",
        pq.TEMA(pl.col("close"), timeperiod=30),
        lambda: talib.TEMA(data_frame["close"].to_numpy(), timeperiod=30),
    ),
    (
        "TRIMA",
        pq.TRIMA(pl.col("close"), timeperiod=30),
        lambda: talib.TRIMA(data_frame["close"].to_numpy(), timeperiod=30),
    ),
    (
        "WMA",
        pq.WMA(pl.col("close"), timeperiod=30),
        lambda: talib.WMA(data_frame["close"].to_numpy(), timeperiod=30),
    ),
]:
    modules["overlap"][entry_name] = (pq_val, tl_val)


modules["price"] = {}
for entry_name, pq_val, tl_val in [
    (
        "AVGPRICE",
        pq.AVGPRICE(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")),
        lambda: talib.AVGPRICE(
            data_frame["open"].to_numpy(),
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
        ),
    ),
    (
        "MEDPRICE",
        pq.MEDPRICE(pl.col("high"), pl.col("low")),
        lambda: talib.MEDPRICE(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy()
        ),
    ),
    (
        "TYPPRICE",
        pq.TYPPRICE(pl.col("high"), pl.col("low"), pl.col("close")),
        lambda: talib.TYPPRICE(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
        ),
    ),
    (
        "WCLPRICE",
        pq.WCLPRICE(pl.col("high"), pl.col("low"), pl.col("close")),
        lambda: talib.WCLPRICE(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
        ),
    ),
]:
    modules["price"][entry_name] = (pq_val, tl_val)


modules["volatility"] = {}
for entry_name, pq_val, tl_val in [
    (
        "ATR",
        pq.ATR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.ATR(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "NATR",
        pq.NATR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.NATR(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "TRANGE",
        pq.TRANGE(pl.col("high"), pl.col("low"), pl.col("close")),
        lambda: talib.TRANGE(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
        ),
    ),
]:
    modules["volatility"][entry_name] = (pq_val, tl_val)


modules["volume"] = {}
for entry_name, pq_val, tl_val in [
    (
        "AD",
        pq.AD(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")),
        lambda: talib.AD(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            data_frame["volume"].to_numpy(),
        ),
    ),
    (
        "ADOSC",
        pq.ADOSC(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            fastperiod=3,
            slowperiod=10,
        ),
        lambda: talib.ADOSC(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            data_frame["volume"].to_numpy(),
            fastperiod=3,
            slowperiod=10,
        ),
    ),
    (
        "OBV",
        pq.OBV(pl.col("close"), pl.col("volume")),
        lambda: talib.OBV(
            data_frame["close"].to_numpy(), data_frame["volume"].to_numpy()
        ),
    ),
]:
    modules["volume"][entry_name] = (pq_val, tl_val)


modules["momentum"] = {}
for entry_name, pq_val, tl_val in [
    (
        "ADX",
        pq.ADX(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.ADX(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "ADXR",
        pq.ADXR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.ADXR(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "APO",
        pq.APO(pl.col("close"), fastperiod=12, slowperiod=26, matype=0),
        lambda: talib.APO(
            data_frame["close"].to_numpy(), fastperiod=12, slowperiod=26, matype=0
        ),
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

aroon_up, aroon_down = pq.AROON(pl.col("high"), pl.col("low"), timeperiod=14)
for entry_name, pq_val, tl_val in [
    (
        "AROON_down",
        aroon_down,
        lambda: talib.AROON(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy(), timeperiod=14
        )[1],
    ),
    (
        "AROON_up",
        aroon_up,
        lambda: talib.AROON(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy(), timeperiod=14
        )[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

for entry_name, pq_val, tl_val in [
    (
        "AROONOSC",
        pq.AROONOSC(pl.col("high"), pl.col("low"), timeperiod=14),
        lambda: talib.AROONOSC(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy(), timeperiod=14
        ),
    ),
    (
        "BOP",
        pq.BOP(pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")),
        lambda: talib.BOP(
            data_frame["open"].to_numpy(),
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
        ),
    ),
    (
        "CCI",
        pq.CCI(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.CCI(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "CMO",
        pq.CMO(pl.col("close"), timeperiod=14),
        lambda: talib.CMO(data_frame["close"].to_numpy(), timeperiod=14),
    ),
    (
        "DX",
        pq.DX(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.DX(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

macd_line, macd_signal, macd_hist = pq.MACD(
    pl.col("close"), fastperiod=12, slowperiod=26, signalperiod=9
)
for entry_name, pq_val, tl_val in [
    (
        "MACD_hist",
        macd_hist,
        lambda: talib.MACD(
            data_frame["close"].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9
        )[2],
    ),
    (
        "MACD_signal",
        macd_signal,
        lambda: talib.MACD(
            data_frame["close"].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9
        )[1],
    ),
    (
        "MACD",
        macd_line,
        lambda: talib.MACD(
            data_frame["close"].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9
        )[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

macd_dif, macd_dea, macd_hist2 = pq.MACDEXT(
    pl.col("close"),
    fastperiod=12,
    fastmatype=0,
    slowperiod=26,
    slowmatype=0,
    signalperiod=9,
    signalmatype=0,
)
for entry_name, pq_val, tl_val in [
    (
        "MACDEXT_hist",
        macd_hist2,
        lambda: talib.MACDEXT(
            data_frame["close"].to_numpy(),
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0,
        )[2],
    ),
    (
        "MACDEXT_signal",
        macd_dea,
        lambda: talib.MACDEXT(
            data_frame["close"].to_numpy(),
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0,
        )[1],
    ),
    (
        "MACDEXT",
        macd_dif,
        lambda: talib.MACDEXT(
            data_frame["close"].to_numpy(),
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0,
        )[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

macdfix_line, macdfix_signal, macdfix_hist = pq.MACDFIX(pl.col("close"), signalperiod=9)
for entry_name, pq_val, tl_val in [
    (
        "MACDFIX_hist",
        macdfix_hist,
        lambda: talib.MACDFIX(data_frame["close"].to_numpy(), signalperiod=9)[2],
    ),
    (
        "MACDFIX_signal",
        macdfix_signal,
        lambda: talib.MACDFIX(data_frame["close"].to_numpy(), signalperiod=9)[1],
    ),
    (
        "MACDFIX",
        macdfix_line,
        lambda: talib.MACDFIX(data_frame["close"].to_numpy(), signalperiod=9)[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

for entry_name, pq_val, tl_val in [
    (
        "MFI",
        pq.MFI(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            timeperiod=14,
        ),
        lambda: talib.MFI(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            data_frame["volume"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "MINUS_DI",
        pq.MINUS_DI(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.MINUS_DI(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "MINUS_DM",
        pq.MINUS_DM(pl.col("high"), pl.col("low"), timeperiod=14),
        lambda: talib.MINUS_DM(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy(), timeperiod=14
        ),
    ),
    (
        "MOM",
        pq.MOM(pl.col("close"), timeperiod=10),
        lambda: talib.MOM(data_frame["close"].to_numpy(), timeperiod=10),
    ),
    (
        "PLUS_DI",
        pq.PLUS_DI(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.PLUS_DI(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
    (
        "PLUS_DM",
        pq.PLUS_DM(pl.col("high"), pl.col("low"), timeperiod=14),
        lambda: talib.PLUS_DM(
            data_frame["high"].to_numpy(), data_frame["low"].to_numpy(), timeperiod=14
        ),
    ),
    (
        "PPO",
        pq.PPO(pl.col("close"), fastperiod=12, slowperiod=26, matype=0),
        lambda: talib.PPO(
            data_frame["close"].to_numpy(), fastperiod=12, slowperiod=26, matype=0
        ),
    ),
    (
        "ROC",
        pq.ROC(pl.col("close"), timeperiod=10),
        lambda: talib.ROC(data_frame["close"].to_numpy(), timeperiod=10),
    ),
    (
        "ROCP",
        pq.ROCP(pl.col("close"), timeperiod=10),
        lambda: talib.ROCP(data_frame["close"].to_numpy(), timeperiod=10),
    ),
    (
        "ROCR",
        pq.ROCR(pl.col("close"), timeperiod=10),
        lambda: talib.ROCR(data_frame["close"].to_numpy(), timeperiod=10),
    ),
    (
        "ROCR100",
        pq.ROCR100(pl.col("close"), timeperiod=10),
        lambda: talib.ROCR100(data_frame["close"].to_numpy(), timeperiod=10),
    ),
    (
        "RSI",
        pq.RSI(pl.col("close"), timeperiod=14),
        lambda: talib.RSI(data_frame["close"].to_numpy(), timeperiod=14),
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

stoch_slowk, stoch_slowd = pq.STOCH(
    pl.col("high"),
    pl.col("low"),
    pl.col("close"),
    fastk_period=5,
    slowk_period=3,
    slowk_matype=0,
    slowd_period=3,
    slowd_matype=0,
)
for entry_name, pq_val, tl_val in [
    (
        "STOCH_slowd",
        stoch_slowd,
        lambda: talib.STOCH(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )[1],
    ),
    (
        "STOCH_slowk",
        stoch_slowk,
        lambda: talib.STOCH(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

stochf_fastk, stochf_fastd = pq.STOCHF(
    pl.col("high"),
    pl.col("low"),
    pl.col("close"),
    fastk_period=5,
    fastd_period=3,
    fastd_matype=0,
)
for entry_name, pq_val, tl_val in [
    (
        "STOCHF_fastd",
        stochf_fastd,
        lambda: talib.STOCHF(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0,
        )[1],
    ),
    (
        "STOCHF_fastk",
        stochf_fastk,
        lambda: talib.STOCHF(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0,
        )[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

stochrsi_fastk, stochrsi_fastd = pq.STOCHRSI(
    pl.col("close"), timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
)
for entry_name, pq_val, tl_val in [
    (
        "STOCHRSI_fastd",
        stochrsi_fastd,
        lambda: talib.STOCHRSI(
            data_frame["close"].to_numpy(),
            timeperiod=14,
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0,
        )[1],
    ),
    (
        "STOCHRSI_fastk",
        stochrsi_fastk,
        lambda: talib.STOCHRSI(
            data_frame["close"].to_numpy(),
            timeperiod=14,
            fastk_period=5,
            fastd_period=3,
            fastd_matype=0,
        )[0],
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)

for entry_name, pq_val, tl_val in [
    (
        "TRIX",
        pq.TRIX(pl.col("close"), timeperiod=30),
        lambda: talib.TRIX(data_frame["close"].to_numpy(), timeperiod=30),
    ),
    (
        "ULTOSC",
        pq.ULTOSC(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28,
        ),
        lambda: talib.ULTOSC(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28,
        ),
    ),
    (
        "WILLR",
        pq.WILLR(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=14),
        lambda: talib.WILLR(
            data_frame["high"].to_numpy(),
            data_frame["low"].to_numpy(),
            data_frame["close"].to_numpy(),
            timeperiod=14,
        ),
    ),
]:
    modules["momentum"][entry_name] = (pq_val, tl_val)


modules_order = ["overlap", "price", "volatility", "volume", "momentum"]

selected_modules = modules_order
if len(sys.argv) > 1:
    selected_modules = [arg.lower() for arg in sys.argv[1:]]
    unknown = [m for m in selected_modules if m not in modules]
    if unknown:
        print(f"Unknown modules: {unknown}")
        print(f"Available: {', '.join(modules_order)}")
        sys.exit(1)


print()
total_pq_time = 0.0
total_tl_time = 0.0
total_functions = 0

for module_name in modules_order:
    if module_name not in selected_modules:
        continue

    current_module = modules[module_name]
    sorted_names = sorted(current_module.keys())

    bar = "=" * 74
    print(bar)
    print(f"  {module_name.title()}")
    print(bar)

    for name in sorted_names:
        pq_expr, tl_callable = current_module[name]
        status = run_single(name, pq_expr, tl_callable, data_frame)
        print(status)

    print()
    print(f"  {module_name.title()} speed (ms):")

    batch_exprs = [
        expr.alias(name) for name, (expr, _) in sorted(current_module.items())
    ]
    for _ in range(WARMUP):
        data_frame.with_columns(batch_exprs)
    pq_time = (
        timeit.timeit(lambda: data_frame.with_columns(batch_exprs), number=REPEATS)
        / REPEATS
        * 1000
    )

    tl_fns = [tl for _, (_, tl) in sorted(current_module.items())]
    for _ in range(WARMUP):
        for fn in tl_fns:
            fn()
    tl_time = (
        timeit.timeit(lambda: [fn() for fn in tl_fns], number=REPEATS) / REPEATS * 1000
    )

    ratio = tl_time / pq_time
    n = len(sorted_names)
    print(f"  polars_quant (batch): {pq_time:.3f} ms")
    print(f"  TA-Lib ({n} calls):     {tl_time:.3f} ms")
    print(f"  ratio:                {ratio:.1f}x")
    print()

    total_pq_time += pq_time
    total_tl_time += tl_time
    total_functions += n

print(bar)
print("  Total")
print(bar)
print(f"  Functions: {total_functions}")
print(f"  polars_quant (batch, {len(selected_modules)} groups): {total_pq_time:.3f} ms")
print(f"  TA-Lib ({total_functions} calls):                    {total_tl_time:.3f} ms")
print(
    f"  ratio:                                       {total_tl_time / total_pq_time:.1f}x"
)
