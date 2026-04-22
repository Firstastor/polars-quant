# ruff: noqa
import inspect
import polars_quant as pq
import polars as pl
import numpy as np

def get_module_talib_functions(module_name):
    skip_funcs = ["MAVP", "SAR", "SAREXT"]
    module = getattr(pq.talib, module_name)
    funcs = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name.isupper():
            if name not in skip_funcs:
                funcs.append(name)
    return sorted(funcs)

def get_function_inputs(func_name, stock_data, mode="series"):
    if "CDL" in func_name:
        if mode == "series": return [stock_data["open"], stock_data["high"], stock_data["low"], stock_data["close"]]
        elif mode == "expr": return [pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")]
        else: return [stock_data["open"].to_numpy(), stock_data["high"].to_numpy(), stock_data["low"].to_numpy(), stock_data["close"].to_numpy()]
    # get module
    func = getattr(pq, func_name)
    sig = inspect.signature(func)

    args = []
    for param_name in sig.parameters:
        col_name = None
        if param_name in ["real", "close"]:
            col_name = "close"
        elif param_name in ["open", "o"]:
            col_name = "open"
        elif param_name in ["high", "h"]:
            col_name = "high"
        elif param_name in ["low", "l"]:
            col_name = "low"
        elif param_name == "volume":
            col_name = "volume"

        if col_name:
            if mode == "series":
                args.append(stock_data[col_name])
            elif mode == "expr":
                args.append(pl.col(col_name))
            elif mode == "numpy":
                args.append(stock_data[col_name].to_numpy())
        elif param_name == "periods":
            periods_np = np.random.randint(2, 30, size=len(stock_data)).astype(np.float64)
            if mode == "series":
                args.append(pl.Series("periods", periods_np))
            elif mode == "expr":
                args.append(pl.lit(pl.Series("periods", periods_np)))
            elif mode == "numpy":
                args.append(periods_np)

    return args

def calculate_max_diff(res_ta, res_pq):
    if isinstance(res_ta, tuple):
        max_diff = 0.0
        for i in range(len(res_ta)):
            rt = res_ta[i]
            rp = res_pq[i]
            mask = ~np.isnan(rt) & ~np.isnan(rp)
            if mask.any():
                diff = np.max(np.abs(rt[mask] - rp[mask]))
                max_diff = max(max_diff, diff)
        return max_diff
    else:
        mask = ~np.isnan(res_ta) & ~np.isnan(res_pq)
        if mask.any():
            return np.max(np.abs(res_ta[mask] - res_pq[mask]))
        return 0.0

def get_all_talib_functions():
    funcs = []
    for name, obj in inspect.getmembers(pq):
        if inspect.isfunction(obj) and name.isupper():
            if name not in skip_funcs:
                funcs.append(name)
    return sorted(funcs)
