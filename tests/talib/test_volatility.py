import polars_quant as pq
import pytest

import talib
from tests.talib_test_utils import (
    calculate_max_diff,
    get_function_inputs,
    get_module_talib_functions,
)

funcs = get_module_talib_functions("volatility")


@pytest.mark.parametrize("func_name", funcs)
def test_volatility_parity_with_talib(func_name, stock_data):
    if not hasattr(talib, func_name):
        pytest.skip(f"{func_name} not available in TA-Lib")

    pq_func = getattr(pq, func_name)
    ta_func = getattr(talib, func_name)

    pq_args = get_function_inputs(func_name, stock_data, mode="expr")
    ta_args = get_function_inputs(func_name, stock_data, mode="numpy")

    ta_res = ta_func(*ta_args)

    try:
        expr = pq_func(*pq_args)
        pq_df = stock_data.lazy().select(expr).collect()
        pq_res = pq_df.to_series(0).to_numpy()
    except Exception as exc:
        pytest.fail(f"Polars-Quant execution failed for {func_name}: {exc}")

    if isinstance(ta_res, tuple):
        assert isinstance(pq_res, tuple)
        assert len(ta_res) == len(pq_res)
    else:
        assert not isinstance(pq_res, tuple)

    diff = calculate_max_diff(ta_res, pq_res)

    assert diff <= 1e-4
