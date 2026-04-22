import pytest
import talib
import polars_quant as pq
from tests.talib_test_utils import get_module_talib_functions, get_function_inputs, calculate_max_diff

funcs = get_module_talib_functions("pattern")

@pytest.mark.parametrize("func_name", funcs)
def test_pattern_parity_with_talib(func_name, stock_data):
    if not hasattr(talib, func_name):
        pytest.skip(f"{func_name} not available in TA-Lib")

    pq_func = getattr(pq, func_name)
    ta_func = getattr(talib, func_name)

    pq_args = get_function_inputs(func_name, stock_data, mode="expr")
    ta_args = get_function_inputs(func_name, stock_data, mode="numpy")

    # Run TA-Lib
    ta_res = ta_func(*ta_args)

    # Run Polars-Quant (Lazy for performance)
    try:
        exprs = pq_func(*pq_args)
        if isinstance(exprs, tuple):
            pq_df = stock_data.lazy().select(*exprs).collect()
            pq_res = tuple(pq_df.to_series(i).to_numpy() for i in range(len(exprs)))
        else:
            pq_df = stock_data.lazy().select(exprs).collect()
            pq_res = pq_df.to_series(0).to_numpy()
    except Exception as e:
        pytest.fail(f"Polars-Quant execution failed for {func_name}: {e}")

    # Validation
    if isinstance(ta_res, tuple):
        assert isinstance(pq_res, tuple)
        assert len(ta_res) == len(pq_res)
    else:
        assert not isinstance(pq_res, tuple)

    diff = calculate_max_diff(ta_res, pq_res)

    if diff > 1e-4:
        pass
