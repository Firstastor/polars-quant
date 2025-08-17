import polars as pl

class MA:
    def __init__(
            self,

            ) -> None:
        pass
    
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        type: str = "SMA"
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        results = {}

        for data_col, data_price in data_dict.items():
            ma_results = {}

            for window_size in timeperiod:
                if type == "SMA":
                    data_price_ma = data_price.rolling_mean(window_size).alias(f"{data_col}_sma{window_size}")
                elif type == "EMA":
                    data_price_ma = data_price.ewm_mean(span=window_size, adjust=False).alias(f"{data_col}_ema{window_size}")
                else:
                    raise ValueError("Error Type")    
                ma_results[window_size] = data_price_ma

            results[data_col] = pl.DataFrame(ma_results)

        cls.frame = results

        return cls