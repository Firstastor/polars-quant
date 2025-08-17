import polars as pl

class EMA:
    def __init__(
            self,

            ) -> None:
        pass
    
    def cross(
            self,
            name: str,
            first_ma: int,
            second_ma: int
            ):
        if first_ma in self.timeperiod and second_ma in self.timeperiod:
            data_first: pl.Series =  self.frame[name][f"{name}_ma{first_ma}"]
            data_second: pl.Series =  self.frame[name][f"{name}_ma{second_ma}"]
            return (data_first > data_second) & (data_second.shift(1) > data_first.shift(1))
        else:
            raise ValueError("Missing required timeperiod")
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        adjust: bool =False
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        results = {}

        for data_col, data_price in data_dict.items():
            ma_results = {}
            for window_size in timeperiod:
                data_price_ma = data_price.ewm_mean(span=window_size, adjust=adjust).alias(f"{data_col}_ema{window_size}")
                ma_results[window_size] = data_price_ma

            results[data_col] = pl.DataFrame(ma_results)

        cls.frame = results
        cls.timeperiod = timeperiod
        return cls()
    
class MA:
    def __init__(
            self,
            ) -> None:
        pass
    
    def cross(
            self,
            name: str,
            first_ma: int,
            second_ma: int
            ):
        if first_ma in self.timeperiod and second_ma in self.timeperiod:
            data_first: pl.Series =  self.frame[name][f"{name}_ma{first_ma}"]
            data_second: pl.Series =  self.frame[name][f"{name}_ma{second_ma}"]
            return (data_first > data_second) & (data_second.shift(1) > data_first.shift(1))
        else:
            raise ValueError("Missing required timeperiod")
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        results = {}

        for data_col, data_price in data_dict.items():
            ma_results = []
            for window_size in timeperiod:
                data_price_ma = data_price.rolling_mean(window_size).alias(f"{data_col}_ma{window_size}")
                ma_results.append(data_price_ma)
            results[data_col] = pl.DataFrame(ma_results)

        cls.frame = results
        cls.timeperiod = timeperiod
        return cls()
    
class SMA:
    def __init__(
            self,
            ) -> None:
        pass

    def cross(
            self,
            name: str,
            first_ma: int,
            second_ma: int
            ):
        if first_ma in self.timeperiod and second_ma in self.timeperiod:
            data_first: pl.Series =  self.frame[name][f"{name}_ma{first_ma}"]
            data_second: pl.Series =  self.frame[name][f"{name}_ma{second_ma}"]
            return (data_first > data_second) & (data_second.shift(1) > data_first.shift(1))
        else:
            raise ValueError("Missing required timeperiod")
          
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        results = {}

        for data_col, data_price in data_dict.items():
            ma_results = []
            for window_size in timeperiod:
                data_price_ma = data_price.rolling_mean(window_size).alias(f"{data_col}_sma{window_size}")
                ma_results.append(data_price_ma)
            results[data_col] = pl.DataFrame(ma_results)

        cls.frame = results
        cls.timeperiod = timeperiod
        return cls()