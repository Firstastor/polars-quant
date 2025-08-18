import polars as pl

class BBANDS:
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 20,
        nbdevup: float = 2, 
        nbdevdn: float = 2
        ):
        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod
        if len(data) > timeperiod:
            for data_col, data_price in cls.data_dict.items():
                middle = data_price.rolling_mean(timeperiod).alias(f"{data_col}_middle")
                upper = (middle + nbdevup * middle.std()).alias(f"{data_col}_upper")
                lower = (middle - nbdevdn * middle.std()).alias(f"{data_col}_lower")
                results[data_col] = cls.data_object.with_columns(middle,upper,lower)
        else:
            raise ValueError("the length of data is less than or equal timeperiod")
        cls.frame = results
        
        return cls()    
       
class EMA:
    def __init__(self) -> None:
        pass
    
    def cross(
            self,
            first_ma: int,
            second_ma: int
            ):
        results ={}
        for col in self.data_dict:
            if first_ma in self.timeperiod and second_ma in self.timeperiod:
                data_first: pl.Series =  self.frame[col][f"{col}_ema{first_ma}"]
                data_second: pl.Series =  self.frame[col][f"{col}_ema{second_ma}"]
                results[col] = self.data_object.with_columns(((data_first > data_second) & (data_second.shift(1) > data_first.shift(1)))
                                                             .alias(f"{col}_ema{first_ma}_cross_ema{second_ma}"))
            else:
                raise ValueError("Missing required timeperiod")
        return results
    
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        adjust: bool =False
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod

        for data_col, data_price in cls.data_dict.items():
            ma_results = []
            for window_size in timeperiod:
                data_price_ma = data_price.ewm_mean(span=window_size, adjust=adjust).alias(f"{data_col}_ema{window_size}")
                ma_results.append(data_price_ma)
            results[data_col] = cls.data_object.with_columns(ma_results)

        cls.frame = results
        return cls()
    
class MA:
    def __init__(self) -> None:
        pass
    
    def cross(
            self,
            first_ma: int,
            second_ma: int
            ):
        results ={}
        for col in self.data_dict:
            if first_ma in self.timeperiod and second_ma in self.timeperiod:
                data_first: pl.Series =  self.frame[col][f"{col}_ma{first_ma}"]
                data_second: pl.Series =  self.frame[col][f"{col}_ma{second_ma}"]
                results[col] = self.data_object.with_columns(((data_first > data_second) & (data_second.shift(1) > data_first.shift(1)))
                                                             .alias(f"{col}_ma{first_ma}_cross_ma{second_ma}"))
            else:
                raise ValueError("Missing required timeperiod")
        return results
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod

        for data_col, data_price in cls.data_dict.items():
            ma_results = []
            for window_size in timeperiod:
                data_price_ma = data_price.rolling_mean(window_size).alias(f"{data_col}_ma{window_size}")
                ma_results.append(data_price_ma)
            results[data_col] = cls.data_object.with_columns(ma_results)

        cls.frame = results
        
        return cls()
    
class MACD:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9):
        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())

        for data_col, data_price in cls.data_dict.items():
            fast_ma = data_price.rolling_mean(fastperiod).alias(f"{data_col}_fast")
            slow_ma = data_price.rolling_mean(slowperiod).alias(f"{data_col}_slow")
            dif = (fast_ma - slow_ma).alias(f"{data_col}_dif")
            dea = dif.ewm_mean(span=signalperiod).alias(f"{data_col}_dea")
            macd = (dif - dea).alias(f"{data_col}_macd")
            results[data_col] = cls.data_object.with_columns(dif,dea,macd)

        cls.frame = results
        
        return cls()

class RSI:
    def __init__(self):
        pass
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 14
        ):
        results = {}
        cls.data_dict = {col: data[[col]] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod

        for data_col, data_price in cls.data_dict.items():
            rsi = data_price.with_columns(
                delta=pl.col(data_col).diff(1),
                gain=pl.when(pl.col(data_col).diff(1) > 0).then(pl.col(data_col).diff(1)).otherwise(0),
                loss=pl.when(pl.col(data_col).diff(1) < 0).then(-pl.col(data_col).diff(1)).otherwise(0),
            ).with_columns(
                avg_gain=pl.col("gain").ewm_mean(span=timeperiod),
                avg_loss=pl.col("loss").ewm_mean(span=timeperiod),
            ).with_columns(
                rs=pl.col("avg_gain") / pl.col("avg_loss"),
                rsi=100 - (100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))),
            )
            results[data_col] = cls.data_object.with_columns(rsi)
        cls.frame = results
        
        return cls()
    
class SMA:
    def __init__(self) -> None:
        pass
    
    def cross(
            self,
            first_ma: int,
            second_ma: int
            ):
        results ={}
        for col in self.data_dict:
            if first_ma in self.timeperiod and second_ma in self.timeperiod:
                data_first: pl.Series =  self.frame[col][f"{col}_sma{first_ma}"]
                data_second: pl.Series =  self.frame[col][f"{col}_sma{second_ma}"]
                results[col] = self.data_object.with_columns(((data_first > data_second) & (data_second.shift(1) > data_first.shift(1)))
                                                             .alias(f"{col}_sma{first_ma}_cross_sma{second_ma}"))
            else:
                raise ValueError("Missing required timeperiod")
        return results
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod

        for data_col, data_price in cls.data_dict.items():
            ma_results = []
            for window_size in timeperiod:
                data_price_ma = data_price.rolling_mean(window_size).alias(f"{data_col}_sma{window_size}")
                ma_results.append(data_price_ma)
            results[data_col] = cls.data_object.with_columns(ma_results)

        cls.frame = results
        
        return cls()
    
class WMA:
    def __init__(self) -> None:
        pass
    
    def cross(
            self,
            first_ma: int,
            second_ma: int
            ):
        results ={}
        for col in self.data_dict:
            if first_ma in self.timeperiod and second_ma in self.timeperiod:
                data_first: pl.Series =  self.frame[col][f"{col}_wma{first_ma}"]
                data_second: pl.Series =  self.frame[col][f"{col}_wma{second_ma}"]
                results[col] = self.data_object.with_columns(((data_first > data_second) & (data_second.shift(1) > data_first.shift(1)))
                                                             .alias(f"{col}_wma{first_ma}_cross_wma{second_ma}"))
            else:
                raise ValueError("Missing required timeperiod")
        return results
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int | list[int],
        ):
        if isinstance(timeperiod, int):
            timeperiod = [timeperiod]

        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod

        for data_col, data_price in cls.data_dict.items():
            ma_results = []
            for window_size in timeperiod:
                weights = (pl.arange(window_size, 0, -1, eager=True) / (window_size * (window_size + 1) / 2))
                data_price_ma = data_price.rolling_sum(window_size, weights).alias(f"{data_col}_wma{window_size}")
                ma_results.append(data_price_ma)
            results[data_col] = cls.data_object.with_columns(ma_results)

        cls.frame = results

        return cls()
    
