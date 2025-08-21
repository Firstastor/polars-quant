import polars as pl

class ATR:
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 14,
    ):
        results = {}
        cls.data_dict = {col: data[[col]] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())

        high_col = next((col for col in data.columns if col.lower() in ["high", "h"]), None)
        low_col = next((col for col in data.columns if col.lower() in ["low", "l"]), None)
        close_col = next((col for col in data.columns if col.lower() in ["close", "c"]), None)

        if not all([high_col, low_col, close_col]):
            raise ValueError("Could not find required columns (High, Low, Close) in the data")

        tr = pl.max_horizontal(
            data[high_col] - data[low_col],
            (data[high_col] - data[close_col].shift(1)).abs(),
            (data[low_col] - data[close_col].shift(1)).abs()
        )
        atr = tr.rolling_mean(timeperiod)

        results["ATR"] = cls.data_object.with_columns(atr.alias("ATR"))
        cls.frame = results

        return cls()

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
                std = data_price.rolling_std(timeperiod)
                upper = (middle + nbdevup * std).alias(f"{data_col}_upper")
                lower = (middle - nbdevdn * std).alias(f"{data_col}_lower")
                results[data_col] = cls.data_object.with_columns(middle,upper,lower)
        else:
            raise ValueError("the length of data is less than or equal timeperiod")
        cls.frame = results
        
        return cls()    
       
class CCI:
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 20
        ):
        results = {}
        cls.data_dict = {col: data[col] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod
        high_col = next((col for col in data.columns if col.lower() == 'high' or col.lower() == 'h'), None)
        low_col = next((col for col in data.columns if col.lower() == 'low' or col.lower() == 'l'), None)
        close_col = next((col for col in data.columns if col.lower() == 'close' or col.lower() == 'c'), None)
        
        if not all([high_col, low_col, close_col]):
            raise ValueError("Could not find required columns (High, Low, Close) in the data")
        typ = ((data[high_col] + data[low_col] + data[close_col]) / 3).to_series()
        sma = typ.rolling_mean(timeperiod)
        mean_dev = (typ - sma).abs().rolling_mean(timeperiod)
        cci = (typ - sma) / (0.015 * mean_dev)
        results["CCI"] = cls.data_object.with_columns(cci.alias("CCI"))
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
        results: dict[str, pl.DataFrame] ={}
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
    
class KAMA:
    def __init__(self):
        pass
        
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 14,
        fast_period: int = 2,
        slow_period: int = 30
        ):
        results = {}
        cls.data_dict = {col: data[[col]] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        cls.timeperiod = timeperiod
        cls.fast_period = fast_period
        cls.slow_period = slow_period

        for data_col, data_price in cls.data_dict.items():
            change = pl.col(data_col).diff(timeperiod).abs()
            volatility = pl.col(data_col).diff(1).abs().rolling_sum(timeperiod)
            
            er = (change / volatility).alias("er")
            
            fast_sc = 2 / (fast_period + 1)
            slow_sc = 2 / (slow_period + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc).pow(2).alias("sc")
            
            kama = data_price.with_columns(
                change=change,
                volatility=volatility,
                er=er,
                sc=sc
            ).with_columns(
                (pl.col(data_col).shift(1) + pl.col("sc") * (pl.col(data_col) - pl.col(data_col).shift(1))).alias(f"{data_col}_kama")
            )[f"{data_col}_kama"]
            
            results[data_col] = cls.data_object.with_columns(kama)
            
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
        results = self.data_object
        for idx, col in enumerate(self.data_dict):
            if first_ma in self.timeperiod and second_ma in self.timeperiod:
                data_first: pl.Series =  self.frame[col][f"{col}_ma{first_ma}"]
                data_second: pl.Series =  self.frame[col][f"{col}_ma{second_ma}"]
                results = results.with_columns(((data_first > data_second) & (data_second.shift(1) > data_first.shift(1)))
                                                             .alias(f"{idx}"))
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

class NATR:
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 14,
    ):
        atr_result = ATR.run(data, timeperiod)
        atr: pl.Series = atr_result.frame["ATR"]["ATR"]
        close_col = next((col for col in data.columns if col.lower() in ["close", "c"]), None)
        if not close_col:
            raise ValueError("Close price column not found")
        natr = (atr / data.get_column(close_col)) * 100

        results = {}
        cls.data_object = atr_result.data_object
        results["NATR"] = cls.data_object.with_columns(natr.alias("NATR"))
        cls.frame = results
        return cls()
    
class OBV:
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        volume_col: str = "Volume",
        price_col: str = "Close",
    ):
        results = {}
        cls.data_dict = {col: data[[col]] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())
        price_col = next((col for col in data.columns if col.lower() == price_col.lower()), None)
        volume_col = next((col for col in data.columns if col.lower() == volume_col.lower()), None)

        if not all([price_col, volume_col]):
            raise ValueError("Could not find required columns (Price, Volume) in the data")

        price_diff = data[price_col].diff()
        obv = (
            pl.when(price_diff > 0)
            .then(data[volume_col])
            .when(price_diff < 0)
            .then(-data[volume_col])
            .otherwise(0)
        ).cum_sum()

        results["OBV"] = cls.data_object.with_columns(obv.alias("OBV"))
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
                (100 - (100 / (1 + pl.col("avg_gain") / pl.col("avg_loss")))).alias(f"{data_col}_rsi"),
            )[f"{data_col}_rsi"]
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
        results: dict[str, pl.DataFrame] ={}
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

class TEMA:
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
            ema1 = data_price.with_columns(
                ema1=pl.col(data_col).ewm_mean(span=timeperiod)
            )["ema1"]
            
            ema2 = ema1.to_frame().with_columns(
                ema2=pl.col("ema1").ewm_mean(span=timeperiod)
            )["ema2"]
            
            ema3 = ema2.to_frame().with_columns(
                ema3=pl.col("ema2").ewm_mean(span=timeperiod)
            )["ema3"]
            
            tema = (3 * ema1 - 3 * ema2 + ema3).alias(f"{data_col}_tema")
            results[data_col] = cls.data_object.with_columns(tema)
            
        cls.frame = results
        
        return cls()  

class TRANGE:
    """Calculates True Range (TRANGE)"""
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
    ):
        results = {}
        cls.data_dict = {col: data[[col]] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())

        high_col = next((col for col in data.columns if col.lower() in ["high", "h"]), None)
        low_col = next((col for col in data.columns if col.lower() in ["low", "l"]), None)
        close_col = next((col for col in data.columns if col.lower() in ["close", "c"]), None)
        if not all([high_col, low_col, close_col]):
            raise ValueError("High/Low/Close columns not found")

        tr1 = data[high_col] - data[low_col]
        tr2 = (data[high_col] - data[close_col].shift(1)).abs()
        tr3 = (data[low_col] - data[close_col].shift(1)).abs()
        tr = pl.max_horizontal(tr1, tr2, tr3)

        results["TRANGE"] = cls.data_object.with_columns(tr.alias("TRANGE"))
        cls.frame = results
        return cls()

class TRIMA:
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
            window_size = (timeperiod + 1) // 2
            sma1 = data_price.with_columns(
                sma1=pl.col(data_col).rolling_mean(window_size=window_size)
            )["sma1"]
            trima = sma1.to_frame().with_columns(
                trima=pl.col("sma1").rolling_mean(window_size=window_size)
            )[f"{data_col}_trima"]
            
            results[data_col] = cls.data_object.with_columns(trima)
            
        cls.frame = results
        
        return cls()

class VOL:
    def __init__(self):
        pass

    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        timeperiod: int = 20,
        method: str = "std",  # Options: "std" (standard deviation), "range" (high-low)
    ):
        results = {}
        cls.data_dict = {col: data[[col]] for col in data.columns if data[col].dtype.is_numeric()}
        cls.data_object = data.select(col for col in data.columns if not data[col].dtype.is_numeric())

        price_col = next((col for col in data.columns if col.lower() in ["close", "c"]), None)
        if not price_col:
            raise ValueError("Could not find price column (Close) in the data")

        if method == "std":
            volatility = data[price_col].pct_change().rolling_std(timeperiod)
        elif method == "range":
            high_col = next((col for col in data.columns if col.lower() in ["high", "h"]), None)
            low_col = next((col for col in data.columns if col.lower() in ["low", "l"]), None)
            if not all([high_col, low_col]):
                raise ValueError("Could not find High/Low columns for range volatility")
            volatility = (data.get_column(high_col) - data.get_column(low_col)).rolling_mean(timeperiod)
        else:
            raise ValueError("Invalid method. Choose 'std' or 'range'")

        results["Volatility"] = cls.data_object.with_columns(volatility.alias("Volatility"))
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
        results: dict[str, pl.DataFrame] ={}
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
    
