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

