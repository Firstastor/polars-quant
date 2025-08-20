from typing import Optional
import polars as pl

class Backtrade:
    results: Optional[pl.DataFrame]
    trades: Optional[pl.DataFrame]
    summary: Optional[dict]

    def __init__(self, results: Optional[pl.DataFrame] = None, trades: Optional[pl.DataFrame] = None) -> None: ...
    
    @classmethod
    def run(
        cls,
        data: pl.DataFrame,
        entries: pl.DataFrame,
        exits: pl.DataFrame,
        init_cash: float,
        fee: float,
        slip: float,
        size: float,
    ) -> "Backtrade": ...
    
    def results(self) -> Optional[pl.DataFrame]: ...
    
    def trades(self) -> Optional[pl.DataFrame]: ...
    
    def summary(self) -> str: ...
