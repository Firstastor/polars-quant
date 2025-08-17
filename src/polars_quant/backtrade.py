import polars as pl

class Backtrade():
    def __init__(self,
                 results: pl.DataFrame |None = None,
                 trades: pl.DataFrame |None = None
                 ):
        self._results = results
        self._trades = trades
        
    def _cal_summary(self):
        if self._results is None or self._trades is None:
            raise ValueError("No results or trades data available")

        summary = {}
        equity = self._results["Equity"]
        returns: pl.Series = equity.diff() / equity.shift(1)
        
        summary["initial_capital"] = equity[0]
        summary["final_equity"] = equity[-1]
        summary["total_return"] = (equity[-1] - equity[0]) / equity[0]
        summary["annualized_return"] = None
        
        if len(self._trades) > 0:
            trades = self._trades.filter(
                pl.col("sell_date").is_not_null() & pl.col("buy_date").is_not_null()
            )
            
            if len(trades) > 0:
                trades = trades.with_columns(
                    pl.col("sell_price").sub(pl.col("buy_price")).alias("profit"),
                    ((pl.col("sell_price") - pl.col("buy_price")) / pl.col("buy_price")).alias("return")
                )
                
                summary["total_trades"] = len(trades)
                summary["winning_trades"] = len(trades.filter(pl.col("profit") > 0))
                summary["losing_trades"] = len(trades.filter(pl.col("profit") <= 0))
                summary["win_rate"] = summary["winning_trades"] / summary["total_trades"]
                
                avg_profit = trades.filter(pl.col("profit") > 0)["profit"].mean()
                avg_loss = trades.filter(pl.col("profit") < 0)["profit"].mean()
                summary["profit_factor"] = abs(avg_profit / avg_loss) if (avg_loss != 0) & (avg_loss !=None) else float("inf")
                
                summary["avg_trade_return"] = trades["return"].mean()
                summary["max_drawdown"] = (1 - (equity / equity.cum_max())).max()
                
        dates = self._results["Date"].cast(pl.Date)
        if len(dates) > 1:
            days = (dates[-1] - dates[0]).days
            years = days / 365.25
            
            if years > 0:
                summary["annualized_return"] = (1 + summary["total_return"]) ** (1/years) - 1
                summary["sharpe_ratio"] = returns.mean() / returns.std() * (252 ** 0.5)
        
        self._summary = summary

    @staticmethod
    def _buy_order(
                    symbol: str,
                    date: pl.Date,
                    price: float,
                    cash: float,
                    fee: float,
                    slip: float,
                    size: float
                   ):
        price = (1 + slip) * price
        max_share = (1 - fee) * cash / price // 100 * 100 
        buy_share = int(max_share * size)
        cash -= buy_share * price
        trade = {
            "symbol": symbol,
            "buy_date": date,
            "buy_price": price,
            "share": buy_share
        }
        return cash, buy_share, trade



    @staticmethod
    def _sell_order(
                    symbol: str,
                    date: pl.Date,
                    position: int,
                    price: float,
                    cash: float,
                    fee: float,
                    slip: float
                   ):
        price = (1 - fee) * (1 - slip) * price
        cash += price * position   
        trade = {
            "symbol": symbol,
            "sell_date": date,
            "sell_price": price,
        }

        return cash, trade
    
    @classmethod
    def run(
            cls,
            data: pl.DataFrame,
            entries: pl.DataFrame,
            exits: pl.DataFrame,
            init_cash: float = 100000.0,
            fee: float = 0.0,
            slip: float = 0.0,
            size: float = 1.0
            ):
        h = data.height
        w = data.width
        data = data.fill_null(0.0)
        date = data[:, 0]
        results: list[dict] = []
        trades: list[dict] = []
        trades_index = 0
        for width in range(1,w):
            symbol = data.columns[width]
            price = data[:, width]
            entry = entries[:, width].fill_null(False)
            exit = exits[:, width].fill_null(False)
            cash = init_cash
            position: int = 0

            for height in range(h):
                today = date[height]
                today_price = price[height]
                if entry[height] & (position==0):
                    cash, position, trade = cls._buy_order(symbol, today, today_price, cash, fee, slip, size)
                    trades.append(trade)
                    trades_index = len(trades) - 1
                elif exit[height] & (position>0):
                    cash, trade = cls._sell_order(symbol, today, position, today_price, cash, fee, slip)
                    position = 0
                    trades[trades_index].update(trade)
                
                result = {
                    "Symbol": symbol,
                    "Date": today,
                    "Cash": cash,
                    "Equity": cash + today_price * position
                }
                results.append(result)

        return cls(pl.DataFrame(results), pl.DataFrame(trades))

    def plot(self):
        pass

    def results(self):
        print(self._results)
        return self
    
    def summary(self):
        if not hasattr(self, "_summary"):
            self._cal_summary()
        
        s = self._summary
        print("Summary")
        print("="*40)
        print(f"{'Initial Capital:':<20} {s.get('initial_capital', 0):,.2f}")
        print(f"{'Final Equity:':<20} {s.get('final_equity', 0):,.2f}")
        print(f"{'Total Return:':<20} {s.get('total_return', 0):.2%}")
        print(f"{'Annualized Return:':<20} {s.get('annualized_return', 0):.2%}")
        print("\nTrade Analysis")
        print("-"*40)
        print(f"{'Total Trades:':<20} {s.get('total_trades', 0)}")
        print(f"{'Winning Trades:':<20} {s.get('winning_trades', 0)}")
        print(f"{'Losing Trades:':<20} {s.get('losing_trades', 0)}")
        print(f"{'Win Rate:':<20} {s.get('win_rate', 0):.2%}")
        print(f"{'Profit Factor:':<20} {s.get('profit_factor', 0):.2f}")
        print(f"{'Avg Trade Return:':<20} {s.get('avg_trade_return', 0):.2%}")
        print("\nRisk Metrics")
        print("-"*40)
        print(f"{'Max Drawdown:':<20} {s.get('max_drawdown', 0):.2%}")
        print(f"{'Sharpe Ratio:':<20} {s.get('sharpe_ratio', 0):.2f}")
        print("="*40)
        
        return self

    def trades(self):
        print(self._trades)
        return self
    



