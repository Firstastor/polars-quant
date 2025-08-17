import polars as pl

class Backtrade():
    def __init__(self,
                 results: pl.DataFrame |None = None,
                 trades: pl.DataFrame |None = None
                 ):
        self.results = results
        self.trades = trades

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
            "buy_data": date,
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
            "sell_data": date,
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

    def get_results(self):
        print(self.results)
        return self
    
    def get_trades(self):
        print(self.trades)
        return self
    




