import numpy as np
import talib

data = np.random.random(100)
a = talib.NATR(data, timeperiod=30)