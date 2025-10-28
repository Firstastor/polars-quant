use pyo3::prelude::*;

mod talib;
mod backtest;
mod selector;
mod strategy;
mod factor;

#[pymodule]
fn polars_quant(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ====================================================================
    // 回测模块 (Backtesting)
    // ====================================================================
    m.add_class::<backtest::Backtest>()?;
    
    // ====================================================================
    // 选股模块 (Stock Selection)
    // ====================================================================
    m.add_class::<selector::Selector>()?;
    
    // ====================================================================
    // 策略模块 (Trading Strategies)
    // ====================================================================
    m.add_class::<strategy::Strategy>()?;
    
    // ====================================================================
    // 因子模块 (Factor Mining & Evaluation)
    // ====================================================================
    m.add_class::<factor::Factor>()?;

    
    // ====================================================================
    // 重叠研究指标 (Overlap Studies) - 移动平均线及相关指标
    // ====================================================================
    
    // 布林带
    m.add_function(wrap_pyfunction!(talib::bband, m)?)?;
    
    // 移动平均线系列
    m.add_function(wrap_pyfunction!(talib::sma, m)?)?;        // 简单移动平均
    m.add_function(wrap_pyfunction!(talib::ema, m)?)?;        // 指数移动平均
    m.add_function(wrap_pyfunction!(talib::wma, m)?)?;        // 加权移动平均
    m.add_function(wrap_pyfunction!(talib::dema, m)?)?;       // 双指数移动平均
    m.add_function(wrap_pyfunction!(talib::tema, m)?)?;       // 三指数移动平均
    m.add_function(wrap_pyfunction!(talib::trima, m)?)?;      // 三角移动平均
    m.add_function(wrap_pyfunction!(talib::kama, m)?)?;       // 考夫曼自适应移动平均
    m.add_function(wrap_pyfunction!(talib::mama, m)?)?;       // 母线自适应移动平均
    m.add_function(wrap_pyfunction!(talib::t3, m)?)?;         // T3移动平均
    m.add_function(wrap_pyfunction!(talib::ma, m)?)?;         // 通用移动平均
    m.add_function(wrap_pyfunction!(talib::mavp, m)?)?;       // 可变周期移动平均
    
    // 价格位置指标
    m.add_function(wrap_pyfunction!(talib::midpoint, m)?)?;   // 中点价格
    m.add_function(wrap_pyfunction!(talib::midprice_hl, m)?)?; // 最高最低价中点
    
    // 抛物线SAR系统
    m.add_function(wrap_pyfunction!(talib::sar, m)?)?;        // 抛物线SAR
    m.add_function(wrap_pyfunction!(talib::sarext, m)?)?;     // 抛物线SAR扩展版

    // ====================================================================
    // 动量指标 (Momentum Indicators) - 趋势强度和方向指标
    // ====================================================================
    
    // 趋向系统指标
    m.add_function(wrap_pyfunction!(talib::adx, m)?)?;        // 平均趋向指标
    m.add_function(wrap_pyfunction!(talib::adxr, m)?)?;       // 平均趋向指标评级
    m.add_function(wrap_pyfunction!(talib::dx, m)?)?;         // 方向性指标
    m.add_function(wrap_pyfunction!(talib::plus_di, m)?)?;    // 正方向指标
    m.add_function(wrap_pyfunction!(talib::minus_di, m)?)?;   // 负方向指标
    m.add_function(wrap_pyfunction!(talib::plus_dm, m)?)?;    // 正方向移动
    m.add_function(wrap_pyfunction!(talib::minus_dm, m)?)?;   // 负方向移动
    
    // MACD系统
    m.add_function(wrap_pyfunction!(talib::macd, m)?)?;       // MACD指标
    m.add_function(wrap_pyfunction!(talib::macdext, m)?)?;    // MACD扩展版
    m.add_function(wrap_pyfunction!(talib::macdfix, m)?)?;    // MACD固定参数版
    
    // 摆动指标
    m.add_function(wrap_pyfunction!(talib::rsi, m)?)?;        // 相对强弱指标
    m.add_function(wrap_pyfunction!(talib::stoch, m)?)?;      // 随机指标
    m.add_function(wrap_pyfunction!(talib::stochf, m)?)?;     // 快速随机指标
    m.add_function(wrap_pyfunction!(talib::stochrsi, m)?)?;   // RSI随机指标
    m.add_function(wrap_pyfunction!(talib::willr, m)?)?;      // 威廉指标
    m.add_function(wrap_pyfunction!(talib::ultosc, m)?)?;     // 终极摆动指标
    
    // Aroon系统
    m.add_function(wrap_pyfunction!(talib::aroon, m)?)?;      // 阿隆指标
    m.add_function(wrap_pyfunction!(talib::aroonosc, m)?)?;   // 阿隆摆动指标
    
    // 其他动量指标
    m.add_function(wrap_pyfunction!(talib::apo, m)?)?;        // 绝对价格摆动指标
    m.add_function(wrap_pyfunction!(talib::ppo, m)?)?;        // 价格摆动百分比
    m.add_function(wrap_pyfunction!(talib::bop, m)?)?;        // 均势指标
    m.add_function(wrap_pyfunction!(talib::cci, m)?)?;        // 顺势指标
    m.add_function(wrap_pyfunction!(talib::cmo, m)?)?;        // 钱德动量摆动指标
    m.add_function(wrap_pyfunction!(talib::mfi, m)?)?;        // 资金流量指标
    m.add_function(wrap_pyfunction!(talib::mom, m)?)?;        // 动量指标
    m.add_function(wrap_pyfunction!(talib::trix, m)?)?;       // TRIX指标
    
    // 变化率指标
    m.add_function(wrap_pyfunction!(talib::roc, m)?)?;        // 变化率
    m.add_function(wrap_pyfunction!(talib::rocp, m)?)?;       // 变化率百分比
    m.add_function(wrap_pyfunction!(talib::rocr, m)?)?;       // 变化率比率
    m.add_function(wrap_pyfunction!(talib::rocr100, m)?)?;    // 变化率比率100

    // ====================================================================
    // 成交量指标(Volume Indicators) - 成交量相关指标
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(talib::ad, m)?)?;         // 累积/派发线
    m.add_function(wrap_pyfunction!(talib::adosc, m)?)?;      // 累积/派发摆动指标
    m.add_function(wrap_pyfunction!(talib::obv, m)?)?;        // 能量潮指标
    
    // ====================================================================
    // 波动率指标 (Volatility Indicators) - 价格波动测量
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(talib::atr, m)?)?;        // 平均真实波幅
    m.add_function(wrap_pyfunction!(talib::natr, m)?)?;       // 标准化平均真实波幅
    m.add_function(wrap_pyfunction!(talib::trange, m)?)?;     // 真实波幅
    
    // ====================================================================
    // 价格变换函数 (Price Transform) - 价格数据变换
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(talib::avgprice, m)?)?;   // 平均价格
    m.add_function(wrap_pyfunction!(talib::medprice, m)?)?;   // 中位价格
    m.add_function(wrap_pyfunction!(talib::typprice, m)?)?;   // 典型价格
    m.add_function(wrap_pyfunction!(talib::wclprice, m)?)?;   // 加权收盘价格
    
    // ====================================================================
    // 周期指标 (Cycle Indicators) - 希尔伯特变换系列
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(talib::ht_trendline, m)?)?;  // 希尔伯特变换-趋势线
    m.add_function(wrap_pyfunction!(talib::ht_dcperiod, m)?)?;   // 希尔伯特变换-主导周期
    m.add_function(wrap_pyfunction!(talib::ht_dcphase, m)?)?;    // 希尔伯特变换-主导周期相位
    m.add_function(wrap_pyfunction!(talib::ht_phasor, m)?)?;     // 希尔伯特变换-相量分量
    m.add_function(wrap_pyfunction!(talib::ht_sine, m)?)?;       // 希尔伯特变换-正弦波
    m.add_function(wrap_pyfunction!(talib::ht_trendmode, m)?)?;  // 希尔伯特变换-趋势模式
    
    // ====================================================================
    // 蜡烛图模式识别 (Candlestick Pattern Recognition) - K线形态
    // ====================================================================
    
    // 基本反转形态
    m.add_function(wrap_pyfunction!(talib::cdldoji, m)?)?;           // 十字星
    m.add_function(wrap_pyfunction!(talib::cdldojistar, m)?)?;       // 十字星
    m.add_function(wrap_pyfunction!(talib::cdldragonflydoji, m)?)?;  // 蜻蜓十字
    m.add_function(wrap_pyfunction!(talib::cdlgravestonedoji, m)?)?; // 墓石十字
    m.add_function(wrap_pyfunction!(talib::cdllongleggeddoji, m)?)?; // 长腿十字
    
    // 锤头线系列
    m.add_function(wrap_pyfunction!(talib::cdlhammer, m)?)?;         // 锤头线
    m.add_function(wrap_pyfunction!(talib::cdlhangingman, m)?)?;     // 上吊线
    m.add_function(wrap_pyfunction!(talib::cdlinvertedhammer, m)?)?; // 倒锤头线
    m.add_function(wrap_pyfunction!(talib::cdlshootingstar, m)?)?;   // 射击之星
    
    // 吞没和孕线形态
    m.add_function(wrap_pyfunction!(talib::cdlengulfing, m)?)?;      // 吞没形态
    m.add_function(wrap_pyfunction!(talib::cdlharami, m)?)?;         // 孕线形态
    m.add_function(wrap_pyfunction!(talib::cdlharamicross, m)?)?;    // 孕线十字
    
    // 星形态系列
    m.add_function(wrap_pyfunction!(talib::cdlmorningstar, m)?)?;        // 早晨之星
    m.add_function(wrap_pyfunction!(talib::cdlmorningdojistar, m)?)?;    // 早晨十字星
    m.add_function(wrap_pyfunction!(talib::cdleveningstar, m)?)?;        // 黄昏之星
    m.add_function(wrap_pyfunction!(talib::cdleveningdojistar, m)?)?;    // 黄昏十字星
    m.add_function(wrap_pyfunction!(talib::cdlabandonedbaby, m)?)?;      // 弃婴形态
    m.add_function(wrap_pyfunction!(talib::cdltristar, m)?)?;            // 三星形态
    
    // 穿透形态
    m.add_function(wrap_pyfunction!(talib::cdlpiercing, m)?)?;           // 穿透形态
    m.add_function(wrap_pyfunction!(talib::cdldarkcloudcover, m)?)?;     // 乌云盖顶
    
    // 三根K线形态
    m.add_function(wrap_pyfunction!(talib::cdl3blackcrows, m)?)?;        // 三只乌鸦
    m.add_function(wrap_pyfunction!(talib::cdl3whitesoldiers, m)?)?;     // 三个白武士
    m.add_function(wrap_pyfunction!(talib::cdl3inside, m)?)?;            // 三内部上涨下降
    m.add_function(wrap_pyfunction!(talib::cdl3outside, m)?)?;           // 三外部上涨下降
    m.add_function(wrap_pyfunction!(talib::cdl3linestrike, m)?)?;        // 三线攻击
    m.add_function(wrap_pyfunction!(talib::cdl3starsinsouth, m)?)?;      // 南方三星
    m.add_function(wrap_pyfunction!(talib::cdlidentical3crows, m)?)?;    // 相同三乌鸦
    
    // 跳空形态
    m.add_function(wrap_pyfunction!(talib::cdlgapsidesidewhite, m)?)?;   // 向上跳空并列阳线
    m.add_function(wrap_pyfunction!(talib::cdlupsidegap2crows, m)?)?;    // 向上跳空两只乌鸦
    m.add_function(wrap_pyfunction!(talib::cdltasukigap, m)?)?;          // 跳空并列阴阳线
    m.add_function(wrap_pyfunction!(talib::cdlxsidegap3methods, m)?)?;   // 上升/下降跳空三法
    
    // 特殊形态
    m.add_function(wrap_pyfunction!(talib::cdl2crows, m)?)?;             // 两只乌鸦
    m.add_function(wrap_pyfunction!(talib::cdladvanceblock, m)?)?;       // 前进阻挡
    m.add_function(wrap_pyfunction!(talib::cdlbelthold, m)?)?;           // 捉腰带线
    m.add_function(wrap_pyfunction!(talib::cdlbreakaway, m)?)?;          // 脱离形态
    m.add_function(wrap_pyfunction!(talib::cdlclosingmarubozu, m)?)?;    // 收盘秃头
    m.add_function(wrap_pyfunction!(talib::cdlconcealbabyswall, m)?)?;   // 藏婴吞没
    m.add_function(wrap_pyfunction!(talib::cdlcounterattack, m)?)?;      // 反击线
    m.add_function(wrap_pyfunction!(talib::cdlhighwave, m)?)?;           // 长影线
    m.add_function(wrap_pyfunction!(talib::cdlhikkake, m)?)?;            // 陷阱形态
    m.add_function(wrap_pyfunction!(talib::cdlhikkakemod, m)?)?;         // 修正陷阱形态
    m.add_function(wrap_pyfunction!(talib::cdlhomingpigeon, m)?)?;       // 家鸽形态
    m.add_function(wrap_pyfunction!(talib::cdlinneck, m)?)?;             // 颈内线
    m.add_function(wrap_pyfunction!(talib::cdlkicking, m)?)?;            // 反冲形态
    m.add_function(wrap_pyfunction!(talib::cdlkickingbylength, m)?)?;    // 由长度决定的反冲形态
    m.add_function(wrap_pyfunction!(talib::cdlladderbottom, m)?)?;       // 梯底形态
    m.add_function(wrap_pyfunction!(talib::cdllongline, m)?)?;           // 长线形态
    m.add_function(wrap_pyfunction!(talib::cdlmarubozu, m)?)?;           // 秃头形态
    m.add_function(wrap_pyfunction!(talib::cdlmatchinglow, m)?)?;        // 相同低价
    m.add_function(wrap_pyfunction!(talib::cdlmathold, m)?)?;            // 铺垫形态
    m.add_function(wrap_pyfunction!(talib::cdlonneck, m)?)?;             // 颈上线
    m.add_function(wrap_pyfunction!(talib::cdlrickshawman, m)?)?;        // 黄包车夫
    m.add_function(wrap_pyfunction!(talib::cdlrisefall3methods, m)?)?;   // 上升/下降三法
    m.add_function(wrap_pyfunction!(talib::cdlseparatinglines, m)?)?;    // 分离线
    m.add_function(wrap_pyfunction!(talib::cdlshortline, m)?)?;          // 短线形态
    m.add_function(wrap_pyfunction!(talib::cdlspinningtop, m)?)?;        // 纺锤形态
    m.add_function(wrap_pyfunction!(talib::cdlstalledpattern, m)?)?;     // 停顿形态
    m.add_function(wrap_pyfunction!(talib::cdlsticksandwich, m)?)?;      // 条形三明治
    m.add_function(wrap_pyfunction!(talib::cdltakuri, m)?)?;             // 探水竿
    m.add_function(wrap_pyfunction!(talib::cdlthrusting, m)?)?;          // 插入形态
    m.add_function(wrap_pyfunction!(talib::cdlunique3river, m)?)?;       // 奇特三河床
    
    
    Ok(())
}


