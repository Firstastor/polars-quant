use pyo3::prelude::*;

mod qbacktrade;
mod qtalib;

#[pymodule]
fn polars_quant(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ====================================================================
    // 重叠研究指标 (Overlap Studies) - 移动平均线及相关指标
    // ====================================================================
    
    // 布林带
    m.add_function(wrap_pyfunction!(qtalib::bband, m)?)?;
    
    // 移动平均线系列
    m.add_function(wrap_pyfunction!(qtalib::sma, m)?)?;        // 简单移动平均
    m.add_function(wrap_pyfunction!(qtalib::ema, m)?)?;        // 指数移动平均
    m.add_function(wrap_pyfunction!(qtalib::wma, m)?)?;        // 加权移动平均
    m.add_function(wrap_pyfunction!(qtalib::dema, m)?)?;       // 双指数移动平均
    m.add_function(wrap_pyfunction!(qtalib::tema, m)?)?;       // 三指数移动平均
    m.add_function(wrap_pyfunction!(qtalib::trima, m)?)?;      // 三角移动平均
    m.add_function(wrap_pyfunction!(qtalib::kama, m)?)?;       // 考夫曼自适应移动平均
    m.add_function(wrap_pyfunction!(qtalib::mama, m)?)?;       // 母线自适应移动平均
    m.add_function(wrap_pyfunction!(qtalib::t3, m)?)?;         // T3移动平均
    m.add_function(wrap_pyfunction!(qtalib::ma, m)?)?;         // 通用移动平均
    m.add_function(wrap_pyfunction!(qtalib::mavp, m)?)?;       // 可变周期移动平均
    
    // 价格位置指标
    m.add_function(wrap_pyfunction!(qtalib::midpoint, m)?)?;   // 中点价格
    m.add_function(wrap_pyfunction!(qtalib::midprice_hl, m)?)?; // 最高最低价中点
    
    // 抛物线SAR系统
    m.add_function(wrap_pyfunction!(qtalib::sar, m)?)?;        // 抛物线SAR
    m.add_function(wrap_pyfunction!(qtalib::sarext, m)?)?;     // 抛物线SAR扩展版

    // ====================================================================
    // 动量指标 (Momentum Indicators) - 趋势强度和方向指标
    // ====================================================================
    
    // 趋向系统指标
    m.add_function(wrap_pyfunction!(qtalib::adx, m)?)?;        // 平均趋向指标
    m.add_function(wrap_pyfunction!(qtalib::adxr, m)?)?;       // 平均趋向指标评级
    m.add_function(wrap_pyfunction!(qtalib::dx, m)?)?;         // 方向性指标
    m.add_function(wrap_pyfunction!(qtalib::plus_di, m)?)?;    // 正方向指标
    m.add_function(wrap_pyfunction!(qtalib::minus_di, m)?)?;   // 负方向指标
    m.add_function(wrap_pyfunction!(qtalib::plus_dm, m)?)?;    // 正方向移动
    m.add_function(wrap_pyfunction!(qtalib::minus_dm, m)?)?;   // 负方向移动
    
    // MACD系统
    m.add_function(wrap_pyfunction!(qtalib::macd, m)?)?;       // MACD指标
    m.add_function(wrap_pyfunction!(qtalib::macdext, m)?)?;    // MACD扩展版
    m.add_function(wrap_pyfunction!(qtalib::macdfix, m)?)?;    // MACD固定参数版
    
    // 摆动指标
    m.add_function(wrap_pyfunction!(qtalib::rsi, m)?)?;        // 相对强弱指标
    m.add_function(wrap_pyfunction!(qtalib::stoch, m)?)?;      // 随机指标
    m.add_function(wrap_pyfunction!(qtalib::stochf, m)?)?;     // 快速随机指标
    m.add_function(wrap_pyfunction!(qtalib::stochrsi, m)?)?;   // RSI随机指标
    m.add_function(wrap_pyfunction!(qtalib::willr, m)?)?;      // 威廉指标
    m.add_function(wrap_pyfunction!(qtalib::ultosc, m)?)?;     // 终极摆动指标
    
    // Aroon系统
    m.add_function(wrap_pyfunction!(qtalib::aroon, m)?)?;      // 阿隆指标
    m.add_function(wrap_pyfunction!(qtalib::aroonosc, m)?)?;   // 阿隆摆动指标
    
    // 其他动量指标
    m.add_function(wrap_pyfunction!(qtalib::apo, m)?)?;        // 绝对价格摆动指标
    m.add_function(wrap_pyfunction!(qtalib::ppo, m)?)?;        // 价格摆动百分比
    m.add_function(wrap_pyfunction!(qtalib::bop, m)?)?;        // 均势指标
    m.add_function(wrap_pyfunction!(qtalib::cci, m)?)?;        // 顺势指标
    m.add_function(wrap_pyfunction!(qtalib::cmo, m)?)?;        // 钱德动量摆动指标
    m.add_function(wrap_pyfunction!(qtalib::mfi, m)?)?;        // 资金流量指标
    m.add_function(wrap_pyfunction!(qtalib::mom, m)?)?;        // 动量指标
    m.add_function(wrap_pyfunction!(qtalib::trix, m)?)?;       // TRIX指标
    
    // 变化率指标
    m.add_function(wrap_pyfunction!(qtalib::roc, m)?)?;        // 变化率
    m.add_function(wrap_pyfunction!(qtalib::rocp, m)?)?;       // 变化率百分比
    m.add_function(wrap_pyfunction!(qtalib::rocr, m)?)?;       // 变化率比率
    m.add_function(wrap_pyfunction!(qtalib::rocr100, m)?)?;    // 变化率比率*100

    // ====================================================================
    // 成交量指标 (Volume Indicators) - 成交量相关指标
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(qtalib::ad, m)?)?;         // 累积/派发线
    m.add_function(wrap_pyfunction!(qtalib::adosc, m)?)?;      // 累积/派发摆动指标
    m.add_function(wrap_pyfunction!(qtalib::obv, m)?)?;        // 能量潮指标
    
    // ====================================================================
    // 波动率指标 (Volatility Indicators) - 价格波动测量
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(qtalib::atr, m)?)?;        // 平均真实波幅
    m.add_function(wrap_pyfunction!(qtalib::natr, m)?)?;       // 标准化平均真实波幅
    m.add_function(wrap_pyfunction!(qtalib::trange, m)?)?;     // 真实波幅
    
    // ====================================================================
    // 价格变换函数 (Price Transform) - 价格数据变换
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(qtalib::avgprice, m)?)?;   // 平均价格
    m.add_function(wrap_pyfunction!(qtalib::medprice, m)?)?;   // 中位价格
    m.add_function(wrap_pyfunction!(qtalib::typprice, m)?)?;   // 典型价格
    m.add_function(wrap_pyfunction!(qtalib::wclprice, m)?)?;   // 加权收盘价格
    
    // ====================================================================
    // 周期指标 (Cycle Indicators) - 希尔伯特变换系列
    // ====================================================================
    
    m.add_function(wrap_pyfunction!(qtalib::ht_trendline, m)?)?;  // 希尔伯特变换-趋势线
    m.add_function(wrap_pyfunction!(qtalib::ht_dcperiod, m)?)?;   // 希尔伯特变换-主导周期
    m.add_function(wrap_pyfunction!(qtalib::ht_dcphase, m)?)?;    // 希尔伯特变换-主导周期相位
    m.add_function(wrap_pyfunction!(qtalib::ht_phasor, m)?)?;     // 希尔伯特变换-相量分量
    m.add_function(wrap_pyfunction!(qtalib::ht_sine, m)?)?;       // 希尔伯特变换-正弦波
    m.add_function(wrap_pyfunction!(qtalib::ht_trendmode, m)?)?;  // 希尔伯特变换-趋势模式
    
    // ====================================================================
    // 蜡烛图模式识别 (Candlestick Pattern Recognition) - K线形态
    // ====================================================================
    
    // 基本反转形态
    m.add_function(wrap_pyfunction!(qtalib::cdldoji, m)?)?;           // 十字星
    m.add_function(wrap_pyfunction!(qtalib::cdldojistar, m)?)?;       // 十字星
    m.add_function(wrap_pyfunction!(qtalib::cdldragonflydoji, m)?)?;  // 蜻蜓十字
    m.add_function(wrap_pyfunction!(qtalib::cdlgravestonedoji, m)?)?; // 墓石十字
    m.add_function(wrap_pyfunction!(qtalib::cdllongleggeddoji, m)?)?; // 长腿十字
    
    // 锤头线系列
    m.add_function(wrap_pyfunction!(qtalib::cdlhammer, m)?)?;         // 锤头线
    m.add_function(wrap_pyfunction!(qtalib::cdlhangingman, m)?)?;     // 上吊线
    m.add_function(wrap_pyfunction!(qtalib::cdlinvertedhammer, m)?)?; // 倒锤头线
    m.add_function(wrap_pyfunction!(qtalib::cdlshootingstar, m)?)?;   // 射击之星
    
    // 吞没和孕线形态
    m.add_function(wrap_pyfunction!(qtalib::cdlengulfing, m)?)?;      // 吞没形态
    m.add_function(wrap_pyfunction!(qtalib::cdlharami, m)?)?;         // 孕线形态
    m.add_function(wrap_pyfunction!(qtalib::cdlharamicross, m)?)?;    // 孕线十字
    
    // 星形态系列
    m.add_function(wrap_pyfunction!(qtalib::cdlmorningstar, m)?)?;        // 早晨之星
    m.add_function(wrap_pyfunction!(qtalib::cdlmorningdojistar, m)?)?;    // 早晨十字星
    m.add_function(wrap_pyfunction!(qtalib::cdleveningstar, m)?)?;        // 黄昏之星
    m.add_function(wrap_pyfunction!(qtalib::cdleveningdojistar, m)?)?;    // 黄昏十字星
    m.add_function(wrap_pyfunction!(qtalib::cdlabandonedbaby, m)?)?;      // 弃婴形态
    m.add_function(wrap_pyfunction!(qtalib::cdltristar, m)?)?;            // 三星形态
    
    // 穿透形态
    m.add_function(wrap_pyfunction!(qtalib::cdlpiercing, m)?)?;           // 穿透形态
    m.add_function(wrap_pyfunction!(qtalib::cdldarkcloudcover, m)?)?;     // 乌云盖顶
    
    // 三根K线形态
    m.add_function(wrap_pyfunction!(qtalib::cdl3blackcrows, m)?)?;        // 三只乌鸦
    m.add_function(wrap_pyfunction!(qtalib::cdl3whitesoldiers, m)?)?;     // 三个白武士
    m.add_function(wrap_pyfunction!(qtalib::cdl3inside, m)?)?;            // 三内部上升/下降
    m.add_function(wrap_pyfunction!(qtalib::cdl3outside, m)?)?;           // 三外部上升/下降
    m.add_function(wrap_pyfunction!(qtalib::cdl3linestrike, m)?)?;        // 三线攻击
    m.add_function(wrap_pyfunction!(qtalib::cdl3starsinsouth, m)?)?;      // 南方三星
    m.add_function(wrap_pyfunction!(qtalib::cdlidentical3crows, m)?)?;    // 相同三乌鸦
    
    // 跳空形态
    m.add_function(wrap_pyfunction!(qtalib::cdlgapsidesidewhite, m)?)?;   // 向上跳空并列阳线
    m.add_function(wrap_pyfunction!(qtalib::cdlupsidegap2crows, m)?)?;    // 向上跳空两只乌鸦
    m.add_function(wrap_pyfunction!(qtalib::cdltasukigap, m)?)?;          // 跳空并列阴阳线
    m.add_function(wrap_pyfunction!(qtalib::cdlxsidegap3methods, m)?)?;   // 上升/下降跳空三法
    
    // 特殊形态
    m.add_function(wrap_pyfunction!(qtalib::cdl2crows, m)?)?;             // 两只乌鸦
    m.add_function(wrap_pyfunction!(qtalib::cdladvanceblock, m)?)?;       // 前进阻挡
    m.add_function(wrap_pyfunction!(qtalib::cdlbelthold, m)?)?;           // 捉腰带线
    m.add_function(wrap_pyfunction!(qtalib::cdlbreakaway, m)?)?;          // 脱离形态
    m.add_function(wrap_pyfunction!(qtalib::cdlclosingmarubozu, m)?)?;    // 收盘秃头
    m.add_function(wrap_pyfunction!(qtalib::cdlconcealbabyswall, m)?)?;   // 藏婴吞没
    m.add_function(wrap_pyfunction!(qtalib::cdlcounterattack, m)?)?;      // 反击线
    m.add_function(wrap_pyfunction!(qtalib::cdlhighwave, m)?)?;           // 长影线
    m.add_function(wrap_pyfunction!(qtalib::cdlhikkake, m)?)?;            // 陷阱形态
    m.add_function(wrap_pyfunction!(qtalib::cdlhikkakemod, m)?)?;         // 修正陷阱形态
    m.add_function(wrap_pyfunction!(qtalib::cdlhomingpigeon, m)?)?;       // 家鸽形态
    m.add_function(wrap_pyfunction!(qtalib::cdlinneck, m)?)?;             // 颈内线
    m.add_function(wrap_pyfunction!(qtalib::cdlkicking, m)?)?;            // 反冲形态
    m.add_function(wrap_pyfunction!(qtalib::cdlkickingbylength, m)?)?;    // 由长度决定的反冲形态
    m.add_function(wrap_pyfunction!(qtalib::cdlladderbottom, m)?)?;       // 梯底形态
    m.add_function(wrap_pyfunction!(qtalib::cdllongline, m)?)?;           // 长线形态
    m.add_function(wrap_pyfunction!(qtalib::cdlmarubozu, m)?)?;           // 秃头形态
    m.add_function(wrap_pyfunction!(qtalib::cdlmatchinglow, m)?)?;        // 相同低价
    m.add_function(wrap_pyfunction!(qtalib::cdlmathold, m)?)?;            // 铺垫形态
    m.add_function(wrap_pyfunction!(qtalib::cdlonneck, m)?)?;             // 颈上线
    m.add_function(wrap_pyfunction!(qtalib::cdlrickshawman, m)?)?;        // 黄包车夫
    m.add_function(wrap_pyfunction!(qtalib::cdlrisefall3methods, m)?)?;   // 上升/下降三法
    m.add_function(wrap_pyfunction!(qtalib::cdlseparatinglines, m)?)?;    // 分离线
    m.add_function(wrap_pyfunction!(qtalib::cdlshortline, m)?)?;          // 短线形态
    m.add_function(wrap_pyfunction!(qtalib::cdlspinningtop, m)?)?;        // 纺锤形态
    m.add_function(wrap_pyfunction!(qtalib::cdlstalledpattern, m)?)?;     // 停顿形态
    m.add_function(wrap_pyfunction!(qtalib::cdlsticksandwich, m)?)?;      // 条形三明治
    m.add_function(wrap_pyfunction!(qtalib::cdltakuri, m)?)?;             // 探水竿
    m.add_function(wrap_pyfunction!(qtalib::cdlthrusting, m)?)?;          // 插入形态
    m.add_function(wrap_pyfunction!(qtalib::cdlunique3river, m)?)?;       // 奇特三河床
    
    // ====================================================================
    // 回测功能 (Backtesting) - 量化回测和策略验证
    // ====================================================================
    
    // 添加Backtrade类
    m.add_class::<qbacktrade::Backtrade>()?;
    
    Ok(())
}

