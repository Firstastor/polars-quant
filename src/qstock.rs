use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct HistoryStock {
    day: String,
    open: String,
    close: String,
    high: String,
    low: String,
    volume: String
}

#[derive(Debug, Deserialize)]
struct InfoStock {
    symbol: Option<String>,
    name: Option<String>,
}

#[pyfunction]
#[pyo3(signature = (stock_code, scale=240, datalen=365*10, timeout=10))]
pub fn history(
    stock_code: String,
    scale: u32,
    datalen: u32,
    timeout: u64,
) -> PyResult<Option<PyDataFrame>> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create tokio runtime: {}",
            e
        ))
    })?;

    let fut = async move {
        let client = Client::new();
        let url = format!(
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData\
            ?symbol={}&scale={}&ma=no&datalen={}",
            stock_code, scale, datalen
        );

        let resp = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(timeout))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let text = r.text().await.unwrap_or_default();
                parse_history(&text, &stock_code)
            }
            Ok(r) => {
                eprintln!("Request failed, status: {}", r.status());
                None
            }
            Err(e) => {
                eprintln!("Request error: {}", e);
                None
            }
        }
    };

    Ok(rt.block_on(fut))
}

#[pyfunction]
#[pyo3(signature = (stock_code, scale=240, datalen=365*10, timeout=10))]
pub fn history_save(
    stock_code: String,
    scale: u32,
    datalen: u32,
    timeout: u64
) -> PyResult<()> {
    let path = format!("{}.parquet", stock_code);
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create tokio runtime: {}",
            e
        ))
    })?;

    rt.block_on(async move {
        let client = Client::new();
        let url = format!(
            "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData\
            ?symbol={}&scale={}&ma=no&datalen={}",
            stock_code, scale, datalen
        );

        let resp = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(timeout))
            .send()
            .await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Request failed with status {}",
                resp.status()
            )));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed == "null" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Empty or null response for symbol {}",
                stock_code
            )));
        }

        let data: Vec<HistoryStock> = serde_json::from_str(&text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Parse error: {}",
                e
            )))?;

        if data.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "No stock data found for {}",
                stock_code
            )));
        }

        let mut df = df![
            "date" => data.iter().map(|x| x.day.clone()).collect::<Vec<_>>(),
            "open" => data.iter().map(|x| x.open.parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>(),
            "close" => data.iter().map(|x| x.close.parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>(),
            "high" => data.iter().map(|x| x.high.parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>(),
            "low" => data.iter().map(|x| x.low.parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>(),
            "volume" => data.iter().map(|x| x.volume.parse::<i64>().unwrap_or(0)).collect::<Vec<_>>()
        ]
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let file = std::fs::File::create(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        ParquetWriter::new(file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut df)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    })
}

fn parse_history(response_text: &str, stock_code: &str) -> Option<PyDataFrame> {
    let trimmed = response_text.trim();
    if trimmed.is_empty() || trimmed == "null" {
        eprintln!("Empty or null response for symbol: {}", stock_code);
        return None;
    }

    let parsed: Result<Vec<HistoryStock>, _> = serde_json::from_str(response_text);

    match parsed {
        Ok(data) => {
            if data.is_empty() {
                eprintln!("No stock data found for {}", stock_code);
                return None;
            }

            // 分别收集每一列
            let dates: Vec<String> = data.iter().map(|x| x.day.clone()).collect();
            let opens: Vec<f64> = data.iter().map(|x| x.open.parse().unwrap_or(0.0)).collect();
            let closes: Vec<f64> = data.iter().map(|x| x.close.parse().unwrap_or(0.0)).collect();
            let highs: Vec<f64> = data.iter().map(|x| x.high.parse().unwrap_or(0.0)).collect();
            let lows: Vec<f64> = data.iter().map(|x| x.low.parse().unwrap_or(0.0)).collect();
            let volumes: Vec<i64> = data.iter().map(|x| x.volume.parse().unwrap_or(0)).collect();

            // 构造 polars DataFrame
            let df = df!(
                "date" => dates,
                "open" => opens,
                "close" => closes,
                "high" => highs,
                "low" => lows,
                "volume" => volumes
            ).ok()?;

            Some(PyDataFrame(df))
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            None
        }
    }
}

#[pyfunction]
pub fn info() -> PyResult<PyDataFrame> {
    let total_pages = 60;
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create tokio runtime: {}",
            e
        ))
    })?;

    let fut = async {
        let client = Client::new();
        let mut all_stocks: Vec<InfoStock> = Vec::new();
        let mut tasks = Vec::new();

        for page in 1..=total_pages {
            let client = client.clone();
            tasks.push(async move {
                let url = format!(
                    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?\
                    page={page}&num=100&sort=symbol&asc=1&node=hs_a&_s_r_a=page"
                );
                
                let response = client.get(&url).send().await;
                match response {
                    Ok(r) if r.status().is_success() => {
                        let text = r.text().await.unwrap_or_default();
                        parse_info(&text)
                    }
                    Ok(r) => {
                        eprintln!("Request failed, status: {}", r.status());
                        None
                    }
                    Err(e) => {
                        eprintln!("Request error: {}", e);
                        None
                    }
                }
            });
        }
        let results = futures::future::join_all(tasks).await;
        for result in results {
            if let Some(stocks) = result {
                all_stocks.extend(stocks);
            }
        }
        all_stocks
    };

    let all_stocks: Vec<InfoStock> = rt.block_on(fut);
    let symbols: Vec<Option<String>> = all_stocks.iter().map(|x| x.symbol.clone()).collect();
    let names: Vec<Option<String>> = all_stocks.iter().map(|x| x.name.clone()).collect();
    let df = df![
        "symbol" => symbols,
        "name" => names
    ].map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("DataFrame error: {}", e)))?;

    Ok(PyDataFrame(df))
}

#[pyfunction]
pub fn info_save(path: String) -> PyResult<()> {
    let total_pages = 60;
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to create tokio runtime: {}",
            e
        ))
    })?;

    let all_stocks: Vec<InfoStock> = rt.block_on(async {
        let client = Client::new();
        let mut all_stocks: Vec<InfoStock> = Vec::new();
        let mut tasks = Vec::new();

        for page in 1..=total_pages {
            let client = client.clone();
            tasks.push(async move {
                let url = format!(
                    "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?\
                     page={page}&num=100&sort=symbol&asc=1&node=hs_a&_s_r_a=page"
                );

                match client.get(&url).send().await {
                    Ok(r) if r.status().is_success() => {
                        let text = r.text().await.unwrap_or_default();
                        let parsed: Result<Vec<InfoStock>, _> = serde_json::from_str(&text);
                        parsed.ok()
                    }
                    _ => None,
                }
            });
        }

        let results = futures::future::join_all(tasks).await;
        for result in results {
            if let Some(stocks) = result {
                all_stocks.extend(stocks);
            }
        }
        all_stocks
    });

    let symbols: Vec<Option<String>> = all_stocks.iter().map(|x| x.symbol.clone()).collect();
    let names: Vec<Option<String>> = all_stocks.iter().map(|x| x.name.clone()).collect();

    let mut df = df![
        "symbol" => symbols,
        "name" => names
    ].map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
        format!("DataFrame error: {}", e)
    ))?;

    let file = std::fs::File::create(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("File create error: {}", e)
        ))?;
    
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut df)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("Parquet write error: {}", e)
        ))?;

    Ok(())
}

fn parse_info(response_text: &str) -> Option<Vec<InfoStock>> {
    let trimmed = response_text.trim();
    if trimmed.is_empty() || trimmed == "null" {
        return None;
    }
    let parsed: Result<Vec<InfoStock>, _> = serde_json::from_str(response_text);
    match parsed {
        Ok(stocks) if !stocks.is_empty() => Some(stocks),
        Ok(_) => None,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            None
        }
    }
}