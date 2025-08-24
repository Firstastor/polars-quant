use pyo3::prelude::*;
use pyo3::types::PyDict;
use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct StockItem {
    day: String,
    open: String,
    close: String,
    high: String,
    low: String,
    volume: String,
}

#[pyfunction]
#[pyo3(signature = (stock_code, scale=240, datalen=365*10, timeout=10))]
pub fn history(
    stock_code: String,
    scale: u32,
    datalen: u32,
    timeout: u64,
) -> PyResult<Option<Vec<PyObject>>> {
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

fn parse_history(response_text: &str, stock_code: &str) -> Option<Vec<PyObject>> {
    let trimmed = response_text.trim();
    if trimmed.is_empty() || trimmed == "null" {
        eprintln!("Empty or null response for symbol: {}", stock_code);
        return None;
    }

    let parsed: Result<Vec<StockItem>, _> = serde_json::from_str(&response_text);

    match parsed {
        Ok(data) => {
            if data.is_empty() {
                eprintln!("No stock data found for {}", stock_code);
                return None;
            }

            Python::with_gil(|py| {
                let mut result = Vec::new();
                for item in data {
                    let dict = PyDict::new(py);
                    dict.set_item("symbol", stock_code).unwrap();
                    dict.set_item("date", item.day).unwrap();
                    dict.set_item("open", item.open.parse::<f64>().unwrap_or(0.0)).unwrap();
                    dict.set_item("close", item.close.parse::<f64>().unwrap_or(0.0)).unwrap();
                    dict.set_item("high", item.high.parse::<f64>().unwrap_or(0.0)).unwrap();
                    dict.set_item("low", item.low.parse::<f64>().unwrap_or(0.0)).unwrap();
                    dict.set_item("volume", item.volume.parse::<i64>().unwrap_or(0)).unwrap();

                    result.push(dict.into());
                }
                Some(result)
            })
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            None
        }
    }
}
