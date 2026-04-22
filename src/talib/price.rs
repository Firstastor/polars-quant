use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// ====================================================================
// Price Transform - 价格变换 (Alphabetical Order)
// ====================================================================

#[polars_expr(output_type=Float64)]
pub fn avgprice(inputs: &[Series]) -> PolarsResult<Series> {
    let open = inputs[0].cast(&DataType::Float64)?;
    let high = inputs[1].cast(&DataType::Float64)?;
    let low = inputs[2].cast(&DataType::Float64)?;
    let close = inputs[3].cast(&DataType::Float64)?;

    let open = open.f64()?;
    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let n = open.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("avgprice".into(), n);

    for (o, h, l, c) in izip!(
        open.into_iter(),
        high.into_iter(),
        low.into_iter(),
        close.into_iter()
    ) {
        match (o, h, l, c) {
            (Some(o), Some(h), Some(l), Some(c)) => {
                builder.append_value((o + h + l + c) * 0.25);
            }
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn medprice(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;

    let n = high.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("medprice".into(), n);

    for (h, l) in izip!(high.into_iter(), low.into_iter()) {
        match (h, l) {
            (Some(h), Some(l)) => builder.append_value((h + l) * 0.5),
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn typprice(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let n = high.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("typprice".into(), n);

    for (h, l, c) in izip!(high, low, close) {
        match (h, l, c) {
            (Some(h), Some(l), Some(c)) => builder.append_value((h + l + c) / 3.0),
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn wclprice(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;
    let close = inputs[2].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;
    let close = close.f64()?;

    let n = high.len();
    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("wclprice".into(), n);

    for (h, l, c) in izip!(high, low, close) {
        match (h, l, c) {
            (Some(h), Some(l), Some(c)) => builder.append_value((h + l + c * 2.0) * 0.25),
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}
