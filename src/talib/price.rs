use itertools::izip;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

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

    let length = open.len();

    if length == 0
        || high.len() != length
        || low.len() != length
        || close.len() != length
        || open.null_count() + high.null_count() + low.null_count() + close.null_count() > 0
    {
        return Ok(Float64Chunked::full_null("avgprice".into(), length).into_series());
    }

    let open = open.rechunk();
    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("avgprice".into(), length);

    izip!(
        open.no_null_iter(),
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .for_each(|(open, high, low, close)| builder.append_value((open + high + low + close) * 0.25));

    Ok(builder.finish().into_series())
}

#[polars_expr(output_type=Float64)]
pub fn medprice(inputs: &[Series]) -> PolarsResult<Series> {
    let high = inputs[0].cast(&DataType::Float64)?;
    let low = inputs[1].cast(&DataType::Float64)?;

    let high = high.f64()?;
    let low = low.f64()?;

    let length = high.len();

    if length == 0 || low.len() != length || high.null_count() + low.null_count() > 0 {
        return Ok(Float64Chunked::full_null("medprice".into(), length).into_series());
    }

    let high = high.rechunk();
    let low = low.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("medprice".into(), length);

    izip!(high.no_null_iter(), low.no_null_iter())
        .for_each(|(high, low)| builder.append_value((high + low) * 0.5));

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

    let length = high.len();

    if length == 0
        || low.len() != length
        || close.len() != length
        || high.null_count() + low.null_count() + close.null_count() > 0
    {
        return Ok(Float64Chunked::full_null("typprice".into(), length).into_series());
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("typprice".into(), length);

    izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .for_each(|(high, low, close)| builder.append_value((high + low + close) / 3.0));

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

    let length = high.len();

    if length == 0
        || low.len() != length
        || close.len() != length
        || high.null_count() + low.null_count() + close.null_count() > 0
    {
        return Ok(Float64Chunked::full_null("wclprice".into(), length).into_series());
    }

    let high = high.rechunk();
    let low = low.rechunk();
    let close = close.rechunk();

    let mut builder = PrimitiveChunkedBuilder::<Float64Type>::new("wclprice".into(), length);

    izip!(
        high.no_null_iter(),
        low.no_null_iter(),
        close.no_null_iter()
    )
    .for_each(|(high, low, close)| builder.append_value((high + low + 2.0 * close) / 4.0));

    Ok(builder.finish().into_series())
}
