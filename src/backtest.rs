pub mod metrics;
pub mod sequential;
pub mod vectorized;

use pyo3::prelude::*;

pub fn register_submodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sequential::SequentialBacktester>()?;
    m.add_class::<vectorized::VectorizedBacktester>()?;
    Ok(())
}
