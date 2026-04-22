pub mod cycle;
pub mod momentum;
pub mod overlap;
pub mod pattern;
pub mod price;
pub mod volatility;
pub mod volume;

use pyo3::prelude::*;

pub fn register_submodule(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), "talib")?;

    parent_module.add_submodule(&child_module)?;
    Ok(())
}
