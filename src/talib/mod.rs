pub mod momentum;
pub mod overlap;
pub mod price;
pub mod volatility;
pub mod volume;

use pyo3::prelude::*;

pub fn register_submodule(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_submodule(&PyModule::new(parent_module.py(), "talib")?)?;
    Ok(())
}
