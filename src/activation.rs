use std::f64::consts::E;

use serde::Deserialize;

#[derive(Clone, Copy, Debug)]
pub struct Activation {
    pub fx: fn(f64) -> f64,
    pub fdx: fn(f64) -> f64,
}

impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D>(_: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {
        Ok(Activation::default())
    }
}

impl Default for Activation {
    fn default() -> Self {
        Self {
            fx: |x| 1.0 / (1.0 + E.powf(-x)),
            fdx: |x| x * (1.0 - x),
        }
    }
}
