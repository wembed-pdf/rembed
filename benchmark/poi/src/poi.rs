use serde::{Deserialize, Serialize};

use crate::category::Category;

/// A point of interest extracted from OSM data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Poi {
    pub category: Category,
    pub lat: f64,
    pub lon: f64,
    /// Projected x coordinate in meters (equirectangular).
    pub x: f64,
    /// Projected y coordinate in meters (equirectangular).
    pub y: f64,
    pub name: Option<String>,
    pub osm_id: i64,
}
