use crate::poi::Poi;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Equirectangular projection centered on the dataset centroid.
///
/// Converts (lat, lon) in degrees to (x, y) in meters.
/// Accurate for regional datasets (< ~500 km extent).
pub struct Projection {
    lat0_rad: f64,
    lon0_rad: f64,
    cos_lat0: f64,
}

impl Projection {
    /// Compute a projection centered on the centroid of the given POIs.
    pub fn from_centroid(pois: &[Poi]) -> Self {
        let n = pois.len() as f64;
        let lat0 = pois.iter().map(|p| p.lat).sum::<f64>() / n;
        let lon0 = pois.iter().map(|p| p.lon).sum::<f64>() / n;
        let lat0_rad = lat0.to_radians();
        let lon0_rad = lon0.to_radians();
        Projection {
            lat0_rad,
            lon0_rad,
            cos_lat0: lat0_rad.cos(),
        }
    }

    /// Project a single (lat, lon) pair to (x, y) in meters.
    pub fn project(&self, lat: f64, lon: f64) -> [f64; 2] {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let x = (lon_rad - self.lon0_rad) * self.cos_lat0 * EARTH_RADIUS_M;
        let y = (lat_rad - self.lat0_rad) * EARTH_RADIUS_M;
        [x, y]
    }
}
