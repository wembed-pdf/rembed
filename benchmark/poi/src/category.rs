use serde::{Deserialize, Serialize};

/// A POI category extracted from OSM tags.
///
/// Each variant maps to specific OSM tag combinations.
/// Add new variants as needed for your benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    // Transport
    TrainStation,
    BusStop,
    TramStop,
    Parking,

    // Finance
    Atm,
    Bank,

    // Food & Drink
    Restaurant,
    Cafe,
    FastFood,
    Bar,

    // Shopping
    Supermarket,
    Bakery,
    Pharmacy,

    // Health
    Hospital,
    Doctor,

    // Education
    School,
    University,

    // Other
    PlaceOfWorship,
    PostOffice,
    FuelStation,
}

impl Category {
    /// All known categories, for iteration.
    #[allow(dead_code)]
    pub const ALL: &[Category] = &[
        Category::TrainStation,
        Category::BusStop,
        Category::TramStop,
        Category::Parking,
        Category::Atm,
        Category::Bank,
        Category::Restaurant,
        Category::Cafe,
        Category::FastFood,
        Category::Bar,
        Category::Supermarket,
        Category::Bakery,
        Category::Pharmacy,
        Category::Hospital,
        Category::Doctor,
        Category::School,
        Category::University,
        Category::PlaceOfWorship,
        Category::PostOffice,
        Category::FuelStation,
    ];
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
