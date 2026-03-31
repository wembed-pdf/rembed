use crate::category::Category;

/// Classify an OSM element into a POI category based on its tags.
///
/// Takes an iterator of (key, value) tag pairs and returns the first matching category.
/// OSM elements can have many tags; we check the most specific ones first.
pub fn classify<'a>(tags: impl Iterator<Item = (&'a str, &'a str)>) -> Option<Category> {
    let mut amenity = None;
    let mut shop = None;
    let mut railway = None;
    let mut highway = None;
    let mut public_transport = None;

    for (k, v) in tags {
        match k {
            "amenity" => amenity = Some(v),
            "shop" => shop = Some(v),
            "railway" => railway = Some(v),
            "highway" => highway = Some(v),
            "public_transport" => public_transport = Some(v),
            _ => {}
        }
    }

    // Railway / public transport (check first — train stations also have amenity tags)
    match railway {
        Some("station" | "halt") => return Some(Category::TrainStation),
        Some("tram_stop") => return Some(Category::TramStop),
        _ => {}
    }
    if public_transport == Some("station") && railway.is_some() {
        return Some(Category::TrainStation);
    }

    // Highway
    if highway == Some("bus_stop") {
        return Some(Category::BusStop);
    }
    if public_transport == Some("stop_position") || public_transport == Some("platform") {
        // Could be bus or tram — without railway tag, assume bus
        return Some(Category::BusStop);
    }

    // Amenity
    match amenity {
        Some("atm") => return Some(Category::ATM),
        Some("bank") => return Some(Category::Bank),
        Some("restaurant") => return Some(Category::Restaurant),
        Some("cafe") => return Some(Category::Cafe),
        Some("fast_food") => return Some(Category::FastFood),
        Some("bar" | "pub") => return Some(Category::Bar),
        Some("pharmacy") => return Some(Category::Pharmacy),
        Some("hospital") => return Some(Category::Hospital),
        Some("doctors") => return Some(Category::Doctor),
        Some("school") => return Some(Category::School),
        Some("university") => return Some(Category::University),
        Some("place_of_worship") => return Some(Category::PlaceOfWorship),
        Some("post_office") => return Some(Category::PostOffice),
        Some("fuel") => return Some(Category::FuelStation),
        Some("parking") => return Some(Category::Parking),
        _ => {}
    }

    // Shop
    match shop {
        Some("supermarket") => return Some(Category::Supermarket),
        Some("bakery") => return Some(Category::Bakery),
        _ => {}
    }

    None
}
