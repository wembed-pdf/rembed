use std::path::Path;

use osmpbf::{Element, ElementReader};

use crate::classify::classify;
use crate::poi::Poi;

/// Extract all classifiable POIs from an OSM PBF file.
///
/// Uses osmpbf's parallel reader to process blocks concurrently.
/// Only nodes are extracted — ways/relations representing POIs (e.g., building
/// outlines) are ignored for now since we only need point coordinates.
/// The x/y fields are left at 0.0; call `project_pois` afterwards.
pub fn extract_pois(path: &Path) -> Result<Vec<Poi>, Box<dyn std::error::Error>> {
    let reader = ElementReader::from_path(path)?;

    let pois = reader.par_map_reduce(
        |element| {
            let mut local = Vec::new();
            let (tags, lat, lon, id) = match &element {
                Element::Node(node) => (node.tags().collect::<Vec<_>>(), node.lat(), node.lon(), node.id()),
                Element::DenseNode(node) => (node.tags().collect::<Vec<_>>(), node.lat(), node.lon(), node.id()),
                _ => return local,
            };
            if let Some(category) = classify(tags.iter().map(|(k, v)| (*k, *v))) {
                let name = tags
                    .iter()
                    .find(|(k, _)| *k == "name")
                    .map(|(_, v)| v.to_string());
                local.push(Poi {
                    category,
                    lat,
                    lon,
                    x: 0.0,
                    y: 0.0,
                    name,
                    osm_id: id,
                });
            }
            local
        },
        Vec::new,
        |mut a, b| {
            a.extend(b);
            a
        },
    )?;

    Ok(pois)
}
