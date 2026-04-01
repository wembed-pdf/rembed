use std::collections::HashMap;
use std::path::Path;

use osmpbf::{Element, ElementReader};

use crate::category::Category;
use crate::classify::classify;
use crate::poi::Poi;

/// Intermediate data collected in a single pass over the PBF file.
#[derive(Default)]
struct RawData {
    /// All node coordinates: id -> (lat, lon)
    node_coords: HashMap<i64, (f64, f64)>,
    /// Node POIs already have coordinates, store them directly.
    node_pois: Vec<Poi>,
    /// Way node references: way_id -> node_ids
    way_refs: HashMap<i64, Vec<i64>>,
    /// Ways that classified as a POI: (way_id, category, name)
    way_pois: Vec<(i64, Category, Option<String>)>,
    /// Relations that classified as a POI: (rel_id, category, name, member_way_ids)
    rel_pois: Vec<(i64, Category, Option<String>, Vec<i64>)>,
}

impl RawData {
    fn merge(mut self, other: RawData) -> RawData {
        self.node_coords.extend(other.node_coords);
        self.node_pois.extend(other.node_pois);
        self.way_refs.extend(other.way_refs);
        self.way_pois.extend(other.way_pois);
        self.rel_pois.extend(other.rel_pois);
        self
    }
}

/// Extract all classifiable POIs from an OSM PBF file in a single pass.
///
/// Collects node coordinates, way references, and all POI-tagged elements,
/// then resolves way/relation centroids in memory.
pub fn extract_pois(path: &Path) -> Result<Vec<Poi>, Box<dyn std::error::Error>> {
    let reader = ElementReader::from_path(path)?;

    let raw = reader.par_map_reduce(
        |element| {
            let mut data = RawData::default();
            match &element {
                Element::Node(node) => {
                    let id = node.id();
                    let lat = node.lat();
                    let lon = node.lon();
                    data.node_coords.insert(id, (lat, lon));
                    let tags: Vec<_> = node.tags().collect();
                    if let Some(category) = classify(tags.iter().map(|(k, v)| (*k, *v))) {
                        let name = tags.iter().find(|(k, _)| *k == "name").map(|(_, v)| v.to_string());
                        data.node_pois.push(Poi {
                            category, lat, lon, x: 0.0, y: 0.0, name, osm_id: id,
                        });
                    }
                }
                Element::DenseNode(node) => {
                    let id = node.id();
                    let lat = node.lat();
                    let lon = node.lon();
                    data.node_coords.insert(id, (lat, lon));
                    let tags: Vec<_> = node.tags().collect();
                    if let Some(category) = classify(tags.iter().map(|(k, v)| (*k, *v))) {
                        let name = tags.iter().find(|(k, _)| *k == "name").map(|(_, v)| v.to_string());
                        data.node_pois.push(Poi {
                            category, lat, lon, x: 0.0, y: 0.0, name, osm_id: id,
                        });
                    }
                }
                Element::Way(way) => {
                    let id = way.id();
                    let node_ids: Vec<i64> = way.refs().collect();
                    let tags: Vec<_> = way.tags().collect();
                    if let Some(category) = classify(tags.iter().map(|(k, v)| (*k, *v))) {
                        let name = tags.iter().find(|(k, _)| *k == "name").map(|(_, v)| v.to_string());
                        data.way_pois.push((id, category, name));
                    }
                    // Store all way refs — relations may reference any way
                    data.way_refs.insert(id, node_ids);
                }
                Element::Relation(rel) => {
                    let tags: Vec<_> = rel.tags().collect();
                    if let Some(category) = classify(tags.iter().map(|(k, v)| (*k, *v))) {
                        let name = tags.iter().find(|(k, _)| *k == "name").map(|(_, v)| v.to_string());
                        let way_ids: Vec<i64> = rel.members()
                            .filter(|m| m.member_type == osmpbf::RelMemberType::Way)
                            .map(|m| m.member_id)
                            .collect();
                        data.rel_pois.push((rel.id(), category, name, way_ids));
                    }
                }
            }
            data
        },
        RawData::default,
        |a, b| a.merge(b),
    )?;

    eprintln!("  {} nodes, {} ways, {} node POIs, {} way POIs, {} relation POIs",
        raw.node_coords.len(), raw.way_refs.len(),
        raw.node_pois.len(), raw.way_pois.len(), raw.rel_pois.len());

    // Start with node POIs (already have coordinates)
    let mut pois = raw.node_pois;

    // Resolve way POI centroids
    for (way_id, category, name) in &raw.way_pois {
        if let Some(node_ids) = raw.way_refs.get(way_id) {
            if let Some((lat, lon)) = centroid(node_ids, &raw.node_coords) {
                pois.push(Poi {
                    category: *category, lat, lon, x: 0.0, y: 0.0,
                    name: name.clone(), osm_id: *way_id,
                });
            }
        }
    }

    // Resolve relation POI centroids
    for (rel_id, category, name, way_ids) in &raw.rel_pois {
        let all_node_ids: Vec<i64> = way_ids.iter()
            .filter_map(|wid| raw.way_refs.get(wid))
            .flat_map(|nids| nids.iter().copied())
            .collect();
        if let Some((lat, lon)) = centroid(&all_node_ids, &raw.node_coords) {
            pois.push(Poi {
                category: *category, lat, lon, x: 0.0, y: 0.0,
                name: name.clone(), osm_id: *rel_id,
            });
        }
    }

    Ok(pois)
}

fn centroid(node_ids: &[i64], coords: &HashMap<i64, (f64, f64)>) -> Option<(f64, f64)> {
    let mut lat_sum = 0.0;
    let mut lon_sum = 0.0;
    let mut count = 0u64;
    for &nid in node_ids {
        if let Some(&(lat, lon)) = coords.get(&nid) {
            lat_sum += lat;
            lon_sum += lon;
            count += 1;
        }
    }
    (count > 0).then(|| (lat_sum / count as f64, lon_sum / count as f64))
}
