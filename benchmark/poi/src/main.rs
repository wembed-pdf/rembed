mod category;
mod classify;
mod extract;
mod poi;
mod project;

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use clap::Parser;

use crate::category::Category;
use crate::extract::extract_pois;
use crate::project::Projection;

#[derive(Parser)]
#[command(name = "poi", about = "Extract POIs from OSM PBF files")]
struct Cli {
    /// Path to the .osm.pbf input file.
    input: PathBuf,

    /// Output directory for CSV files (one per category).
    #[arg(short, long, default_value = "output")]
    output: PathBuf,

    /// Only extract these categories (comma-separated). If omitted, extract all.
    #[arg(short, long, value_delimiter = ',')]
    categories: Option<Vec<String>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    eprintln!("Reading OSM data from {:?}...", cli.input);
    let mut pois = extract_pois(&cli.input)?;
    eprintln!("Extracted {} POIs", pois.len());

    // Project lat/lon to Euclidean coordinates
    let projection = Projection::from_centroid(&pois);
    for poi in &mut pois {
        let [x, y] = projection.project(poi.lat, poi.lon);
        poi.x = x;
        poi.y = y;
    }

    // Group by category
    let mut by_category: HashMap<Category, Vec<_>> = HashMap::new();
    for poi in &pois {
        by_category.entry(poi.category).or_default().push(poi);
    }

    // Print summary
    let mut counts: Vec<_> = by_category.iter().map(|(c, v)| (*c, v.len())).collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("\nCategory counts:");
    for (cat, count) in &counts {
        eprintln!("  {cat:<20} {count:>6}");
    }

    // Filter categories if requested
    let filter: Option<Vec<String>> = cli.categories;

    // Write CSV files
    std::fs::create_dir_all(&cli.output)?;

    for (category, pois) in &by_category {
        let cat_name = format!("{category:?}").to_lowercase();
        if let Some(ref filter) = filter {
            if !filter.iter().any(|f| f.to_lowercase() == cat_name) {
                continue;
            }
        }

        let path = cli.output.join(format!("{cat_name}.csv"));
        let file = File::create(&path)?;
        let mut w = BufWriter::new(file);
        write_csv(&mut w, pois)?;
        eprintln!("Wrote {} POIs to {:?}", pois.len(), path);
    }

    Ok(())
}

fn write_csv(
    w: &mut impl std::io::Write,
    pois: &[&crate::poi::Poi],
) -> Result<(), Box<dyn std::error::Error>> {
    // writeln!(w, "osm_id,lat,lon,x,y,name")?;
    for poi in pois {
        let name = poi.name.as_deref().unwrap_or("");
        // Escape name for CSV (quote if it contains comma or quote)
        let name_escaped = if name.contains(',') || name.contains('"') || name.contains('\n') {
            format!("\"{}\"", name.replace('"', "\"\""))
        } else {
            name.to_string()
        };
        // writeln!(
        //     w,
        //     "{},{},{},{},{},{}",
        //     poi.osm_id, poi.lat, poi.lon, poi.x, poi.y, name_escaped
        // )?;
        writeln!(w, "{},{}", poi.x, poi.y)?;
    }
    Ok(())
}
