use std::{collections::HashSet, env, path::Path};

use sqlx::PgPool;

// Add this function to main.rs
pub async fn cleanup_orphaned_files(
    pool: &PgPool,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashSet;
    use std::path::Path;

    let data_directory = env::var("DATA_DIRECTORY").unwrap_or("../data/".to_string());
    let data_path = Path::new(&data_directory);

    if !data_path.exists() {
        return Err(format!("Data directory does not exist: {}", data_directory).into());
    }

    println!("Scanning data directory: {}", data_directory);

    // Get all files from filesystem
    let mut all_files = HashSet::new();
    scan_directory(&data_path.join("generated"), &data_path, &mut all_files)?;

    println!("Found {} files in data directory", all_files.len());

    // Get all referenced file paths from database
    let mut referenced_files = HashSet::new();

    // From graphs table
    let graph_files = sqlx::query_scalar!("SELECT file_path FROM graphs WHERE file_path != ''")
        .fetch_all(pool)
        .await?;
    for file in graph_files {
        referenced_files.insert(file);
    }

    // From position_results table
    let position_files =
        sqlx::query_scalar!("SELECT file_path FROM position_results WHERE file_path != ''")
            .fetch_all(pool)
            .await?;
    for file in position_files {
        referenced_files.insert(file);
    }

    // From tests table
    let test_files = sqlx::query_scalar!("SELECT file_path FROM tests WHERE file_path != ''")
        .fetch_all(pool)
        .await?;
    for file in test_files {
        referenced_files.insert(file);
    }

    println!(
        "Found {} files referenced in database",
        referenced_files.len()
    );

    // Find orphaned files
    let orphaned_files: Vec<_> = all_files.difference(&referenced_files).cloned().collect();

    if orphaned_files.is_empty() {
        println!("No orphaned files found!");
        return Ok(());
    }

    // Calculate total size
    let mut total_size = 0u64;
    for file_path in &orphaned_files {
        let full_path = data_path.join(file_path);
        if let Ok(metadata) = std::fs::metadata(&full_path) {
            total_size += metadata.len();
        }
    }

    println!(
        "\nFound {} orphaned files ({:.1} MB):",
        orphaned_files.len(),
        total_size as f64 / 1_000_000.0
    );

    // Show preview (limit to first 20 files)
    for (i, file_path) in orphaned_files.iter().enumerate() {
        if i >= 20 {
            println!("  ... and {} more files", orphaned_files.len() - 20);
            break;
        }
        println!("  {}", file_path);
    }

    if dry_run {
        println!("\nDry run mode - no files were deleted");
        return Ok(());
    }

    // Ask for confirmation
    print!(
        "\nAre you sure you want to delete these {} files? (y/N): ",
        orphaned_files.len()
    );
    std::io::Write::flush(&mut std::io::stdout())?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() != "y" {
        println!("Cancelled");
        return Ok(());
    }

    // Delete files
    let mut deleted_count = 0;
    let mut failed_count = 0;

    for file_path in orphaned_files {
        let full_path = data_path.join(&file_path);
        match std::fs::remove_file(&full_path) {
            Ok(_) => {
                deleted_count += 1;
                if deleted_count % 100 == 0 {
                    println!("Deleted {} files...", deleted_count);
                }
            }
            Err(e) => {
                eprintln!("Failed to delete {}: {}", file_path, e);
                failed_count += 1;
            }
        }
    }

    println!("\nCleanup complete:");
    println!("  {} files deleted", deleted_count);
    if failed_count > 0 {
        println!("  {} files failed to delete", failed_count);
    }

    Ok(())
}

fn scan_directory(
    dir: &Path,
    base_path: &Path,
    files: &mut HashSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            // Get relative path from base directory
            let relative_path = path.strip_prefix(base_path)?;
            files.insert(relative_path.to_string_lossy().to_string());
        } else if path.is_dir() {
            // Recursively scan subdirectories
            scan_directory(&path, base_path, files)?;
        }
    }
    Ok(())
}
