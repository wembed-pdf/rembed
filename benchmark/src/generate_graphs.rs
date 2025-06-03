

use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgDatabaseError;
use sqlx::{Pool, Postgres};
use std::path::Path;
use std::process::Command;

struct Seed {
    wseed: i32, // weight seed default 12
    pseed: i32, // position seed default 130
    sseed: i32, // sampling seed default 1400
}

pub struct GraphGenerator {
    pub girgs_path: String,
    pub output_path: String,
}

impl GraphGenerator {
    pub fn new(girgs_path: String, output_path: String) -> Self {
        Self {
            girgs_path,
            output_path,
        }
    }

    pub async fn generate(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Connect to database
        let database_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://localhost/rembed".to_string());
        let pool = sqlx::PgPool::connect(&database_url).await?;

        println!("Generating graphs using GIRGs at: {}", self.girgs_path);
        println!("Output will be saved to: {}", self.output_path);

        let n_s = half_log10(1000.0, 1000005.0);
        let seeds = generate_seeds();
        let avg_degrees = generate_avg_degrees();
        let total_graphs = n_s.len() * seeds.len() * avg_degrees.len();

        let pb = ProgressBar::new(total_graphs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Ensure output directory exists
        println!("generating dir");
        std::fs::create_dir_all(&self.output_path)?;

        for seed in seeds {
            for &avg_degree in &avg_degrees {
                for &n in &n_s {
                    let mut tx = pool.begin().await?;
                    // Insert graph record into database first to get graph_id
                    let graph_id = sqlx::query_scalar!(
                        r#"
                        INSERT INTO graphs (n, deg, wseed, pseed, sseed, file_path, checksum)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING graph_id
                        "#,
                        n,
                        avg_degree,
                        seed.wseed,
                        seed.pseed,
                        seed.sseed,
                        "", // temporary, will update after file generation
                        ""  // temporary, will update after checksum calculation
                    )
                    .fetch_one(&mut *tx)
                    .await;

                    let graph_id: i64 = match graph_id {
                        Ok(graph_id) => graph_id,
                        Err(sqlx::Error::Database(x)) => {
                            let error: PgDatabaseError =  *x.downcast();
                            if error.code() == "23505" {
                                continue;
                            }
                            unreachable!()
                            }
            
                                Err(e) => return Err(e.into())
                    };

                    // Generate filename with graph_id
                    let filename = format!(
                        "{}_genhrg_n-{}_deg-{}_wseed-{}_pseed-{}_sseed-{}",
                        graph_id, n, avg_degree, seed.wseed, seed.pseed, seed.sseed
                    );
                    let file_path = format!("{}/{}", self.output_path, filename);

                    pb.set_message(format!("Generating graph {}", graph_id));

                    // Generate the graph file
                    let status = Command::new(&self.girgs_path)
                        .stdout(std::process::Stdio::null())
                        .arg("-n")
                        .arg(n.to_string())
                        .arg("-deg")
                        .arg(avg_degree.to_string())
                        .arg("-file")
                        .arg(&file_path)
                        .arg("-edge")
                        .arg("1")
                        .arg("-wseed")
                        .arg(seed.wseed.to_string())
                        .arg("-pseed")
                        .arg(seed.pseed.to_string())
                        .arg("-sseed")
                        .arg(seed.sseed.to_string())
                        .status()?;
                    let file_path = &format!("{}.txt", file_path);

                    if !status.success() {
                        // Delete the database record if generation failed
                        sqlx::query!("DELETE FROM graphs WHERE graph_id = $1", graph_id)
                            .execute(&mut *tx)
                            .await?;
                        return Err(format!("Failed to generate graph {}", graph_id).into());
                    }

                    // Calculate checksum
                    let checksum = calculate_file_checksum(file_path)?;

                    // Update database record with file path and checksum
                    sqlx::query!(
                        r#"
                        UPDATE graphs 
                        SET file_path = $1, checksum = $2
                        WHERE graph_id = $3
                        "#,
                        file_path,
                        checksum,
                        graph_id,
                    )
                    .execute(&mut *tx)
                    .await?;
                    tx.commit().await.unwrap();

                    pb.inc(1);
                }
            }
        }

        pb.finish_with_message("Graph generation complete");

        // Sync files using rsync
        self.sync_files().await?;

        Ok(())
    }

    async fn sync_files(&self) -> Result<(), Box<dyn std::error::Error>> {
        let sync_destination =
            std::env::var("RSYNC_DESTINATION").expect("Please set the RSYNC_DESTINATION env var") ;
        let sync_source=
            std::env::var("DATA_DIERECTORY").expect("Please set the DATA_DIERECTORY env var") ;

        println!("Syncing files to: {}", sync_destination);

        let status = tokio::process::Command::new("rsync")
            .arg("-rlvz")
            .arg("--progress")
            .arg(sync_source)
            .arg(&sync_destination)
            .status()
            .await?;

        if !status.success() {
            return Err("Rsync failed".into());
        }

        println!("File sync completed successfully");
        Ok(())
    }
}

fn calculate_file_checksum(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let contents = std::fs::read(file_path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

fn half_log10(start: f64, end: f64) -> Vec<i32> {
    // Generate successive values by multiplying by √10 and rounding.
    (0..)
        .scan(start, |state, _| {
            if *state > end {
                return None;
            }
            let current = *state as i32;
            *state *= 10f64.powf(0.5); // multiply by √10 ≈ 3.16227766
            Some(current)
        })
        .collect()
}

fn generate_seeds() -> Vec<Seed> {
    // Generate seeds for the graphs
    let mut seeds = Vec::new();
    for i in 0..3 {
        seeds.push(Seed {
            wseed: 12 + i as i32,
            pseed: 130 + i as i32,
            sseed: 1400 + i as i32,
        });
    }
    seeds
}

fn generate_avg_degrees() -> Vec<i32> {
    let mut avg_degrees = Vec::new();
    for i in (5..20).step_by(5) {
        avg_degrees.push(i);
    }
    avg_degrees
}
