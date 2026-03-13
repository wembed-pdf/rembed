use rayon::prelude::*;
use sqlx::{Pool, Postgres, Row};
use tokio::runtime;

macro_rules! dispatch_dim {
    ($dim:ident, $file_path:ident, $only_last_iteration:expr, $($c_dim:literal,)*) => {
        match  $dim {
            $($c_dim => compute_intrinsic_dimension::<$c_dim>($file_path, $only_last_iteration),)*
            _ => panic!("dim {} not covered",$dim),
        }
    };
}

pub async fn compute_missing_intrinsic_dimensions(
    pool: Pool<Postgres>,
    only_last_iteration: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tx = pool.begin().await?;

    // Select all result_ids, iteration_ids combinations where no entry exists in the intrinsic_dim table
    let query = "
        SELECT file_path, result_id, embedding_dim
        FROM position_results
        WHERE NOT EXISTS (
            SELECT 1 FROM intrinsic_dim 
            WHERE intrinsic_dim.result_id = position_results.result_id 
        )
        ";

    let rows = sqlx::query(query).fetch_all(&mut *tx).await?;

    let data_directory = std::env::var("DATA_DIRECTORY").unwrap_or(String::from("../data/"));

    let positions_files = rows
        .into_iter()
        .map(|row| {
            let file_path = row.get::<String, _>("file_path");
            let result_id = row.get::<i64, _>("result_id");
            let dim = row.get::<i32, _>("embedding_dim");
            (
                format!("{data_directory}/{file_path}"),
                result_id,
                dim as usize,
            )
        })
        .collect::<Vec<_>>();

    // Assert that all files exist
    // positions_files.par_iter().for_each(|(file_path, _, _)| {
    //     if !std::path::Path::new(file_path).exists() {
    //         panic!("File does not exist: {}", file_path);
    //     }
    // });

    // Filter out files that do not exist
    let positions_files: Vec<_> = positions_files
        .par_iter()
        .filter(|(file_path, _, _)| {
            if !std::path::Path::new(file_path).exists() {
                eprintln!("File does not exist: {}", file_path);
                false
            } else {
                true
            }
        })
        .collect();

    // Spawn thread to send data to database recieving data from parallel computation via channel

    let (tx, rx) = std::sync::mpsc::channel::<(i64, usize, f64)>();

    std::thread::spawn(move || {
        let rt = runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let pool = pool.clone();
            while let Ok((result_id, iteration_number, intrinsic_dim)) = rx.recv() {
                sqlx::query(
                    "
                    INSERT INTO intrinsic_dim (result_id, iteration_number, intrinsic_dimension)
                    VALUES ($1, $2, $3)
                    ",
                )
                .bind(result_id)
                .bind(iteration_number as i64)
                .bind(intrinsic_dim)
                .execute(&pool)
                .await
                .expect("Failed to insert intrinsic dimension");
            }
        });
    });

    // Compute intrinsic dimensions in parallel
    positions_files
        .par_iter()
        .map(|row| {
            let file_path = &row.0;
            let result_id = row.1;
            let dim = row.2;
            let intrinsic_dims = dispatch_dim!(
                dim,
                file_path,
                only_last_iteration,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                32,
            );
            (result_id, intrinsic_dims, only_last_iteration)
        })
        .for_each(|(result_id, intrinsic_dims, only_last_iteration)| {
            for (iteration_number, intrinsic_dim) in intrinsic_dims {
                tx.send((result_id, iteration_number, intrinsic_dim))
                    .unwrap();
            }
        });
    Ok(())
}

fn compute_intrinsic_dimension<const D: usize>(
    file_path: &str,
    only_last_iteration: bool,
) -> Vec<(usize, f64)> {
    let iterations: rembed::parsing::Iterations<D> =
        rembed::parsing::parse_positions_file(file_path).unwrap();
    if only_last_iteration {
        let last_iteration = iterations.iterations().last().unwrap();
        let intrinsic_dim =
            rembed::intrinsic_dimension::intrinsic_dimension(last_iteration.positions.as_ref());
        return vec![(last_iteration.number, intrinsic_dim)];
    }
    iterations
        .iterations()
        .iter()
        .map(|iteration| {
            (
                iteration.number,
                rembed::intrinsic_dimension::intrinsic_dimension(iteration.positions.as_ref()),
            )
        })
        .collect()
}
