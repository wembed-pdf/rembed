use rayon::prelude::*;
use sqlx::{Pool, Postgres, Row};
use tokio::runtime;

macro_rules! dispatch_dim {
    ($dim:ident, $file_path:ident, $graph_file_path:ident, $only_last_iteration:expr, $($c_dim:literal,)*) => {
        match  $dim {
            $($c_dim => compute_fscore::<$c_dim>($file_path, $graph_file_path, $only_last_iteration),)*
            _ => panic!("dim {} not covered",$dim),
        }
    };
}

pub async fn compute_missing_fscores(
    pool: Pool<Postgres>,
    only_last_iteration: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tx = pool.begin().await?;

    // Select all result_ids, iteration_ids combinations where no entry exists in the intrinsic_dim table
    let query = "
        SELECT position_results.file_path, position_results.result_id, position_results.embedding_dim, graphs.file_path as graph_file_path
        FROM position_results
        JOIN graphs ON position_results.graph_id = graphs.graph_id
        WHERE NOT EXISTS (
            SELECT 1 FROM fscores 
            WHERE fscores.result_id = position_results.result_id 
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
            let graph_file_path = row.get::<String, _>("graph_file_path");
            (
                format!("{data_directory}/{file_path}"),
                result_id,
                dim as usize,
                format!("{data_directory}/{graph_file_path}"),
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
        .filter(|(file_path, _, _, graph_file_path)| {
            if !std::path::Path::new(file_path).exists() {
                eprintln!("File does not exist: {}", file_path);
                false
            } else if !std::path::Path::new(graph_file_path).exists() {
                eprintln!("Graph file does not exist: {}", graph_file_path);
                false
            } else {
                true
            }
        })
        .collect();

    // Spawn thread to send data to database recieving data from parallel computation via channel

    let (tx, rx) = std::sync::mpsc::channel::<(i64, usize, f64, f64, f64)>();

    std::thread::spawn(move || {
        let rt = runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let pool = pool.clone();
            while let Ok((result_id, iteration_number, recall, prec, fscore)) = rx.recv() {
                sqlx::query(
                    "
                    INSERT INTO fscores (result_id, iteration_number, recall, prec, fscore)
                    VALUES ($1, $2, $3, $4, $5)
                    ",
                )
                .bind(result_id)
                .bind(iteration_number as i64)
                .bind(recall)
                .bind(prec)
                .bind(fscore)
                .execute(&pool)
                .await
                .expect("Failed to insert fscores");
            }
        });
    });

    // Compute fscores in parallel
    positions_files
        .par_iter()
        .map(|row| {
            let file_path = &row.0;
            let result_id = row.1;
            let dim = row.2;
            let graph_file_path = &row.3;
            let fscores = dispatch_dim!(
                dim,
                file_path,
                graph_file_path,
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
            (result_id, fscores)
        })
        .for_each(|(result_id, fscores)| {
            for (iteration_number, recall, prec, fscore) in fscores {
                tx.send((result_id, iteration_number, recall, prec, fscore))
                    .unwrap();
            }
        });
    Ok(())
}

fn compute_fscore<const D: usize>(
    positions_file_path: &str,
    graph_file_path: &str,
    only_last_iteration: bool,
) -> Vec<(usize, f64, f64, f64)> {
    let iterations: rembed::parsing::Iterations<D> =
        rembed::parsing::parse_positions_file(positions_file_path).unwrap();
    let graph = rembed::graph::Graph::parse_from_edge_list_file(graph_file_path, D, D).unwrap();
    if only_last_iteration {
        let last_iteration = iterations.iterations().last().unwrap();
        let embedding = rembed::Embedding {
            positions: last_iteration.positions.iter().cloned().collect(),
            graph: &graph,
        };
        let sprk = rembed::Sprk::new(&embedding);
        let (percision, recall) = rembed::query::Embedder::graph_statistics(&sprk);
        vec![(
            last_iteration.number,
            recall,
            percision,
            2. / (recall.recip() + percision.recip()),
        )]
    } else {
        iterations
            .iterations()
            .iter()
            .map(|iteration| {
                let embedding = rembed::Embedding {
                    positions: iteration.positions.iter().cloned().collect(),
                    graph: &graph,
                };
                let sprk = rembed::Sprk::new(&embedding);
                let (percision, recall) = rembed::query::Embedder::graph_statistics(&sprk);
                (
                    iteration.number,
                    recall,
                    percision,
                    2. / (recall.recip() + percision.recip()),
                )
            })
            .collect()
    }
}
