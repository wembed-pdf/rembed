use chrono::{DateTime, Utc};
use sqlx::{Pool, Postgres};

#[derive(Debug, Clone)]
pub struct PositionJob {
    pub job_id: i64,
    pub graph_id: i64,
    pub embedding_dim: i32,
    pub dim_hint: i32,
    pub max_iterations: i32,
    pub seed: i32,
    pub graph_file_path: String,
    pub processed_n: i32,
    pub processed_avg_degree: f64,
}

#[derive(Debug, Clone)]
pub struct PositionResult {
    pub result_id: i64,
    pub graph_id: i64,
    pub embedding_dim: i32,
    pub dim_hint: i32,
    pub max_iterations: i32,
    pub actual_iterations: Option<i32>,
    pub seed: i32,
    pub file_path: String,
    pub checksum: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct JobManager {
    pool: Pool<Postgres>,
    hostname: String,
}

impl JobManager {
    pub fn new(pool: Pool<Postgres>) -> Self {
        let hostname = gethostname::gethostname().to_string_lossy().to_string();
        Self { pool, hostname }
    }

    pub async fn claim_next_job(&self) -> Result<Option<PositionJob>, sqlx::Error> {
        let mut tx = self.pool.begin().await?;
        // Claim next job
        let job = sqlx::query!(
            r#"
            UPDATE position_jobs 
            SET status = 'running', claimed_at = NOW(), claimed_by_hostname = $1
            WHERE job_id = (
                SELECT job_id FROM position_jobs 
                WHERE status = 'pending' 
                ORDER BY embedding_dim,created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED
            )
            RETURNING job_id, graph_id, embedding_dim, dim_hint, max_iterations, seed
            "#,
            self.hostname
        )
        .fetch_optional(&mut *tx)
        .await?;

        if let Some(job) = job {
            let graph_info = sqlx::query!(
                "SELECT file_path, processed_n, processed_avg_degree FROM graphs WHERE graph_id = $1",
                job.graph_id
            ).fetch_one(&mut *tx).await?;

            tx.commit().await?;

            Ok(Some(PositionJob {
                job_id: job.job_id,
                graph_id: job.graph_id,
                embedding_dim: job.embedding_dim,
                dim_hint: job.dim_hint,
                max_iterations: job.max_iterations,
                seed: job.seed,
                graph_file_path: graph_info.file_path,
                processed_n: graph_info.processed_n,
                processed_avg_degree: graph_info.processed_avg_degree,
            }))
        } else {
            tx.rollback().await?;
            Ok(None)
        }
    }

    pub async fn complete_job(
        &self,
        job_id: i64,
        file_path: &str,
        checksum: &str,
        actual_iterations: Option<i32>,
    ) -> Result<(), sqlx::Error> {
        let mut tx = self.pool.begin().await?;

        // Get job details
        let job = sqlx::query!(
            "SELECT graph_id, embedding_dim, dim_hint, max_iterations, seed FROM position_jobs WHERE job_id = $1",
            job_id
        ).fetch_one(&mut *tx).await?;

        // Insert result
        sqlx::query!(
            r#"
            INSERT INTO position_results (graph_id, embedding_dim, dim_hint, max_iterations, actual_iterations, seed, file_path, checksum)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            job.graph_id, job.embedding_dim, job.dim_hint, job.max_iterations, actual_iterations, job.seed, file_path, checksum
        ).execute(&mut *tx).await?;

        // Mark job complete
        sqlx::query!(
            "UPDATE position_jobs SET status = 'completed', completed_at = NOW() WHERE job_id = $1",
            job_id
        )
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }

    pub async fn fail_job(&self, job_id: i64, error: &str) -> Result<(), sqlx::Error> {
        sqlx::query!(
            "UPDATE position_jobs SET status = 'failed', error_message = $1 WHERE job_id = $2",
            error,
            job_id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn create_jobs_for_graph(&self, graph_id: i64) -> Result<i32, sqlx::Error> {
        // let dimensions = [2, 4, 8, 16, 32];
        // let dimensions = [2, 4, 8];
        // let dimensions = [2, 4, 8, 16];
        // let dimensions = [2, 4, 8, 16, 32];
        let dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let max_iterations = 1000;
        let seed = 42; // Fixed seed for reproducibility
        let mut job_count = 0;

        for embedding_dim in dimensions {
            let result = sqlx::query!(
                r#"
                INSERT INTO position_jobs (graph_id, embedding_dim, dim_hint, max_iterations, seed)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (graph_id, embedding_dim, dim_hint, max_iterations, seed) DO NOTHING
                "#,
                graph_id,
                embedding_dim,
                embedding_dim,
                max_iterations,
                seed
            )
            .execute(&self.pool)
            .await?;

            if result.rows_affected() > 0 {
                job_count += 1;
            }
        }

        Ok(job_count)
    }

    pub async fn create_missing_jobs(&self) -> Result<i32, sqlx::Error> {
        let graphs = sqlx::query!(
            "SELECT graph_id FROM graphs where  deg = 15  and alpha > 1000 and n <= 1000000"
        )
        .fetch_all(&self.pool)
        .await?;
        let mut total_created = 0;
        let pb = crate::create_progress_bar(graphs.len());
        for graph in graphs {
            pb.inc(1);
            total_created += self.create_jobs_for_graph(graph.graph_id).await?;
        }
        Ok(total_created)
    }

    // Query methods for results
    pub async fn get_results_for_graph(
        &self,
        graph_id: i64,
    ) -> Result<Vec<PositionResult>, sqlx::Error> {
        let results = sqlx::query!(
            "SELECT * FROM position_results WHERE graph_id = $1 ORDER BY embedding_dim",
            graph_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(results
            .into_iter()
            .map(|row| PositionResult {
                result_id: row.result_id,
                graph_id: row.graph_id,
                embedding_dim: row.embedding_dim,
                dim_hint: row.dim_hint,
                max_iterations: row.max_iterations,
                actual_iterations: row.actual_iterations,
                seed: row.seed,
                file_path: row.file_path,
                checksum: row.checksum,
                created_at: row.created_at.to_utc(),
            })
            .collect())
    }

    pub async fn get_job_stats(&self) -> Result<(i64, i64, i64, i64), sqlx::Error> {
        let result = sqlx::query!(
            r#"
            SELECT 
                COUNT(*) FILTER (WHERE status = 'pending') as pending,
                COUNT(*) FILTER (WHERE status = 'running') as running,
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                COUNT(*) FILTER (WHERE status = 'failed') as failed
            FROM position_jobs
            "#
        )
        .fetch_one(&self.pool)
        .await?;

        Ok((
            result.pending.unwrap_or(0),
            result.running.unwrap_or(0),
            result.completed.unwrap_or(0),
            result.failed.unwrap_or(0),
        ))
    }

    // For all running jobs: hostname, duration_claimed, embedding_dim, n, graph_id
    pub async fn get_running_jobs(
        &self,
    ) -> Result<Vec<(String, String, i32, i32, i64)>, sqlx::Error> {
        let results = sqlx::query!(
            r#"
            SELECT 
                claimed_by_hostname,
                claimed_at,
                embedding_dim,
                n,
                graph_id
            FROM position_jobs
            JOIN graphs USING (graph_id)
            WHERE status = 'running'
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let now = Utc::now();
        let mut running_jobs = Vec::new();
        for result in results {
            let duration_claimed_delta = match result.claimed_at {
                Some(claimed_at) => now.signed_duration_since(claimed_at),
                None => chrono::Duration::zero(),
            };

            let mut duration_claimed = String::new();
            if duration_claimed_delta.num_hours() > 0 {
                duration_claimed.push_str(&format!("{}h ", duration_claimed_delta.num_hours()));
            }
            if duration_claimed_delta.num_minutes() > 0 {
                duration_claimed
                    .push_str(&format!("{}m ", duration_claimed_delta.num_minutes() % 60));
            }
            duration_claimed.push_str(&format!("{}s", duration_claimed_delta.num_seconds() % 60));

            running_jobs.push((
                result.claimed_by_hostname.unwrap_or_default(),
                duration_claimed,
                result.embedding_dim,
                result.n,
                result.graph_id,
            ));
        }

        Ok(running_jobs)
    }
}
