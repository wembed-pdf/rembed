use chrono::{DateTime, Utc};
use sqlx::{Pool, Postgres};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct RepositoryState {
    pub repo_state_id: i64,
    pub commit_hash: String,
    pub timestamp: DateTime<Utc>,
    pub commit_message: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CodeState {
    pub code_state_id: i64,
    pub repo_state_id: Option<i64>,
    pub checksum: String,
    pub data_structure_name: String,
    pub created_at: DateTime<Utc>,
}

pub struct RepoCodeStateManager {
    pool: Pool<Postgres>,
}

impl RepoCodeStateManager {
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    /// Get the current git repository state
    pub fn get_current_git_state() -> Result<(String, String), Box<dyn std::error::Error>> {
        // Get commit hash
        let commit_output = Command::new("git").args(["rev-parse", "HEAD"]).output()?;

        if !commit_output.status.success() {
            return Err("Failed to get git commit hash".into());
        }

        let commit_hash = String::from_utf8(commit_output.stdout)?.trim().to_string();

        // Get commit message
        let message_output = Command::new("git")
            .args(["log", "-1", "--pretty=format:%s"])
            .output()?;

        if !message_output.status.success() {
            return Err("Failed to get git commit message".into());
        }

        let commit_message = String::from_utf8(message_output.stdout)?.trim().to_string();

        Ok((commit_hash, commit_message))
    }

    /// Get the current git repository state
    pub fn git_dirty() -> Result<bool, Box<dyn std::error::Error>> {
        // Check if repository is dirty (has uncommitted changes)
        let status_output = Command::new("git")
            .args(["status", "--porcelain"])
            .output()?;

        if !status_output.status.success() {
            return Err("Failed to get git status".into());
        }

        Ok(!status_output.stdout.is_empty())
    }

    /// Get or create a repository state record
    pub async fn get_or_create_repo_state(&self) -> Result<RepositoryState, sqlx::Error> {
        let (commit_hash, commit_message) = Self::get_current_git_state()
            .map_err(|e| sqlx::Error::Protocol(format!("Git error: {}", e)))?;

        // Try to find existing repository state
        if let Some(existing) = sqlx::query_as!(
            RepositoryState,
            "SELECT repo_state_id, commit_hash, timestamp, commit_message,  created_at 
             FROM repository_states WHERE commit_hash = $1",
            commit_hash
        )
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(existing);
        }

        // Get commit timestamp
        let timestamp_output = Command::new("git")
            .args(["log", "-1", "--pretty=format:%cI"])
            .output()
            .map_err(|e| sqlx::Error::Protocol(format!("Git timestamp error: {}", e)))?;

        let timestamp_str = String::from_utf8(timestamp_output.stdout)
            .map_err(|e| sqlx::Error::Protocol(format!("Git timestamp parse error: {}", e)))?;

        let timestamp = timestamp_str
            .trim()
            .parse::<DateTime<Utc>>()
            .map_err(|e| sqlx::Error::Protocol(format!("Git timestamp format error: {}", e)))?;

        // Create new repository state
        let repo_state = sqlx::query_as!(
            RepositoryState,
            r#"
            INSERT INTO repository_states (commit_hash, timestamp, commit_message )
            VALUES ($1, $2, $3)
            RETURNING repo_state_id, commit_hash, timestamp, commit_message, created_at
            "#,
            commit_hash,
            timestamp,
            commit_message,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(repo_state)
    }

    /// Get or create a code state record for a data structure
    pub async fn get_or_create_code_state(
        &self,
        data_structure_name: &str,
        checksum: &str,
    ) -> Result<CodeState, sqlx::Error> {
        // Try to find existing code state
        if let Some(existing) = sqlx::query_as!(
            CodeState,
            "SELECT code_state_id, repo_state_id, checksum, data_structure_name, created_at 
             FROM code_states WHERE checksum = $1 AND data_structure_name = $2",
            checksum,
            data_structure_name
        )
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(existing);
        }

        // Get current repository state
        let repo_state = self.get_or_create_repo_state().await?;

        // Create new code state
        let code_state = sqlx::query_as!(
            CodeState,
            r#"
            INSERT INTO code_states (repo_state_id, checksum, data_structure_name)
            VALUES ($1, $2, $3)
            RETURNING code_state_id, repo_state_id, checksum, data_structure_name, created_at
            "#,
            repo_state.repo_state_id,
            checksum,
            data_structure_name
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(code_state)
    }

    /// Get code state by ID
    pub async fn get_code_state(
        &self,
        data_structure_name: &str,
        checksum: &str,
    ) -> Result<Option<CodeState>, sqlx::Error> {
        sqlx::query_as!(
            CodeState,
            "SELECT code_state_id, repo_state_id, checksum, data_structure_name, created_at 
             FROM code_states WHERE checksum = $1 AND data_structure_name = $2",
            checksum,
            data_structure_name
        )
        .fetch_optional(&self.pool)
        .await
    }

    /// List all code states for a data structure
    pub async fn list_code_states_for_structure(
        &self,
        data_structure_name: &str,
    ) -> Result<Vec<CodeState>, sqlx::Error> {
        sqlx::query_as!(
            CodeState,
            "SELECT code_state_id, repo_state_id, checksum, data_structure_name, created_at 
             FROM code_states WHERE data_structure_name = $1 ORDER BY created_at DESC",
            data_structure_name
        )
        .fetch_all(&self.pool)
        .await
    }
}
