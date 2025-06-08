-- Migration: Create repository state and code state tables
-- Up migration

-- Repository state table - tracks git repository state
CREATE TABLE repository_states (
    repo_state_id BIGSERIAL NOT NULL PRIMARY KEY,
    commit_hash CHAR(40) NOT NULL, -- Git commit hash is always 40 characters
    timestamp TIMESTAMPTZ NOT NULL,
    commit_message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure commit hashes are unique
    CONSTRAINT unique_commit_hash UNIQUE (commit_hash)
);

-- Code state table - tracks implementation state of data structures
CREATE TABLE code_states (
    code_state_id BIGSERIAL PRIMARY KEY,
    repo_state_id BIGINT REFERENCES repository_states(repo_state_id) ON DELETE CASCADE,
    checksum CHAR(64) NOT NULL, -- SHA-256 hex string is always 64 characters
    data_structure_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Ensure (checksum, data_structure_name) combinations are unique
    CONSTRAINT unique_code_state UNIQUE (checksum, data_structure_name)
);

-- Create indexes for common queries
CREATE INDEX idx_repository_states_commit_hash ON repository_states (commit_hash);
CREATE INDEX idx_repository_states_timestamp ON repository_states (timestamp);

CREATE INDEX idx_code_states_repo_state_id ON code_states (repo_state_id);
CREATE INDEX idx_code_states_checksum ON code_states (checksum);
CREATE INDEX idx_code_states_data_structure_name ON code_states (data_structure_name);
CREATE INDEX idx_code_states_created_at ON code_states (created_at);

-- Down migration (uncomment when needed)
-- DROP TABLE IF EXISTS code_states;
-- DROP TABLE IF EXISTS repository_states;
