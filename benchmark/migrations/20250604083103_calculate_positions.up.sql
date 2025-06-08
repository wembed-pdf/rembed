-- Add migration script here
-- Job queue table (operational)
CREATE TABLE position_jobs (
    job_id BIGSERIAL PRIMARY KEY,
    graph_id BIGINT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
    embedding_dim INTEGER NOT NULL,
    dim_hint INTEGER NOT NULL,
    max_iterations INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    claimed_by_hostname TEXT,
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT,
    
    CONSTRAINT unique_job_params UNIQUE (graph_id, embedding_dim, dim_hint, max_iterations, seed)
);

-- Results table (permanent data)
CREATE TABLE position_results (
    result_id BIGSERIAL PRIMARY KEY,
    graph_id BIGINT NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
    
    -- Embedding parameters
    embedding_dim INTEGER NOT NULL,
    dim_hint INTEGER NOT NULL,
    max_iterations INTEGER NOT NULL,
    actual_iterations INTEGER,
    seed INTEGER NOT NULL,
    
    -- Output
    file_path TEXT NOT NULL,
    checksum CHAR(64) NOT NULL,
    
    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_result_params UNIQUE (graph_id, embedding_dim, dim_hint, max_iterations, seed)
);

-- Indexes
CREATE INDEX idx_position_jobs_status ON position_jobs (status);
CREATE INDEX idx_position_jobs_pending ON position_jobs (created_at) WHERE status = 'pending';
CREATE INDEX idx_position_jobs_hostname ON position_jobs (claimed_by_hostname) WHERE status = 'running';

CREATE INDEX idx_position_results_graph_id ON position_results (graph_id);
CREATE INDEX idx_position_results_embedding_dim ON position_results (embedding_dim);
CREATE INDEX idx_position_results_params ON position_results (graph_id, embedding_dim, dim_hint);

-- Cleanup function
CREATE OR REPLACE FUNCTION cleanup_stale_jobs(timeout_hours INTEGER DEFAULT 2)
RETURNS INTEGER AS $$
DECLARE
    cleaned_count INTEGER;
BEGIN
    UPDATE position_jobs 
    SET status = 'pending', claimed_at = NULL, claimed_by_hostname = NULL,
        error_message = COALESCE(error_message, '') || ' [Reset due to timeout]'
    WHERE status = 'running' AND claimed_at < NOW() - INTERVAL '1 hour' * timeout_hours;
    
    GET DIAGNOSTICS cleaned_count = ROW_COUNT;
    RETURN cleaned_count;
END;
$$ LANGUAGE plpgsql;
